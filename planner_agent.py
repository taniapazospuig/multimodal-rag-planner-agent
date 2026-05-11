"""
Minimal planner agent with four capabilities:
- C1 TopicSummary (topics studied in course/week from retrieved evidence)
- C2 DurationEstimator (estimate study time for a course-related task)
- C3 FreeSlotFinder (find open slots in a target week/day)
- C4 StudyEnvironmentRecommender (suggest where to do a task)

This module intentionally reuses already-built retrieval artifacts:
- Chroma DB in `chroma_db` (dense index)
- BM25 corpus JSONL from `Settings.text_bm25_path`

It does not perform indexing or backfill at runtime.
"""

from __future__ import annotations

import base64
import csv
import json
import logging
import re
import warnings
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, List, TypedDict

import chromadb
import open_clip
import torch
from rank_bm25 import BM25Okapi
from urllib3.exceptions import NotOpenSSLWarning

from langchain.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from config import LLMBackend, RAGPipelineMode, Settings, TextRetrievalStrategy, load_settings
from text_tokenization import LexicalTokenizerConfig, tokenize_for_bm25


warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="google")
warnings.filterwarnings(
    "ignore",
    message=r".*You are sending unauthenticated requests to the HF Hub.*",
)
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)

_BASE_DIR = Path(__file__).resolve().parent


class AgentState(TypedDict):
    """State carried between LangGraph nodes."""

    messages: List[BaseMessage]


def load_courses(path: Path) -> list[dict[str, str]]:
    """Load course metadata used for optional automatic course filtering."""
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    return [dict(row) for row in rows]


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load a metadata CSV into memory as row dictionaries."""
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    return [dict(row) for row in rows]


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """Load JSONL rows from disk as dictionaries."""
    if not path.exists():
        return []
    output: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                output.append(payload)
    return output


def get_document_rows() -> list[dict[str, str]]:
    """Return cached KB document metadata rows."""
    global _DOCUMENT_ROWS
    if _DOCUMENT_ROWS is None:
        _DOCUMENT_ROWS = _load_csv_rows(DOCUMENTS_CSV_PATH)
    return _DOCUMENT_ROWS


def get_study_environment_rows() -> list[dict[str, Any]]:
    """Return cached study-environment metadata rows."""
    global _STUDY_ENVIRONMENT_ROWS
    if _STUDY_ENVIRONMENT_ROWS is None:
        _STUDY_ENVIRONMENT_ROWS = _load_jsonl_rows(STUDY_ENVIRONMENTS_JSONL_PATH)
    return _STUDY_ENVIRONMENT_ROWS


COURSES_CSV_PATH = _BASE_DIR / "data" / "kb" / "metadata" / "courses.csv"
COURSES = load_courses(COURSES_CSV_PATH)

_SETTINGS: Settings | None = None
_LLM = None
_ESTIMATOR_LLM = None
_RETRIEVER: "CourseRetriever | None" = None
_OPENCLIP_BACKBONE: "OpenCLIPBackbone | None" = None
_DOCUMENT_ROWS: list[dict[str, str]] | None = None
_STUDY_ENVIRONMENT_ROWS: list[dict[str, Any]] | None = None

DOCUMENTS_CSV_PATH = _BASE_DIR / "data" / "kb" / "metadata" / "documents.csv"
STUDY_ENVIRONMENTS_JSONL_PATH = _BASE_DIR / "data" / "kb" / "metadata" / "study_environments.jsonl"

DAY_TO_ABBR = {
    "monday": "Mon",
    "mon": "Mon",
    "tuesday": "Tue",
    "tue": "Tue",
    "wednesday": "Wed",
    "wed": "Wed",
    "thursday": "Thu",
    "thu": "Thu",
    "friday": "Fri",
    "fri": "Fri",
    "saturday": "Sat",
    "sat": "Sat",
    "sunday": "Sun",
    "sun": "Sun",
}


def get_settings() -> Settings:
    """Lazily load runtime settings once per process."""
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = load_settings()
    return _SETTINGS


def _trace_log(message: str, *args: Any) -> None:
    """Emit verbose pipeline trace logs when DEBUG_TRACE_ENABLED is set."""
    if get_settings().debug_trace_enabled:
        logging.getLogger(__name__).info("[trace] " + message, *args)


class OpenCLIPBackbone:
    """OpenCLIP text encoder used to query existing dense vectors in Chroma."""

    def __init__(self, settings: Settings):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, _ = open_clip.create_model_and_transforms(
            settings.open_clip_model,
            pretrained=settings.open_clip_pretrained,
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(settings.open_clip_model)

    def encode_text(self, texts: list[str]) -> list[list[float]]:
        """Encode text prompts into dense vectors."""
        with torch.no_grad():
            tokens = self.tokenizer(texts).to(self.device)
            embeddings = self.model.encode_text(tokens).cpu().numpy().tolist()
        return embeddings


def get_openclip_backbone() -> OpenCLIPBackbone:
    """Return the singleton OpenCLIP backbone instance."""
    global _OPENCLIP_BACKBONE
    if _OPENCLIP_BACKBONE is None:
        _OPENCLIP_BACKBONE = OpenCLIPBackbone(get_settings())
    return _OPENCLIP_BACKBONE


def _normalize_for_match(text: str) -> str:
    """Normalize text for robust substring/word matching."""
    normalized = text.lower().strip().replace("&", " and ")
    normalized = re.sub(r"[^\w\s-]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _term_in_query(term: str, normalized_query: str) -> bool:
    """Match short terms as full words, longer terms as substrings."""
    if not term:
        return False
    if len(term) <= 4 and " " not in term:
        return re.search(rf"\b{re.escape(term)}\b", normalized_query) is not None
    return term in normalized_query


def _detect_course_filter(query: str) -> str:
    """Infer best-matching course slug from query using metadata aliases."""
    normalized_query = _normalize_for_match(query)
    scores: dict[str, int] = defaultdict(int)

    for row in COURSES:
        slug = str(row.get("course", "")).strip().lower()
        if not slug:
            continue

        code = _normalize_for_match(str(row.get("code", "")))
        title = _normalize_for_match(str(row.get("title", "")))
        aliases = str(row.get("aliases", "")).strip()

        if code and _term_in_query(code, normalized_query):
            scores[slug] += 5
        if len(title) >= 5 and _term_in_query(title, normalized_query):
            scores[slug] += 5
        if aliases:
            for alias in aliases.split("|"):
                alias_norm = _normalize_for_match(alias)
                if alias_norm and _term_in_query(alias_norm, normalized_query):
                    scores[slug] += 3
                    break

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    if not ranked:
        return ""
    best_slug, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0
    if best_score >= 5 and (best_score - second_score) >= 2:
        return best_slug
    return ""


def _known_course_slugs() -> frozenset[str]:
    """Dataset course ids as stored in chunk metadata (hyphenated slugs)."""
    return frozenset(
        str(row.get("course", "")).strip().lower()
        for row in COURSES
        if str(row.get("course", "")).strip()
    )


def _canonicalize_course_slug(raw: str) -> str:
    """Normalize LLM/tool course strings toward metadata slugs (e.g. operations-research)."""
    normalized = raw.strip().lower()
    if not normalized:
        return ""
    normalized = re.sub(r"[\s_]+", "-", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
    return normalized


def _detect_course_from_composite(raw_value: str) -> str:
    """Recover known course slugs embedded in larger composite strings."""
    canonical = _canonicalize_course_slug(raw_value)
    if not canonical:
        return ""
    canonical_compact = canonical.replace("-", "")
    known = sorted(_known_course_slugs(), key=len, reverse=True)
    for slug in known:
        if slug in canonical:
            return slug
        if slug.replace("-", "") in canonical_compact:
            return slug
    return ""


def _llm_detect_course_enum(raw_course_filter: str, query: str) -> str:
    """Fallback course classifier constrained to known enum values."""
    text = " ".join(part for part in [raw_course_filter.strip(), query.strip()] if part).strip()
    if not text:
        return ""
    allowed = ("operations-research", "high-dimensional-data", "generative-ai", "unknown")
    try:
        raw_response = _render_assistant_content(
            get_reasoning_llm().invoke(
                [
                    SystemMessage(
                        content=(
                            "Classify the course from the user text.\n"
                            "Allowed course values only: operations-research, high-dimensional-data, "
                            "generative-ai, unknown.\n"
                            "Return strict JSON only with keys:\n"
                            '- course: one of the allowed values\n'
                            '- confidence: one of high, medium, low'
                        )
                    ),
                    HumanMessage(content=f"text={text}"),
                ]
            ).content
        )
    except Exception as exc:
        _trace_log("course_fallback.llm error=%s", exc)
        return ""

    payload = _extract_first_json_object(raw_response) or {}
    course = str(payload.get("course", "")).strip().lower()
    confidence = str(payload.get("confidence", "")).strip().lower()
    _trace_log("course_fallback.llm course=%r confidence=%r text=%r", course, confidence, text)
    if course not in allowed:
        return ""
    if course == "unknown":
        return ""
    if confidence == "high" or course != "unknown":
        return course
    return ""


def _resolve_course_filter(
    raw_course_filter: str,
    query: str,
    *,
    filter_enabled: bool,
) -> str:
    """
    Resolve course slug for Chroma/BM25 metadata filters.

    Tool calls often pass human phrases ("operations research") while indexes store
    slugs ("operations-research"). Match slugs first, then reuse alias scoring.
    """
    raw = raw_course_filter.strip()
    canonical = _canonicalize_course_slug(raw)
    known = _known_course_slugs()

    if canonical and canonical in known:
        return canonical

    composite_match = _detect_course_from_composite(raw)
    if composite_match:
        _trace_log("course_resolve composite_match raw=%r -> %r", raw, composite_match)
        return composite_match

    if raw and filter_enabled:
        detected = _detect_course_filter(raw)
        if detected:
            return detected

    if filter_enabled and query.strip():
        query_detected = _detect_course_filter(query)
        if query_detected:
            return query_detected
        llm_detected = _llm_detect_course_enum(raw, query)
        if llm_detected:
            return llm_detected

    return canonical


_WEEK_WORD_TO_NUM: dict[str, int] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}


def _canonicalize_week_filter(raw_value: str, allow_bare_number: bool = False) -> str:
    """Normalize week-like inputs to canonical week tags (for example, week-04)."""
    normalized = _normalize_for_match(raw_value)
    if not normalized:
        return ""

    if re.fullmatch(r"easter[\s-]*week", normalized):
        return "easter-week"

    numeric_match = re.search(r"\b(?:week|wk|w)[\s\-_]*0*(\d{1,2})\b", normalized)
    if numeric_match:
        week_num = int(numeric_match.group(1))
        return f"week-{week_num:02d}" if week_num > 0 else ""

    word_options = "|".join(_WEEK_WORD_TO_NUM.keys())
    word_match = re.search(rf"\b(?:week|wk|w)[\s\-_]*({word_options})\b", normalized)
    if word_match:
        return f"week-{_WEEK_WORD_TO_NUM[word_match.group(1)]:02d}"

    if allow_bare_number:
        bare_match = re.fullmatch(r"0*(\d{1,2})", normalized)
        if bare_match:
            week_num = int(bare_match.group(1))
            return f"week-{week_num:02d}" if week_num > 0 else ""
    return ""


def _detect_week_filter(query: str) -> str:
    """Infer canonical week filter (week-XX) from free-form query text."""
    return _canonicalize_week_filter(query, allow_bare_number=False)


def _detect_day_filter(query: str) -> str:
    """Infer canonical day abbreviation from free-form query text."""
    normalized_query = _normalize_for_match(query)
    for label, abbr in DAY_TO_ABBR.items():
        if _term_in_query(label, normalized_query):
            return abbr
    return ""


def _normalize_day_filter(value: str) -> str:
    """Normalize user day filter to Mon/Tue/... format."""
    normalized = _normalize_for_match(value)
    return DAY_TO_ABBR.get(normalized, "")


def _resolve_week_filter(raw_week_filter: str, query: str = "") -> str:
    """Resolve week filter from explicit input or query text."""
    candidate = _canonicalize_week_filter(raw_week_filter, allow_bare_number=True)
    if candidate:
        return candidate
    return _detect_week_filter(query)

def _parse_hhmm_to_minutes(value: str) -> int | None:
    """Parse HH:MM into minutes since midnight."""
    raw = value.strip()
    try:
        parsed = datetime.strptime(raw, "%H:%M")
    except ValueError:
        return None
    return parsed.hour * 60 + parsed.minute


def _minutes_to_hhmm(total_minutes: int) -> str:
    """Render minutes since midnight as HH:MM."""
    clamped = max(0, min(24 * 60, int(total_minutes)))
    hours = clamped // 60
    minutes = clamped % 60
    if hours == 24 and minutes == 0:
        return "24:00"
    return f"{hours:02d}:{minutes:02d}"


def _safe_int(value: Any, default: int) -> int:
    """Best-effort integer parsing with fallback."""
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _clamp_int(value: int, low: int, high: int) -> int:
    """Clamp an integer to a closed range."""
    return max(low, min(high, int(value)))


def _extract_first_json_object(raw_text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from raw model output."""
    text = raw_text.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return None
    return None


def _truncate_for_log(value: Any, max_len: int = 280) -> str:
    """Render a compact single-line preview for logs."""
    text = str(value).replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."


def _log_missing_required_tool_args(tool_calls: Any) -> None:
    """Warn only when required tool-call args are missing."""
    if not isinstance(tool_calls, list):
        return
    required_by_tool = {
        "kb_course_retrieval": ("query",),
        "find_free_slots": ("week_filter",),
    }
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        name = str(call.get("name", "")).strip()
        args = call.get("args", {})
        if not name or not isinstance(args, dict):
            continue

        if name == "estimate_study_duration":
            task_query = str(args.get("task_query", "")).strip()
            query = str(args.get("query", "")).strip()
            if not task_query and not query:
                logging.getLogger(__name__).warning(
                    "tool_call missing_required_args tool=%s missing_any_of=task_query|query",
                    name,
                )
            continue

        if name in {"recommend_study_environment", "assess_study_environment_fit"}:
            task_query = str(args.get("task_query", "")).strip()
            query = str(args.get("query", "")).strip()
            if not task_query and not query:
                logging.getLogger(__name__).warning(
                    "tool_call missing_required_args tool=%s missing_any_of=task_query|query",
                    name,
                )
            continue

        required = required_by_tool.get(name, ())
        missing = [arg for arg in required if not str(args.get(arg, "")).strip()]
        if missing:
            logging.getLogger(__name__).warning(
                "tool_call missing_required_args tool=%s missing=%s",
                name,
                ",".join(missing),
            )


def _rrf_fuse(dense_ranked_ids: list[str], bm25_ranked_ids: list[str], k: int) -> dict[str, float]:
    """Fuse dense and lexical rankings with Reciprocal Rank Fusion."""
    scores: dict[str, float] = defaultdict(float)
    for rank, chunk_id in enumerate(dense_ranked_ids, start=1):
        scores[chunk_id] += 1.0 / (k + rank)
    for rank, chunk_id in enumerate(bm25_ranked_ids, start=1):
        scores[chunk_id] += 1.0 / (k + rank)
    return dict(scores)


class CourseRetriever:
    """
    Unified retrieval over existing indexes for C1/C2/C3.
    Reads from Chroma `text_chunks`, Chroma `planner_images`, and BM25 JSONL.
    Text ranking strategy is controlled by TEXT_RETRIEVAL_STRATEGY (hybrid|dense_only).
    RAG_MODE controls whether image retrieval/fusion is applied
    (multimodal_retrieval_mllm only) or text hits are returned directly.
    Optional course/week metadata filtering is applied via Chroma where and BM25 post-filter.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.backbone = get_openclip_backbone()

        chroma_path = _BASE_DIR / "chroma_db"
        self.client = chromadb.PersistentClient(path=str(chroma_path))
        self.collection = self.client.get_or_create_collection(
            name=settings.text_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.image_collection = self.client.get_or_create_collection(
            name="planner_images",
            metadata={"hnsw:space": "cosine"},
        )

        self.by_id: dict[str, dict[str, Any]] = {}
        self.bm25_rows: list[dict[str, Any]] = []
        self.bm25: BM25Okapi | None = None
        self.lexical_tokenizer_config = LexicalTokenizerConfig()
        self._load_bm25_rows()

    def _load_bm25_rows(self) -> None:
        """Load BM25 rows and lexical tokenizer settings from disk."""
        bm25_path = _BASE_DIR / self.settings.text_bm25_path
        if not bm25_path.exists():
            return

        with bm25_path.open("r", encoding="utf-8") as jsonl_file:
            for line in jsonl_file:
                raw = line.strip()
                if not raw:
                    continue
                row = json.loads(raw)
                chunk_id = str(row.get("chunk_id", "")).strip()
                if not chunk_id:
                    continue
                self.bm25_rows.append(row)
                self.by_id[chunk_id] = row

        if not self.bm25_rows:
            return

        first_meta = self.bm25_rows[0].get("metadata", {})
        lexical_payload = first_meta.get("lexical_tokenizer") if isinstance(first_meta, dict) else None
        self.lexical_tokenizer_config = LexicalTokenizerConfig.from_dict(
            lexical_payload if isinstance(lexical_payload, dict) else None,
            fallback=LexicalTokenizerConfig(),
        )
        corpus_tokens = [row.get("tokens", []) for row in self.bm25_rows]
        self.bm25 = BM25Okapi(corpus_tokens)

    @staticmethod
    def _make_where(course_filter: str, week_filter: str) -> dict[str, Any] | None:
        """Build Chroma ``where`` for metadata filtering (None = no filter, search all)."""
        # Each clause is an exact match on indexed metadata fields. Combined with $and
        # when both course and week are present. Callers pass "" to skip a dimension.
        clauses: list[dict[str, str]] = []
        if course_filter:
            clauses.append({"course": course_filter})
        if week_filter:
            clauses.append({"week": week_filter})
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def _dense_search(self, query: str, where: dict[str, Any] | None, n_results: int) -> list[str]:
        """Run dense retrieval against existing Chroma vectors."""
        query_embedding = self.backbone.encode_text([query])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=max(1, n_results),
            where=where,
        )
        return [str(chunk_id) for chunk_id in ((results.get("ids") or [[]])[0] or [])]

    def _bm25_search(self, query: str, course_filter: str, week_filter: str, n_results: int) -> list[str]:
        """Run lexical BM25 retrieval from pre-built corpus rows."""
        if self.bm25 is None or not self.bm25_rows:
            return []

        tokens = tokenize_for_bm25(query, self.lexical_tokenizer_config)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: float(scores[index]),
            reverse=True,
        )

        output: list[str] = []
        for index in ranked_indices:
            if len(output) >= max(1, n_results):
                break
            row = self.bm25_rows[index]
            meta = row.get("metadata", {}) or {}
            if course_filter and str(meta.get("course", "")) != course_filter:
                continue
            if week_filter and str(meta.get("week", "")) != week_filter:
                continue
            output.append(str(row.get("chunk_id", "")))
        return [chunk_id for chunk_id in output if chunk_id]

    def _hydrate_hit(self, chunk_id: str, score: float) -> dict[str, Any]:
        """Materialize a fused hit from BM25 rows, or fallback to Chroma documents."""
        row = self.by_id.get(chunk_id)
        if row:
            return {
                "chunk_id": chunk_id,
                "score": score,
                "text": str(row.get("text", "")),
                "metadata": row.get("metadata", {}) or {},
            }

        fetched = self.collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        docs = fetched.get("documents") or []
        metas = fetched.get("metadatas") or []
        return {
            "chunk_id": chunk_id,
            "score": score,
            "text": str(docs[0]) if docs else "",
            "metadata": metas[0] if metas else {},
        }

    def _image_search(
        self,
        query: str,
        where: dict[str, Any] | None,
        n_results: int,
    ) -> dict[str, float]:
        """Retrieve page images and return normalized image scores by image id."""
        if self.image_collection.count() <= 0:
            return {}
        query_embedding = self.backbone.encode_text([query])[0]
        try:
            results = self.image_collection.query(
                query_embeddings=[query_embedding],
                n_results=max(1, n_results),
                where=where,
                include=["distances"],
            )
        except Exception:
            return {}
        ids = ((results.get("ids") or [[]])[0] or [])
        distances = ((results.get("distances") or [[]])[0] or [])
        image_scores: dict[str, float] = {}
        for image_id, distance in zip(ids, distances):
            try:
                # Chroma cosine distance: smaller is better. Convert to [0, 1]-ish score.
                score = 1.0 / (1.0 + float(distance))
            except (TypeError, ValueError):
                score = 0.0
            image_scores[str(image_id)] = score
        logging.getLogger(__name__).info(
            "multimodal_retrieval image_query_hits=%d",
            len(image_scores),
        )
        return image_scores

    @staticmethod
    def _image_id_from_metadata(metadata: dict[str, Any]) -> str:
        doc_id = str(metadata.get("doc_id", "")).strip()
        unit_id = str(metadata.get("unit_id", "")).strip()
        if not doc_id or not unit_id:
            return ""
        return f"pdf__{doc_id}__{unit_id}"

    def _multimodal_fuse_scores(
        self,
        text_hits: list[dict[str, Any]],
        image_scores: dict[str, float],
    ) -> list[dict[str, Any]]:
        """Blend text retrieval score with matching page-image retrieval score."""
        alpha = max(0.0, min(1.0, float(self.settings.multimodal_fusion_alpha)))
        fused_hits: list[dict[str, Any]] = []
        overlap_count = 0
        for hit in text_hits:
            metadata = hit.get("metadata", {}) or {}
            image_id = self._image_id_from_metadata(metadata)
            image_score = float(image_scores.get(image_id, 0.0))
            if image_score > 0.0:
                overlap_count += 1
            text_score = float(hit.get("score", 0.0))
            fused_score = alpha * text_score + (1.0 - alpha) * image_score
            fused_hits.append(
                {
                    "chunk_id": hit.get("chunk_id", ""),
                    "score": fused_score,
                    "text": hit.get("text", ""),
                    "metadata": metadata,
                    "text_score": text_score,
                    "image_score": image_score,
                    "image_id": image_id,
                }
            )
        fused_hits.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        logging.getLogger(__name__).info(
            "multimodal_retrieval fused_text_hits=%d overlap_with_images=%d alpha=%.2f",
            len(text_hits),
            overlap_count,
            alpha,
        )
        return fused_hits

    def search_planner_images(self, query: str, top_k: int, week_filter: str) -> list[str]:
        """Retrieve planner screenshot image paths for C3 multimodal extraction."""
        if self.image_collection.count() <= 0:
            return []
        where = {"$and": [{"course": "personal-planner"}, {"resource_type": "planner_screenshot"}]}
        if week_filter:
            where["$and"].append({"week": week_filter})
        query_embedding = self.backbone.encode_text([query])[0]
        try:
            results = self.image_collection.query(
                query_embeddings=[query_embedding],
                n_results=max(1, int(top_k)),
                where=where,
                include=["metadatas"],
            )
        except Exception:
            return []
        metadatas = ((results.get("metadatas") or [[]])[0] or [])
        image_paths: list[str] = []
        for metadata in metadatas:
            if not isinstance(metadata, dict):
                continue
            candidate = str(metadata.get("path", "")).strip()
            if not candidate:
                rel = str(metadata.get("relative_path", "")).strip()
                if rel:
                    candidate = str((_BASE_DIR / rel).resolve())
            if candidate and Path(candidate).exists():
                image_paths.append(candidate)
        unique_paths = list(dict.fromkeys(image_paths))
        logging.getLogger(__name__).info("c3_debug planner_image_hits count=%d", len(unique_paths))
        return unique_paths

    def search(
        self,
        query: str,
        top_k: int,
        course_filter: str,
        week_filter: str,
        *,
        strategy_override: TextRetrievalStrategy | None = None,
        enable_multimodal_fusion: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Run configured retrieval strategy with optional course/week filters."""
        use_meta = self.settings.retrieval_metadata_filter_enabled
        eff_course = course_filter if use_meta else ""
        eff_week = week_filter if use_meta else ""
        where = self._make_where(course_filter=eff_course, week_filter=eff_week)
        strategy = strategy_override or self.settings.text_retrieval_strategy
        _trace_log(
            "retrieval.search query=%r top_k=%d strategy=%s use_meta=%s eff_course=%r eff_week=%r where=%s",
            query,
            max(1, int(top_k)),
            strategy.value,
            use_meta,
            eff_course,
            eff_week,
            where if where is not None else "none",
        )
        dense_ids: list[str] = []
        bm25_ids: list[str] = []
        fused_scores: dict[str, float] = {}

        if strategy == TextRetrievalStrategy.DENSE_ONLY:
            dense_ids = self._dense_search(query, where=where, n_results=max(top_k, self.settings.dense_k))
            _trace_log("retrieval.candidates dense_only count=%d dense_head=%s", len(dense_ids), dense_ids[:3])
            for rank, chunk_id in enumerate(dense_ids, start=1):
                fused_scores[chunk_id] = 1.0 / float(rank)
        else:
            dense_ids = self._dense_search(query, where=where, n_results=max(top_k, self.settings.dense_k))
            bm25_ids = self._bm25_search(
                query=query,
                course_filter=eff_course,
                week_filter=eff_week,
                n_results=max(top_k, self.settings.bm25_k),
            )
            _trace_log(
                "retrieval.candidates hybrid dense=%d bm25=%d dense_head=%s bm25_head=%s",
                len(dense_ids),
                len(bm25_ids),
                dense_ids[:3],
                bm25_ids[:3],
            )
            fused_scores = _rrf_fuse(dense_ids, bm25_ids, k=max(1, self.settings.rrf_k))

        ranked_ids = sorted(fused_scores.keys(), key=lambda chunk_id: fused_scores[chunk_id], reverse=True)[: max(1, top_k)]
        _trace_log("retrieval.ranked strategy=%s count=%d head=%s", strategy.value, len(ranked_ids), ranked_ids[:3])
        text_hits = [self._hydrate_hit(chunk_id, score=fused_scores.get(chunk_id, 0.0)) for chunk_id in ranked_ids]

        multimodal_enabled = (
            self.settings.rag_mode == RAGPipelineMode.MULTIMODAL_RETRIEVAL_MLLM
            if enable_multimodal_fusion is None
            else bool(enable_multimodal_fusion)
        )
        if not multimodal_enabled:
            return text_hits

        image_scores = self._image_search(
            query=query,
            where=where,
            n_results=max(top_k, self.settings.multimodal_fusion_k),
        )
        multimodal_hits = self._multimodal_fuse_scores(text_hits=text_hits, image_scores=image_scores)
        _trace_log("retrieval.multimodal final_hits=%d", len(multimodal_hits))
        return multimodal_hits[: max(1, top_k)]


def get_retriever() -> CourseRetriever:
    """Return singleton C1 retriever."""
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = CourseRetriever(get_settings())
    return _RETRIEVER


@tool
def kb_course_retrieval(
    query: str,
    top_k: int = 6,
    course_filter: str = "",
    week_filter: str = "",
) -> str:
    """
    C1a Retrieval (internal): return grounded snippets for course/week/topic.

    Args:
        query: User query/topic to retrieve evidence for.
        top_k: Number of evidence snippets to return.
        course_filter: Optional explicit course slug filter.
        week_filter: Optional explicit week filter (for example, week-03).
    """
    settings = get_settings()
    resolved_course = _resolve_course_filter(
        course_filter,
        query,
        filter_enabled=settings.course_filter_enabled,
    )
    resolved_week = _resolve_week_filter(week_filter, query=query)

    normalized_query = query.strip()
    if not normalized_query:
        return "COURSE_RETRIEVAL_RESULTS:\nquery field was missing."
    _trace_log(
        "c1.kb_course_retrieval query=%r tool_course=%r tool_week=%r resolved_course=%r resolved_week=%r",
        normalized_query,
        course_filter,
        week_filter,
        resolved_course,
        resolved_week,
    )

    # resolved_* are always computed for the tool transcript; only applied inside
    # CourseRetriever.search when settings.retrieval_metadata_filter_enabled is True.
    hits = get_retriever().search(
        query=normalized_query,
        top_k=max(1, int(top_k)),
        course_filter=resolved_course,
        week_filter=resolved_week,
    )
    _trace_log("c1.retrieval hits=%d", len(hits))

    if not hits:
        return "COURSE_RETRIEVAL_RESULTS:\nNo grounded snippets found for the current filters."

    lines: list[str] = ["C1A_RETRIEVAL_RESULTS:"]
    meta_on = settings.retrieval_metadata_filter_enabled
    lines.append(
        f"Applied filters -> metadata_filter={'on' if meta_on else 'off'}"
        f" | course={resolved_course or 'none'} | week={resolved_week or 'none'}"
        f" | top_k={max(1, int(top_k))}"
        f" | rag_mode={settings.rag_mode.value}"
        f" | retrieval_strategy={settings.text_retrieval_strategy.value}"
    )
    image_paths: list[str] = []
    for rank, hit in enumerate(hits, start=1):
        metadata = hit.get("metadata", {}) or {}
        source_path = str(metadata.get("source_path", "?"))
        source_file = Path(source_path).name if source_path and source_path != "?" else "unknown-source"
        page_image_path = str(metadata.get("page_image_path", "")).strip()
        page_image_abs = str((_BASE_DIR / page_image_path).resolve()) if page_image_path else ""
        if page_image_abs and Path(page_image_abs).exists():
            image_paths.append(page_image_abs)
        snippet = str(hit.get("text", "")).replace("\n", " ").strip()
        snippet = re.sub(r"\[[^\]]+\]\s*", "", snippet).strip()
        if len(snippet) > 260:
            snippet = snippet[:260].rstrip() + "..."
        lines.append(
            (
                f"{rank}. score={float(hit.get('score', 0.0)):.5f} "
                f"source_file={source_file} "
                f"course={metadata.get('course', '?')} "
                f"week={metadata.get('week', '?')}"
            )
        )
        if "text_score" in hit or "image_score" in hit:
            lines.append(
                (
                    f"   score_breakdown="
                    f"text:{float(hit.get('text_score', 0.0)):.5f} "
                    f"image:{float(hit.get('image_score', 0.0)):.5f}"
                )
            )
        if page_image_abs:
            lines.append(f"   page_image={page_image_abs}")
        lines.append(f"   snippet={snippet}")
    if image_paths:
        unique_image_paths = list(dict.fromkeys(image_paths))
        lines.append(f"IMAGE_PATHS_JSON={json.dumps(unique_image_paths)}")
    return "\n".join(lines)


@tool
def estimate_study_duration(
    task_query: str = "",
    query: str = "",
    course_filter: str = "",
    week_filter: str = "",
    day_filter: str = "",
    target_outcome: str = "",
) -> str:
    """C2 duration estimator using behavior from active RAG mode."""
    mode = get_settings().rag_mode
    return _estimate_study_duration_mode(
        task_query=task_query,
        query=query,
        course_filter=course_filter,
        week_filter=week_filter,
        day_filter=day_filter,
        target_outcome=target_outcome,
        mode_name=mode.value,
    )


def _build_duration_prompt_payload(
    *,
    mode_name: str,
    task_query: str,
    outcome: str,
    resolved_course: str,
    resolved_week: str,
    resolved_day: str,
    doc_lines: list[str],
    retrieval_lines: list[str],
) -> str:
    return (
        f"mode_name={mode_name}\n"
        f"task_query={task_query}\n"
        f"target_outcome={outcome}\n"
        f"resolved_course={resolved_course or 'none'}\n"
        f"resolved_week={resolved_week or 'none'}\n"
        f"resolved_day={resolved_day or 'none'}\n\n"
        "documents.csv candidates:\n"
        f"{chr(10).join(doc_lines) if doc_lines else '- none'}\n\n"
        "retrieved course evidence snippets:\n"
        f"{chr(10).join(retrieval_lines) if retrieval_lines else '- none'}\n"
    )


def _estimate_study_duration_mode(
    *,
    task_query: str,
    query: str,
    course_filter: str,
    week_filter: str,
    day_filter: str,
    target_outcome: str,
    mode_name: str,
) -> str:
    """Shared C2 implementation for the three RAG mode ablations."""
    logger = logging.getLogger(__name__)
    normalized_task_query = task_query.strip() or query.strip()
    if not normalized_task_query:
        return "C2_DURATION_ESTIMATE:\nquery field was missing."

    settings = get_settings()
    resolved_course = _resolve_course_filter(
        course_filter,
        normalized_task_query,
        filter_enabled=settings.course_filter_enabled,
    )
    resolved_week = _resolve_week_filter(week_filter, query=normalized_task_query)
    resolved_day = _normalize_day_filter(day_filter) or _detect_day_filter(normalized_task_query)
    outcome = target_outcome.strip().lower() or "unspecified"
    logger.info(
        "c2_debug start mode=%s rag_mode=%s query=%r course_in=%r week_in=%r day_in=%r "
        "resolved_course=%r resolved_week=%r resolved_day=%r outcome=%r",
        mode_name,
        settings.rag_mode.value,
        _truncate_for_log(normalized_task_query, 180),
        course_filter,
        week_filter,
        day_filter,
        resolved_course,
        resolved_week,
        resolved_day,
        outcome,
    )

    document_rows = get_document_rows()
    candidate_docs = [
        row
        for row in document_rows
        if (not resolved_course or str(row.get("course", "")).strip().lower() == resolved_course)
        and (not resolved_week or str(row.get("week", "")).strip().lower() in {resolved_week, "na", ""})
        and str(row.get("tags", "")).strip().lower() == "course-content"
    ][:20]
    logger.info("c2_debug candidate_docs count=%d", len(candidate_docs))
    doc_lines: list[str] = []
    for row in candidate_docs[:8]:
        source_path = str(row.get("source_path", "")).strip()
        source_abs = (_BASE_DIR / source_path).resolve() if source_path else None
        file_size_bytes = source_abs.stat().st_size if source_abs and source_abs.exists() else -1
        doc_lines.append(
            (
                f"- doc_id={row.get('doc_id', '')}, "
                f"course={row.get('course', '')}, week={row.get('week', '')}, "
                f"file_size_bytes={file_size_bytes if file_size_bytes >= 0 else 'unknown'}"
            )
        )

    hits = get_retriever().search(
        query=normalized_task_query,
        top_k=max(3, min(8, int(settings.hybrid_k))),
        course_filter=resolved_course,
        week_filter=resolved_week,
    )
    logger.info(
        "c2_debug retrieval_hits count=%d top_k=%d",
        len(hits),
        max(3, min(8, int(settings.hybrid_k))),
    )
    retrieval_lines: list[str] = []
    image_paths: list[str] = []
    course_material_hits: list[dict[str, Any]] = []
    for hit in hits:
        course_material_hits.append(hit)

    logger.info("c2_debug course_material_hits count=%d", len(course_material_hits))
    for hit in course_material_hits[:6]:
        metadata = hit.get("metadata", {}) or {}
        source_file = Path(str(metadata.get("source_path", "unknown"))).name
        snippet = str(hit.get("text", "")).replace("\n", " ").strip()
        if len(snippet) > 220:
            snippet = snippet[:220].rstrip() + "..."
        retrieval_lines.append(
            (
                f"- source={source_file}, "
                f"score={float(hit.get('score', 0.0)):.5f}, snippet={snippet}"
            )
        )
        page_image_path = str(metadata.get("page_image_path", "")).strip()
        page_image_abs = str((_BASE_DIR / page_image_path).resolve()) if page_image_path else ""
        if page_image_abs and Path(page_image_abs).exists():
            image_paths.append(page_image_abs)
    logger.info(
        "c2_debug course_material_images count=%d multimodal_mode=%s",
        len(image_paths),
        settings.rag_mode == RAGPipelineMode.MULTIMODAL_RETRIEVAL_MLLM,
    )

    prompt_text = _build_duration_prompt_payload(
        mode_name=mode_name,
        task_query=normalized_task_query,
        outcome=outcome,
        resolved_course=resolved_course,
        resolved_week=resolved_week,
        resolved_day=resolved_day,
        doc_lines=doc_lines,
        retrieval_lines=retrieval_lines,
    )
    system_text = (
        "You estimate study duration for a personal planner.\n"
        "Use only grounded course-material evidence provided by the user "
        "(documents metadata, retrieved text snippets, and optional rendered PDF images).\n"
        "Infer task duration from document length signals and your own assessment of material difficulty.\n"
        "Use file_size_bytes as the only metadata cue.\n"
        "Explicitly mention the materials used for the estimate.\n"
        "Return strict JSON only with keys:\n"
        "- estimated_minutes (integer)\n"
        "- confidence (one of: low, medium, high)\n"
        "- rationale (short string, mention key evidence signals)\n"
        "- suggested_blocking (string, e.g., '2 x 60min').\n"
        "Do not call any tools."
    )

    if settings.rag_mode == RAGPipelineMode.MULTIMODAL_RETRIEVAL_MLLM and image_paths:
        multimodal_payload: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
        for image_path in list(dict.fromkeys(image_paths))[: max(1, settings.mllm_max_images)]:
            data_url = _image_path_to_data_url(image_path, settings.mllm_max_image_edge)
            if not data_url:
                continue
            multimodal_payload.append({"type": "image_url", "image_url": {"url": data_url}})
        llm_input = [
            SystemMessage(content=system_text),
            HumanMessage(content=multimodal_payload),
        ]
    else:
        llm_input = [
            SystemMessage(content=system_text),
            HumanMessage(content=prompt_text),
        ]
    logger.info(
        "c2_debug estimator_llm_invoke mode=%s multimodal_payload=%s images_attached=%d prompt_chars=%d",
        mode_name,
        settings.rag_mode == RAGPipelineMode.MULTIMODAL_RETRIEVAL_MLLM and bool(image_paths),
        len(list(dict.fromkeys(image_paths))) if image_paths else 0,
        len(prompt_text),
    )

    raw_response = _render_assistant_content(get_reasoning_llm().invoke(llm_input).content)
    payload = _extract_first_json_object(raw_response)
    logger.info(
        "c2_debug estimator_llm_response has_json=%s has_estimated_minutes=%s response_preview=%r",
        bool(payload),
        bool(payload and "estimated_minutes" in payload),
        _truncate_for_log(raw_response, 220),
    )
    if not payload or "estimated_minutes" not in payload:
        logging.getLogger(__name__).warning(
            "c2_estimator invalid_json_payload mode=%s task_query=%s raw_response_preview=%s",
            mode_name,
            _truncate_for_log(normalized_task_query, 140),
            _truncate_for_log(raw_response, 320),
        )
        return (
            "C2_DURATION_ESTIMATE:\n"
            "Estimator did not return valid JSON output. "
            "Ask the agent to retry the duration estimation."
        )

    estimated_minutes = _safe_int(payload.get("estimated_minutes"), 0)
    confidence = str(payload.get("confidence", "medium")).strip().lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "medium"
    rationale = str(payload.get("rationale", "")).strip() or "No rationale provided by estimator."
    suggested_blocking = str(payload.get("suggested_blocking", "")).strip() or f"1 x {estimated_minutes}min"

    result_payload = {
        "estimated_minutes": estimated_minutes,
        "confidence": confidence,
        "rationale": rationale,
        "suggested_blocking": suggested_blocking,
    }
    lines = [
        "C2_DURATION_ESTIMATE:",
        (
            f"Applied filters -> mode={mode_name} | retrieval_strategy={settings.text_retrieval_strategy.value} | "
            f"course={resolved_course or 'none'} | week={resolved_week or 'none'} | day={resolved_day or 'none'} | outcome={outcome}"
        ),
        f"REQUIRED_MINUTES={estimated_minutes}",
        f"RESOLVED_WEEK_FILTER={resolved_week or ''}",
        f"RESOLVED_DAY_FILTER={resolved_day or ''}",
        f"ESTIMATE_JSON={json.dumps(result_payload, ensure_ascii=True)}",
    ]
    return "\n".join(lines)


def _build_c3_planner_query(resolved_week: str, resolved_day: str) -> str:
    """Compose an OCR-aligned retrieval query for planner screenshot extraction."""
    day_scope = resolved_day or "Mon Tue Wed Thu Fri Sat Sun"
    return (
        f"Personal planner screenshot for {resolved_week}. "
        f"Calendar with day labels ({day_scope}) and time grid (07:00-21:00). "
        "Retrieve busy events only."
    )


def _extract_events_from_planner_screenshots(
    *,
    query: str,
    week_filter: str,
    day_filter: str,
    top_k: int,
    include_images: bool,
) -> list[dict[str, Any]]:
    """Extract planner events from indexed planner screenshot OCR snippets."""
    logger = logging.getLogger(__name__)
    settings = get_settings()
    normalized_week = week_filter.strip().lower()
    logger.info(
        "c2_debug screenshot_extract start query=%r week_filter=%r day_filter=%r top_k=%d include_images=%s",
        _truncate_for_log(query, 180),
        normalized_week,
        day_filter,
        max(6, int(top_k)),
        include_images,
    )
    snippet_lines: list[str] = []
    image_paths: list[str] = []
    hits = get_retriever().search(
        query=query,
        top_k=max(6, int(top_k)),
        course_filter="personal-planner",
        week_filter=normalized_week,
        strategy_override=TextRetrievalStrategy.DENSE_ONLY,
        enable_multimodal_fusion=False,
    )
    logger.info("c2_debug screenshot_extract retrieval_hits count=%d", len(hits))
    for hit in hits:
        metadata = hit.get("metadata", {}) or {}
        if str(metadata.get("resource_type", "")).strip().lower() != "planner_screenshot":
            continue
        source = Path(str(metadata.get("source_path", "unknown"))).name
        snippet = str(hit.get("text", "")).replace("\n", " ").strip()
        snippet = re.sub(r"\s+", " ", snippet)
        if len(snippet) > 360:
            snippet = snippet[:360].rstrip() + "..."
        snippet_lines.append(f"- source={source}, snippet={snippet}")
        source_path = str(metadata.get("source_path", "")).strip()
        source_abs = str((_BASE_DIR / source_path).resolve()) if source_path else ""
        if source_abs and Path(source_abs).exists():
            image_paths.append(source_abs)
    if include_images:
        image_paths.extend(
            get_retriever().search_planner_images(
                query=query,
                top_k=max(1, int(top_k)),
                week_filter=normalized_week,
            )
        )
    image_paths = list(dict.fromkeys(image_paths))
    logger.info(
        "c2_debug screenshot_extract planner_hits=%d image_candidates=%d",
        len(snippet_lines),
        len(image_paths),
    )
    if not snippet_lines and not image_paths:
        logger.info("c2_debug screenshot_extract no_snippets -> return []")
        return []

    system_text = (
        "Extract calendar-like busy events from planner evidence.\n"
        "Return strict JSON only with this schema:\n"
        "{ \"events\": [\n"
        "  {\"date\":\"YYYY-MM-DD or empty\", \"day\":\"Mon/Tue/... or empty\", "
        "\"start_time\":\"HH:MM\", \"end_time\":\"HH:MM\", "
        "\"title\":\"short\", \"course\":\"slug or empty\"}\n"
        "]}\n"
        "Rules:\n"
        "- Keep only events that have start_time and end_time.\n"
        "- Use 24-hour HH:MM.\n"
        "- Do not invent missing values.\n"
        "- Preserve uncertainty by leaving date/day/course empty when unknown."
    )
    if include_images:
        system_text += (
            "\nMultimodal guidance:\n"
            "- Filled calendar blocks indicate busy events.\n"
            "- Empty background gaps between busy blocks indicate potential free time.\n"
            "- Extract only busy events; do not output free slots.\n"
            "- Be robust to display themes (do not assume only white backgrounds)."
        )
    snippets_payload = f"{chr(10).join(snippet_lines[:14])}" if snippet_lines else "- no reliable OCR snippets provided"
    prompt_text = (
        f"week_filter={normalized_week or 'none'}\n"
        f"day_filter={day_filter or 'none'}\n"
        f"query={query}\n\n"
        "planner_context:\n"
        f"{snippets_payload}"
    )
    attached_images = 0
    if include_images and image_paths:
        multimodal_payload: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
        for image_path in list(dict.fromkeys(image_paths))[: max(1, settings.mllm_max_images)]:
            data_url = _image_path_to_data_url(image_path, settings.mllm_max_image_edge)
            if not data_url:
                continue
            multimodal_payload.append({"type": "image_url", "image_url": {"url": data_url}})
            attached_images += 1
        messages = [SystemMessage(content=system_text), HumanMessage(content=multimodal_payload)]
    else:
        messages = [SystemMessage(content=system_text), HumanMessage(content=prompt_text)]

    logger.info(
        "c2_debug screenshot_extract llm_invoke multimodal_payload=%s images_attached=%d prompt_chars=%d",
        include_images and attached_images > 0,
        attached_images,
        len(prompt_text),
    )
    raw = _render_assistant_content(get_reasoning_llm().invoke(messages).content)
    parsed = _extract_first_json_object(raw) or {}
    events = parsed.get("events", [])
    logger.info(
        "c2_debug screenshot_extract llm_response has_json=%s raw_events_type=%s raw_events_count=%s preview=%r",
        bool(parsed),
        type(events).__name__,
        len(events) if isinstance(events, list) else "n/a",
        _truncate_for_log(raw, 220),
    )
    if not isinstance(events, list):
        return []

    cleaned: list[dict[str, Any]] = []
    for item in events:
        if not isinstance(item, dict):
            continue
        start_time = str(item.get("start_time", "")).strip()
        end_time = str(item.get("end_time", "")).strip()
        if _parse_hhmm_to_minutes(start_time) is None or _parse_hhmm_to_minutes(end_time) is None:
            continue
        start_min = _parse_hhmm_to_minutes(start_time)
        end_min = _parse_hhmm_to_minutes(end_time)
        if start_min is None or end_min is None or end_min <= start_min:
            continue
        day_value = _normalize_day_filter(str(item.get("day", "")).strip())
        if day_filter and day_value and day_value != day_filter:
            continue
        cleaned.append(
            {
                "date": str(item.get("date", "")).strip(),
                "day": day_value,
                "start_time": start_time,
                "end_time": end_time,
                "title": str(item.get("title", "")).strip(),
                "course": _canonicalize_course_slug(str(item.get("course", "")).strip()),
                "duration_minutes": end_min - start_min,
            }
        )
    logger.info("c2_debug screenshot_extract cleaned_events count=%d", len(cleaned))
    return cleaned


def _compute_free_slots_from_rows(
    *,
    week_rows: list[dict[str, str]],
    resolved_week: str,
    resolved_day: str,
    required: int,
    min_slot: int,
    start_min: int,
    end_min: int,
    day_start: str,
    day_end: str,
    mode_name: str,
) -> str:
    """Compute free slots from normalized busy intervals."""
    settings = get_settings()
    normalized_rows: list[dict[str, Any]] = []
    for row in week_rows:
        raw_day = _normalize_day_filter(str(row.get("day", "")).strip())
        raw_date = str(row.get("date", "")).strip()
        parsed_date: date | None = None
        if raw_date:
            try:
                parsed_date = datetime.strptime(raw_date, "%Y-%m-%d").date()
            except ValueError:
                parsed_date = None

        day_value = raw_day or (parsed_date.strftime("%a") if parsed_date else "")
        if resolved_day and day_value and day_value != resolved_day:
            continue
        if not parsed_date and not day_value:
            continue

        task_start = _parse_hhmm_to_minutes(str(row.get("start_time", "")))
        task_end = _parse_hhmm_to_minutes(str(row.get("end_time", "")))
        if task_start is None or task_end is None or task_end <= task_start:
            continue

        clipped_start = max(start_min, task_start)
        clipped_end = min(end_min, task_end)
        if clipped_end <= clipped_start:
            continue

        date_key = parsed_date.isoformat() if parsed_date else f"{resolved_week}|{day_value or 'UNK'}"
        normalized_rows.append(
            {
                "date": date_key,
                "day": day_value,
                "start": clipped_start,
                "end": clipped_end,
            }
        )

    if not normalized_rows:
        return (
            "C3_FREE_SLOTS:\n"
            f"Applied filters -> mode={mode_name} | week={resolved_week} | day={resolved_day or 'all'}\n"
            "No planner rows matched the selected week/day."
        )

    by_date: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for row in normalized_rows:
        day_value = str(row.get("day", "")).strip() or (resolved_day or "UNK")
        date_value = str(row.get("date", "")).strip()
        clipped_start = int(row.get("start", 0))
        clipped_end = int(row.get("end", 0))
        by_date[(date_value, day_value)].append((clipped_start, clipped_end))

    slots: list[dict[str, Any]] = []
    for (date_value, day_value), intervals in sorted(by_date.items()):
        merged: list[tuple[int, int]] = []
        for interval_start, interval_end in sorted(intervals):
            if not merged or interval_start > merged[-1][1]:
                merged.append((interval_start, interval_end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], interval_end))

        cursor = start_min
        for interval_start, interval_end in merged:
            gap = interval_start - cursor
            if gap >= min_slot:
                slots.append(
                    {
                        "date": date_value,
                        "day": day_value,
                        "start": cursor,
                        "end": interval_start,
                        "duration": gap,
                        "fits_required": gap >= required,
                    }
                )
            cursor = max(cursor, interval_end)
        tail_gap = end_min - cursor
        if tail_gap >= min_slot:
            slots.append(
                {
                    "date": date_value,
                    "day": day_value,
                    "start": cursor,
                    "end": end_min,
                    "duration": tail_gap,
                    "fits_required": tail_gap >= required,
                }
            )

    if not slots:
        return (
            "C3_FREE_SLOTS:\n"
            f"Applied filters -> mode={mode_name} | week={resolved_week} | day={resolved_day or 'all'}\n"
            "No free slots found within the selected day bounds."
        )

    fitting = [slot for slot in slots if bool(slot.get("fits_required"))]
    fitting.sort(key=lambda slot: (str(slot.get("date")), int(slot.get("start", 0)), -int(slot.get("duration", 0))))
    partial = sorted(slots, key=lambda slot: int(slot.get("duration", 0)), reverse=True)
    lines = [
        "C3_FREE_SLOTS:",
        (
            f"Applied filters -> mode={mode_name} | retrieval_strategy={settings.text_retrieval_strategy.value} | "
            f"week={resolved_week} | day={resolved_day or 'all'} | required_minutes={required} | day_bounds={day_start}-{day_end}"
        ),
    ]
    if fitting:
        lines.append("Best fitting slots:")
        for index, slot in enumerate(fitting[:6], start=1):
            lines.append(
                (
                    f"{index}. {slot['date']} {slot['day']} { _minutes_to_hhmm(int(slot['start'])) }-"
                    f"{ _minutes_to_hhmm(int(slot['end'])) } ({slot['duration']} min)"
                )
            )
    else:
        lines.append("No exact-fit slot found. Largest partial slots:")
        for index, slot in enumerate(partial[:6], start=1):
            lines.append(
                (
                    f"{index}. {slot['date']} {slot['day']} { _minutes_to_hhmm(int(slot['start'])) }-"
                    f"{ _minutes_to_hhmm(int(slot['end'])) } ({slot['duration']} min)"
                )
            )
    lines.append(f"SLOTS_JSON={json.dumps(slots[:20], ensure_ascii=True)}")
    return "\n".join(lines)


def _find_free_slots_from_mode(
    *,
    week_filter: str,
    required_minutes: int,
    day_filter: str,
    min_slot_minutes: int,
    day_start: str,
    day_end: str,
    mode_name: str,
) -> str:
    """Shared C3 implementation using evidence-only planner screenshot retrieval."""
    c3_mode_name = (
        RAGPipelineMode.MULTIMODAL_RETRIEVAL_MLLM.value
        if get_settings().rag_mode == RAGPipelineMode.MULTIMODAL_RETRIEVAL_MLLM
        else RAGPipelineMode.TEXT_ONLY.value
    )
    resolved_week = _resolve_week_filter(week_filter)
    if not resolved_week:
        return "C3_FREE_SLOTS:\nweek_filter field was missing."

    resolved_day = _normalize_day_filter(day_filter)
    required = _safe_int(required_minutes, 60)
    min_slot = _safe_int(min_slot_minutes, 30)
    start_min = _parse_hhmm_to_minutes(day_start)
    end_min = _parse_hhmm_to_minutes(day_end)
    if start_min is None or end_min is None or start_min >= end_min:
        return "C3_FREE_SLOTS:\nInvalid day_start/day_end bounds."

    extracted = _extract_events_from_planner_screenshots(
        query=_build_c3_planner_query(resolved_week, resolved_day),
        week_filter=resolved_week,
        day_filter=resolved_day,
        top_k=max(12, int(get_settings().hybrid_k) * 3),
        include_images=c3_mode_name == RAGPipelineMode.MULTIMODAL_RETRIEVAL_MLLM.value,
    )
    if c3_mode_name == RAGPipelineMode.TEXT_ONLY.value and not resolved_day:
        date_anchored = sum(
            1
            for item in extracted
            if isinstance(item, dict)
            and bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(item.get("date", "")).strip()))
        )
        if date_anchored == 0:
            return (
                "C3_FREE_SLOTS:\n"
                f"Applied filters -> mode={c3_mode_name} | week={resolved_week} | day=all\n"
                "Insufficient planner evidence to compute reliable free slots in text_only mode. "
                "Try multimodal mode."
            )
    week_rows = [
        {
            "date": str(item.get("date", "")).strip(),
            "day": str(item.get("day", "")).strip(),
            "start_time": str(item.get("start_time", "")).strip(),
            "end_time": str(item.get("end_time", "")).strip(),
            "title": str(item.get("title", "")).strip(),
        }
        for item in extracted
    ]
    return _compute_free_slots_from_rows(
        week_rows=week_rows,
        resolved_week=resolved_week,
        resolved_day=resolved_day,
        required=required,
        min_slot=min_slot,
        start_min=start_min,
        end_min=end_min,
        day_start=day_start,
        day_end=day_end,
        mode_name=c3_mode_name,
    )


@tool
def find_free_slots(
    week_filter: str,
    required_minutes: int,
    day_filter: str = "",
    min_slot_minutes: int = 30,
    day_start: str = "07:00",
    day_end: str = "22:00",
) -> str:
    """C3 free-slot finder using behavior from active RAG mode."""
    mode = get_settings().rag_mode
    return _find_free_slots_from_mode(
        week_filter=week_filter,
        required_minutes=required_minutes,
        day_filter=day_filter,
        min_slot_minutes=min_slot_minutes,
        day_start=day_start,
        day_end=day_end,
        mode_name=mode.value,
    )


def _query_study_environment_text_scores(
    query: str,
    top_k: int,
) -> tuple[dict[str, float], dict[str, str]]:
    """Query indexed text corpus and aggregate study-environment relevance by source image."""
    retriever = get_retriever()
    hits = retriever.search(
        query=query,
        top_k=max(8, int(top_k)),
        course_filter="study-environment",
        week_filter="",
    )
    by_source_score: dict[str, float] = {}
    by_source_excerpt: dict[str, str] = {}
    for hit in hits:
        metadata = hit.get("metadata", {}) or {}
        source_path = str(metadata.get("source_path", "")).strip().replace("\\", "/")
        if not source_path:
            continue
        resource_type = str(metadata.get("resource_type", "")).strip().lower()
        course = str(metadata.get("course", "")).strip().lower()
        if "study-location-photos/" not in source_path and resource_type != "study_environment_photo":
            continue
        if course and course != "study-environment" and "study-location-photos/" not in source_path:
            continue
        score = float(hit.get("score", 0.0))
        if score > float(by_source_score.get(source_path, 0.0)):
            by_source_score[source_path] = score
            snippet = str(hit.get("text", "")).replace("\n", " ").strip()
            if snippet and snippet != "[IMAGE_NO_OCR]":
                by_source_excerpt[source_path] = snippet[:220]
    return by_source_score, by_source_excerpt


def _query_study_environment_image_scores(query: str, top_k: int) -> dict[str, float]:
    """Query indexed study-location photos in planner_images and return source_path scores."""
    retriever = get_retriever()
    if retriever.image_collection.count() <= 0:
        return {}
    query_embedding = retriever.backbone.encode_text([query])[0]
    where = {"$and": [{"course": "study-environment"}, {"raw_folder": "study-location-photos"}]}
    try:
        results = retriever.image_collection.query(
            query_embeddings=[query_embedding],
            n_results=max(1, top_k),
            where=where,
            include=["distances", "metadatas"],
        )
    except Exception:
        return {}

    distances = ((results.get("distances") or [[]])[0] or [])
    metadatas = ((results.get("metadatas") or [[]])[0] or [])
    image_scores: dict[str, float] = {}
    for distance, metadata in zip(distances, metadatas):
        if not isinstance(metadata, dict):
            continue
        source_path = str(metadata.get("relative_path", "")).strip().replace("\\", "/")
        if not source_path:
            continue
        try:
            score = 1.0 / (1.0 + float(distance))
        except (TypeError, ValueError):
            continue
        prior = float(image_scores.get(source_path, 0.0))
        if score > prior:
            image_scores[source_path] = score
    return image_scores


def _resolve_environment_hint(raw_text: str, env_rows: list[dict[str, Any]]) -> str:
    """Resolve an environment hint to a known environment_id."""
    normalized_text = _normalize_for_match(raw_text)
    if not normalized_text:
        return ""
    for row in env_rows:
        env_id = str(row.get("environment_id", "")).strip()
        env_type = str(row.get("environment_type", "")).strip().lower()
        coarse_geo = str(row.get("coarse_geo", "")).strip().lower()
        env_id_norm = _normalize_for_match(env_id)
        env_type_norm = _normalize_for_match(env_type.replace("_", " "))
        coarse_geo_norm = _normalize_for_match(coarse_geo.replace("_", " "))
        if env_id_norm and _term_in_query(env_id_norm, normalized_text):
            return env_id
        if env_type_norm and _term_in_query(env_type_norm, normalized_text):
            return env_id
        if coarse_geo_norm and _term_in_query(coarse_geo_norm, normalized_text):
            return env_id
    return ""


def _run_study_environment_tool(
    *,
    mode: str,
    task_query: str = "",
    query: str = "",
    environment_hint: str = "",
    top_k: int = 4,
) -> str:
    """
    Shared C4 helper used by recommendation and targeted assessment tools.
    """
    targeted_mode = mode == "assess"
    raw_query = query.strip() or task_query.strip()
    normalized_query = raw_query
    if not normalized_query:
        return "C4_STUDY_ENVIRONMENT:\nquery field was missing."
    env_rows = get_study_environment_rows()
    if not env_rows:
        return "C4_STUDY_ENVIRONMENT:\nNo study environments metadata found."

    settings = get_settings()
    text_scores, text_excerpts = _query_study_environment_text_scores(
        normalized_query,
        top_k=max(8, int(top_k) * 3),
    )
    multimodal_on = settings.rag_mode == RAGPipelineMode.MULTIMODAL_RETRIEVAL_MLLM
    image_scores = (
        _query_study_environment_image_scores(normalized_query, top_k=max(6, int(top_k) * 2))
        if multimodal_on
        else {}
    )
    query_terms = {
        token
        for token in re.findall(r"[a-z0-9]+", _normalize_for_match(normalized_query))
        if len(token) >= 3
    }
    max_indexed_text_score = max(text_scores.values()) if text_scores else 0.0
    candidates: list[dict[str, Any]] = []
    for row in env_rows:
        source_path = str(row.get("source_file_path", "")).strip().replace("\\", "/")
        environment_type = str(row.get("environment_type", "")).strip().lower()
        coarse_geo = str(row.get("coarse_geo", "")).strip().lower()

        retrieved_excerpt = text_excerpts.get(source_path, "")
        indexed_text_raw = float(text_scores.get(source_path, 0.0))
        indexed_text_score = (
            indexed_text_raw / max_indexed_text_score if max_indexed_text_score > 0 else 0.0
        )
        text_blob = f"{environment_type} {coarse_geo} {retrieved_excerpt}".lower()
        overlap = sum(1 for term in query_terms if term in text_blob)
        overlap_score = (overlap / max(1, len(query_terms))) if query_terms else 0.0
        text_score = 0.65 * indexed_text_score + 0.35 * overlap_score
        image_score = float(image_scores.get(source_path, 0.0))

        # Retrieval confidence only. Final suitability decision is delegated to LLM.
        retrieval_support_score = text_score
        if multimodal_on:
            retrieval_support_score = 0.65 * text_score + 0.35 * image_score
        candidates.append(
            {
                "environment_id": str(row.get("environment_id", "")).strip(),
                "source_file_path": source_path,
                "coarse_geo": coarse_geo,
                "environment_type": environment_type,
                "text_score": text_score,
                "image_score": image_score,
                "retrieval_support_score": retrieval_support_score,
                "text_excerpt": retrieved_excerpt[:220].strip(),
            }
        )
    candidates.sort(key=lambda item: float(item.get("retrieval_support_score", 0.0)), reverse=True)
    if not candidates:
        return "C4_STUDY_ENVIRONMENT:\nNo candidate environments found."

    targeted_environment_id = (
        _resolve_environment_hint(environment_hint or raw_query, env_rows) if targeted_mode else ""
    )
    if targeted_mode and not targeted_environment_id:
        return (
            "C4_STUDY_ENVIRONMENT:\n"
            "Could not resolve the requested environment. "
            "Provide environment_hint using environment_id or environment_type."
        )
    image_grounding_guidance = (
        "When evidence_mode is multimodal_index+images, explicitly ground your reasoning in observable "
        "image characteristics (for example: visible natural light, crowd density, desk/chair setup, "
        "indoor vs outdoor context, and distraction cues).\n"
        if multimodal_on
        else ""
    )
    llm_candidates = candidates[: max(3, min(8, int(top_k) * 2))]
    llm_prompt = [
        SystemMessage(
            content=(
                "You are analyzing study-environment suitability for a task.\n"
                "You only have sparse metadata (environment_type/coarse_geo) plus retrieval evidence.\n"
                "Infer likely task fit factors (internet connection, natural light, interruptions, "
                "noise, seating comfort) from available evidence. In multimodal mode, use image evidence when present.\n"
                f"{image_grounding_guidance}"
                "Return strict JSON only with keys:\n"
                "- environment_id (must be one of the candidates)\n"
                "- verdict (yes|maybe|no)\n"
                "- confidence (high|medium|low)\n"
                "- short_justification (1 sentence)\n"
                "- factor_rationale (array of 2-4 short bullet strings)\n"
                "- observable_image_evidence (array of 1-3 short bullet strings; empty array in text-only mode)\n"
                "Do not call tools."
            )
        ),
        HumanMessage(
            content=(
                f"task_query={normalized_query}\n"
                f"rag_mode={settings.rag_mode.value}\n"
                f"mode={mode}\n"
                f"target_environment_id={targeted_environment_id or 'none'}\n"
                f"evidence_mode={'multimodal_index+images' if multimodal_on else 'text_index_only'}\n"
                f"candidates_json={json.dumps(llm_candidates, ensure_ascii=True)}"
            )
        ),
    ]
    raw_selection = _render_assistant_content(get_reasoning_llm().invoke(llm_prompt).content)
    parsed_selection = _extract_first_json_object(raw_selection) or {}
    selected_environment_id = str(parsed_selection.get("environment_id", "")).strip()
    selected = next(
        (candidate for candidate in candidates if str(candidate.get("environment_id", "")) == selected_environment_id),
        candidates[0],
    )
    if targeted_mode:
        selected = next(
            (
                candidate
                for candidate in candidates
                if str(candidate.get("environment_id", "")) == targeted_environment_id
            ),
            selected,
        )
    factor_rationale = parsed_selection.get("factor_rationale", [])
    rationale_items = [
        str(item).strip()
        for item in (factor_rationale if isinstance(factor_rationale, list) else [])
        if str(item).strip()
    ][:4]
    image_evidence = parsed_selection.get("observable_image_evidence", [])
    image_evidence_items = [
        str(item).strip()
        for item in (image_evidence if isinstance(image_evidence, list) else [])
        if str(item).strip()
    ][:3]
    short_justification = str(parsed_selection.get("short_justification", "")).strip()
    confidence = str(parsed_selection.get("confidence", "")).strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"
    verdict = str(parsed_selection.get("verdict", "")).strip().lower()
    if verdict not in {"yes", "maybe", "no"}:
        verdict = "maybe" if targeted_mode else "yes"
    recommendation = {
        "environment_id": selected.get("environment_id", ""),
        "environment_type": selected.get("environment_type", ""),
        "coarse_geo": selected.get("coarse_geo", ""),
        "source_file_path": selected.get("source_file_path", ""),
        "justification": short_justification
        or "Chosen by LLM based on factor fit and available retrieval evidence.",
        "factor_rationale": rationale_items,
        "observable_image_evidence": image_evidence_items if multimodal_on else [],
        "verdict": verdict,
        "assessment_mode": "targeted" if targeted_mode else "recommendation",
        "confidence": confidence,
    }

    lines = [
        "C4_STUDY_ENVIRONMENT:",
        (
            f"Applied mode -> rag_mode={settings.rag_mode.value} | "
            f"evidence_mode={'multimodal_index+images' if multimodal_on else 'text_index_only'} | top_k={max(1, int(top_k))}"
        ),
        f"query={normalized_query}",
        f"ASSESSMENT_MODE={'targeted' if targeted_mode else 'recommendation'}",
        f"TARGET_ENVIRONMENT_ID={targeted_environment_id or ''}",
        f"MULTIMODAL_IMAGE_GROUNDING_REQUIRED={1 if multimodal_on else 0}",
        "Top candidates:",
    ]
    for rank, candidate in enumerate(candidates[: max(1, int(top_k))], start=1):
        lines.append(
            (
                f"{rank}. environment_id={candidate.get('environment_id', '')} "
                f"type={candidate.get('environment_type', '')} "
                f"geo={candidate.get('coarse_geo', '')} "
                f"retrieval_support={float(candidate.get('retrieval_support_score', 0.0)):.3f} "
                f"(text={float(candidate.get('text_score', 0.0)):.3f}, "
                f"image={float(candidate.get('image_score', 0.0)):.3f})"
            )
        )
        text_excerpt = str(candidate.get("text_excerpt", "")).strip()
        if text_excerpt:
            lines.append(f"   indexed_text_excerpt={text_excerpt}")
    lines.append(f"LLM_SELECTION_JSON={json.dumps(parsed_selection, ensure_ascii=True)}")
    lines.append(f"RECOMMENDATION_JSON={json.dumps(recommendation, ensure_ascii=True)}")

    if multimodal_on:
        image_paths: list[str] = []
        for candidate in candidates[: max(1, min(int(top_k), 3))]:
            rel = str(candidate.get("source_file_path", "")).strip()
            if not rel:
                continue
            abs_path = (_BASE_DIR / rel).resolve()
            if abs_path.exists():
                image_paths.append(str(abs_path))
        if image_paths:
            lines.append(f"IMAGE_PATHS_JSON={json.dumps(image_paths, ensure_ascii=True)}")
    return "\n".join(lines)


@tool
def recommend_study_environment(
    task_query: str = "",
    query: str = "",
    top_k: int = 4,
) -> str:
    """
    C4a environment recommender: choose where a task should be done.

    Ablation behavior:
    - text_only: uses study_environments.jsonl + indexed text retrieval for study-location photos.
    - multimodal_retrieval_mllm: additionally uses original study-location photo evidence from planner_images.
    """
    return _run_study_environment_tool(
        mode="recommend",
        task_query=task_query,
        query=query,
        top_k=top_k,
    )


@tool
def assess_study_environment_fit(
    task_query: str = "",
    query: str = "",
    environment_hint: str = "",
    top_k: int = 4,
) -> str:
    """
    C4b targeted assessment: judge whether a specific environment is good for the task.

    Example prompt:
    - "Is it a good idea to do this task in common study space?"
    """
    return _run_study_environment_tool(
        mode="assess",
        task_query=task_query,
        query=query,
        environment_hint=environment_hint,
        top_k=top_k,
    )


TOOLS = [
    kb_course_retrieval,
    estimate_study_duration,
    find_free_slots,
    recommend_study_environment,
    assess_study_environment_fit,
]


SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a study planner assistant with four capabilities.\n"
        "C1 TopicSummary: summarize topics studied in a course/week using kb_course_retrieval.\n"
        "C2 DurationEstimator: estimate task minutes with estimate_study_duration.\n"
        "C3 FreeSlotFinder: find open schedule slots with find_free_slots.\n"
        "C4 StudyEnvironment: use recommend_study_environment to choose where to work, "
        "or assess_study_environment_fit to judge a specific environment.\n"
        "Respect dataset week tags like week-01..week-07 and easter-week.\n"
        "Routing rules:\n"
        "- If user asks to estimate duration, call estimate_study_duration.\n"
        "- If user asks to summarize topics/content, call kb_course_retrieval first.\n"
        "- If user asks for free slots, call find_free_slots.\n"
        "- If user asks where to do a task or asks for best study place/environment, call recommend_study_environment first.\n"
        "- If user asks 'Is it a good idea to do this task in [environment]?', call assess_study_environment_fit first and preserve the environment mention in environment_hint.\n"
        "Always include required tool arguments."
    )
)


def _build_llm(settings: Settings):
    """Create Gemini backend and bind available tools."""
    if settings.llm_backend != LLMBackend.GEMINI:
        raise ValueError("This agent is configured for Gemini-only runtime.")
    from langchain_google_genai import ChatGoogleGenerativeAI

    if not settings.gemini_api_key:
        raise ValueError(
            "Gemini selected but API key is missing. Set GOOGLE_API_KEY or GEMINI_API_KEY."
        )
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=0,
        api_key=settings.gemini_api_key,
        convert_system_message_to_human=True,
    ).bind_tools(TOOLS)


def _build_reasoning_llm(settings: Settings):
    """Create a plain Gemini client without bound tools."""
    if settings.llm_backend != LLMBackend.GEMINI:
        raise ValueError("This agent is configured for Gemini-only runtime.")
    from langchain_google_genai import ChatGoogleGenerativeAI

    if not settings.gemini_api_key:
        raise ValueError(
            "Gemini selected but API key is missing. Set GOOGLE_API_KEY or GEMINI_API_KEY."
        )
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=0,
        api_key=settings.gemini_api_key,
        convert_system_message_to_human=True,
    )


def get_llm():
    """Return singleton LLM instance."""
    global _LLM
    if _LLM is None:
        _LLM = _build_llm(get_settings())
    return _LLM


def get_reasoning_llm():
    """Return singleton plain LLM used for non-tool reasoning steps."""
    global _ESTIMATOR_LLM
    if _ESTIMATOR_LLM is None:
        _ESTIMATOR_LLM = _build_reasoning_llm(get_settings())
    return _ESTIMATOR_LLM


def _latest_user_text(messages: list[BaseMessage]) -> str:
    """Get the latest user text for post-tool answer composition."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def _extract_image_paths(tool_result: str) -> list[str]:
    """Extract image paths from tool output payload for MLLM prompts."""
    marker = "IMAGE_PATHS_JSON="
    for line in tool_result.splitlines():
        if not line.startswith(marker):
            continue
        payload = line[len(marker) :].strip()
        try:
            items = json.loads(payload)
        except json.JSONDecodeError:
            return []
        output: list[str] = []
        for item in items if isinstance(items, list) else []:
            path = str(item).strip()
            if path and Path(path).exists():
                output.append(path)
        return output
    return []


def _image_path_to_data_url(path: str, max_edge: int) -> str:
    """Encode local image file as data URL for multimodal LLM input."""
    _ = max_edge  # Reserved for optional in-memory resize/compression.
    image_path = Path(path)
    if not image_path.exists():
        return ""
    suffix = image_path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    try:
        raw = image_path.read_bytes()
    except OSError:
        return ""
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def agent_node(state: AgentState):
    """LLM node: routes tool calls and composes final C1/C2/C3/C4 responses."""
    messages = state["messages"]
    user_query = _latest_user_text(messages)
    _trace_log("agent_node enter last_message_type=%s user_query=%r", getattr(messages[-1], "type", "unknown"), user_query)

    if messages and getattr(messages[-1], "type", None) == "tool":
        settings = get_settings()
        tool_name = str(getattr(messages[-1], "name", "")).strip()
        tool_result = str(messages[-1].content)
        _trace_log("agent_node tool_result name=%s preview=%r", tool_name, _truncate_for_log(tool_result, 180))
        lower_tool_result = tool_result.lower()
        missing_kb_query = (
            tool_name == "kb_course_retrieval"
            and (
                "query field was missing" in lower_tool_result
                or ("field required" in lower_tool_result and "query" in lower_tool_result)
            )
        )
        kb_no_snippets = tool_name == "kb_course_retrieval" and (
            "no grounded snippets found" in lower_tool_result
        )
        if (missing_kb_query or kb_no_snippets) and user_query.strip():
            logging.getLogger(__name__).warning(
                "tool_retry kb_course_retrieval -> retrying with latest user query only "
                "(missing_query=%s empty_snippets=%s)",
                bool(missing_kb_query),
                bool(kb_no_snippets),
            )
            try:
                _trace_log("agent_node retry kb_course_retrieval with query=%r", user_query.strip())
                tool_result = str(
                    kb_course_retrieval.invoke(
                        {
                            "query": user_query.strip(),
                            "top_k": max(1, settings.hybrid_k),
                        }
                    )
                )
            except Exception as exc:
                logging.getLogger(__name__).warning("tool_retry failed: %s", exc)

        non_tool_messages = [m for m in messages if getattr(m, "type", None) != "tool"]

        if tool_name == "estimate_study_duration":
            c2_payload = (
                f'User request: "{user_query}"\n\n'
                "Duration estimate:\n"
                f"{tool_result}"
            )
            prompt_messages = non_tool_messages + [
                SystemMessage(
                    content=(
                        "Produce a concise C2 duration estimate answer.\n"
                        "Output format:\n"
                        "- Estimated effort: minutes and suggested blocking\n"
                        "- Confidence and rationale grounded in provided evidence\n"
                        "- Optional next step: suggest calling C3 with required_minutes"
                    )
                ),
                HumanMessage(content=c2_payload),
            ]
        elif tool_name == "find_free_slots":
            c3_payload = (
                f'User request: "{user_query}"\n\n'
                "Free slots:\n"
                f"{tool_result}"
            )
            prompt_messages = non_tool_messages + [
                SystemMessage(
                    content=(
                        "Produce a concise C3 slot-finder answer.\n"
                        "Summarize top slot options in bullets and mention if a split is needed."
                    )
                ),
                HumanMessage(content=c3_payload),
            ]
        elif tool_name in {"recommend_study_environment", "assess_study_environment_fit"}:
            targeted_mode = tool_name == "assess_study_environment_fit"
            require_image_grounding = "MULTIMODAL_IMAGE_GROUNDING_REQUIRED=1" in tool_result
            c4_payload_text = (
                f'User request: "{user_query}"\n\n'
                "Environment analysis evidence:\n"
                f"{tool_result}"
            )
            image_paths = _extract_image_paths(tool_result)
            include_images = settings.rag_mode == RAGPipelineMode.MULTIMODAL_RETRIEVAL_MLLM
            c4_user_payload: str | list[dict[str, Any]]
            if include_images and image_paths:
                multimodal_content: list[dict[str, Any]] = [{"type": "text", "text": c4_payload_text}]
                attached_count = 0
                for image_path in image_paths[: max(1, settings.mllm_max_images)]:
                    data_url = _image_path_to_data_url(image_path, settings.mllm_max_image_edge)
                    if not data_url:
                        continue
                    multimodal_content.append({"type": "image_url", "image_url": {"url": data_url}})
                    attached_count += 1
                logging.getLogger(__name__).info(
                    "mllm_input_c4 mode=%s candidate_images=%d attached_images=%d",
                    settings.rag_mode.value,
                    len(image_paths),
                    attached_count,
                )
                c4_user_payload = multimodal_content
            else:
                c4_user_payload = c4_payload_text
            prompt_messages = non_tool_messages + [
                SystemMessage(
                    content=(
                        "Produce a concise C4 environment answer.\n"
                        + (
                            "If assessment_mode is targeted, answer whether that specific environment is a good idea "
                            "for the task (yes/maybe/no) and explain why using environment characteristics.\n"
                            "Output format:\n"
                            "- Verdict for requested environment: natural language only.\n"
                            "  Use internal verdict signals but phrase naturally; do not print raw labels yes/maybe/no.\n"
                            "- Why: 1-2 short bullets grounded in factors "
                            "(internet connection, natural light, interruptions, noise, seating comfort)\n"
                            "- Optional caveat if confidence is not high."
                            if targeted_mode
                            else
                            "If assessment_mode is recommendation, select exactly one environment from the evidence.\n"
                            "Output format:\n"
                            "- Recommended environment: environment_id and a short label\n"
                            "- Why it fits: 1-2 short bullets grounded in factors "
                            "(internet connection, natural light, interruptions, noise, seating comfort)\n"
                            "- Optional caveat if confidence is not high."
                        )
                        + (
                            "\nWhen multimodal image grounding is required, include at least one explicit observation "
                            "from visible image characteristics (for example light level, visible crowd/noise cues, "
                            "desk/chair ergonomics, indoor/outdoor exposure) and tie it to the verdict."
                            if require_image_grounding
                            else ""
                        )
                    )
                ),
                HumanMessage(content=c4_user_payload),
            ]
        else:
            image_paths = _extract_image_paths(tool_result)
            include_images = settings.rag_mode in {
                RAGPipelineMode.TEXT_RETRIEVAL_MLLM,
                RAGPipelineMode.MULTIMODAL_RETRIEVAL_MLLM,
            }
            user_payload: str | list[dict[str, Any]]
            if include_images and image_paths:
                multimodal_content: list[dict[str, Any]] = [
                    {
                        "type": "text",
                        "text": (
                            f'User request: "{user_query}"\n\n'
                            "Retrieved evidence:\n"
                            f"{tool_result}"
                        ),
                    }
                ]
                attached_count = 0
                for image_path in image_paths[: max(1, settings.mllm_max_images)]:
                    data_url = _image_path_to_data_url(image_path, settings.mllm_max_image_edge)
                    if not data_url:
                        continue
                    multimodal_content.append({"type": "image_url", "image_url": {"url": data_url}})
                    attached_count += 1
                logging.getLogger(__name__).info(
                    "mllm_input mode=%s candidate_images=%d attached_images=%d",
                    settings.rag_mode.value,
                    len(image_paths),
                    attached_count,
                )
                user_payload = multimodal_content
            else:
                if include_images:
                    logging.getLogger(__name__).info(
                        "mllm_input mode=%s candidate_images=0 attached_images=0",
                        settings.rag_mode.value,
                    )
                user_payload = (
                    f'User request: "{user_query}"\n\n'
                    "Retrieved evidence:\n"
                    f"{tool_result}"
                )
            prompt_messages = non_tool_messages + [
                SystemMessage(
                    content=(
                        "Given retrieved snippets for one course/week, list the main studied topics only. "
                        "Do not mention unrelated details. If evidence is weak, say so.\n"
                        "Output format:\n"
                        "- 3-6 topic bullets\n"
                        "- Optional short evidence note per topic using filename only in brackets, e.g. [lecture3.pdf]"
                    )
                ),
                HumanMessage(content=user_payload),
            ]
    else:
        prompt_messages = messages

    response = get_llm().invoke(prompt_messages)
    if getattr(response, "tool_calls", None):
        tool_names = [str(call.get("name", "")).strip() for call in response.tool_calls if isinstance(call, dict)]
        logging.getLogger(__name__).info(
            "agent_loop_debug model_requested_tools count=%d names=%s",
            len(response.tool_calls),
            tool_names,
        )
    else:
        logging.getLogger(__name__).info("agent_loop_debug model_requested_tools count=0")
    _trace_log("agent_node llm_response has_tool_calls=%s", bool(getattr(response, "tool_calls", None)))
    if getattr(response, "tool_calls", None):
        _trace_log("agent_node tool_calls=%s", getattr(response, "tool_calls", None))
    _log_missing_required_tool_args(getattr(response, "tool_calls", None))
    return {"messages": messages + [response]}


def route_after_agent(state: AgentState):
    """Route to tool execution if the model emitted tool calls."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        tool_names = [
            str(call.get("name", "")).strip()
            for call in (last.tool_calls if isinstance(last.tool_calls, list) else [])
            if isinstance(call, dict)
        ]
        logging.getLogger(__name__).info(
            "agent_loop_debug route=tools tool_count=%d names=%s",
            len(last.tool_calls),
            tool_names,
        )
        return "tools"
    logging.getLogger(__name__).info("agent_loop_debug route=end")
    return END


tool_node = ToolNode(TOOLS)

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")
app = graph.compile()


def _render_assistant_content(content: Any) -> str:
    """Render structured LangChain assistant content as plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    text_parts.append(text)
        if text_parts:
            return "\n".join(text_parts)
    return str(content)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger(__name__).info("PLANNER_RAG_MODE=%s", get_settings().rag_mode.value)
    print(f"[ablation] PLANNER_RAG_MODE={get_settings().rag_mode.value}")
    print("\nPlanner agent (C1 + C2 + C3 + C4) ready. Type 'exit' to quit.\n")
    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in {"exit", "quit"}:
            break

        result = app.invoke(
            {
                "messages": [
                    SYSTEM_PROMPT,
                    HumanMessage(content=user_text),
                ]
            }
        )
        last = result["messages"][-1]
        print(f"\nAgent: {_render_assistant_content(last.content)}\n")
