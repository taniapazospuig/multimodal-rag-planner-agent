"""
Minimal planner agent with three capabilities:
- C1 TopicSummary (topics studied in course/week from retrieved evidence)
- C2 DurationEstimator (estimate study time for a course-related task)
- C3 FreeSlotFinder (find open slots in a target week/day)

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

from config import LLMBackend, RAGPipelineMode, Settings, load_settings
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


def get_task_rows() -> list[dict[str, str]]:
    """Return cached planner task rows."""
    global _TASK_ROWS
    if _TASK_ROWS is None:
        _TASK_ROWS = _load_csv_rows(TASKS_CSV_PATH)
    return _TASK_ROWS


def get_document_rows() -> list[dict[str, str]]:
    """Return cached KB document metadata rows."""
    global _DOCUMENT_ROWS
    if _DOCUMENT_ROWS is None:
        _DOCUMENT_ROWS = _load_csv_rows(DOCUMENTS_CSV_PATH)
    return _DOCUMENT_ROWS


def get_dependency_rows() -> list[dict[str, str]]:
    """Return cached task-to-assignment dependency rows."""
    global _DEPENDENCY_ROWS
    if _DEPENDENCY_ROWS is None:
        _DEPENDENCY_ROWS = _load_csv_rows(DEPENDENCIES_CSV_PATH)
    return _DEPENDENCY_ROWS


def get_assignment_rows() -> list[dict[str, str]]:
    """Return cached assignment metadata rows."""
    global _ASSIGNMENT_ROWS
    if _ASSIGNMENT_ROWS is None:
        _ASSIGNMENT_ROWS = _load_csv_rows(ASSIGNMENTS_CSV_PATH)
    return _ASSIGNMENT_ROWS


COURSES_CSV_PATH = _BASE_DIR / "data" / "kb" / "metadata" / "courses.csv"
COURSES = load_courses(COURSES_CSV_PATH)

_SETTINGS: Settings | None = None
_LLM = None
_ESTIMATOR_LLM = None
_RETRIEVER: "CourseRetriever | None" = None
_OPENCLIP_BACKBONE: "OpenCLIPBackbone | None" = None
_TASK_ROWS: list[dict[str, str]] | None = None
_DOCUMENT_ROWS: list[dict[str, str]] | None = None
_DEPENDENCY_ROWS: list[dict[str, str]] | None = None
_ASSIGNMENT_ROWS: list[dict[str, str]] | None = None

TASKS_CSV_PATH = _BASE_DIR / "data" / "kb" / "metadata" / "tasks.csv"
DOCUMENTS_CSV_PATH = _BASE_DIR / "data" / "kb" / "metadata" / "documents.csv"
DEPENDENCIES_CSV_PATH = _BASE_DIR / "data" / "kb" / "metadata" / "dependencies.csv"
ASSIGNMENTS_CSV_PATH = _BASE_DIR / "data" / "kb" / "metadata" / "assignments.csv"

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


def _detect_week_filter(query: str) -> str:
    """Infer canonical week filter (week-XX) from free-form query text."""
    match = re.search(r"\b(?:week|wk)[\s\-_]*(\d{1,2})\b", query.lower())
    if not match:
        return ""
    week_num = int(match.group(1))
    if week_num <= 0:
        return ""
    return f"week-{week_num:02d}"


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
    candidate = raw_week_filter.strip().lower()
    if candidate:
        return candidate
    return _detect_week_filter(query)


def _week_to_screenshot_stem(week_filter: str) -> str:
    """Map canonical week tags to screenshot stems used by tasks.csv."""
    normalized = week_filter.strip().lower()
    if not normalized:
        return ""
    if normalized == "easter-week":
        return "easter-week"
    match = re.match(r"^week-(\d{1,2})$", normalized)
    if not match:
        return normalized
    return f"week{int(match.group(1))}"


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


def _extract_marker_value(raw_text: str, marker: str) -> str:
    """Extract a line marker value from tool output."""
    for line in raw_text.splitlines():
        if line.startswith(marker):
            return line[len(marker) :].strip()
    return ""


def _filter_tasks_for_week(task_rows: list[dict[str, str]], week_filter: str) -> list[dict[str, str]]:
    """Filter tasks.csv rows by dataset week tag."""
    stem = _week_to_screenshot_stem(week_filter)
    if not stem:
        return []
    output: list[dict[str, str]] = []
    for row in task_rows:
        screenshot = str(row.get("screenshot", "")).strip().lower()
        screenshot_stem = Path(screenshot).stem
        if screenshot_stem == stem:
            output.append(row)
    return output


def _build_assignment_pressure_lines(course_filter: str, week_filter: str) -> list[str]:
    """Summarize assignment pressure linked to tasks in the selected week/course."""
    if not course_filter or not week_filter:
        return []
    week_tasks = [
        row
        for row in _filter_tasks_for_week(get_task_rows(), week_filter)
        if str(row.get("course", "")).strip().lower() == course_filter
    ]
    if not week_tasks:
        return []
    task_ids = {str(row.get("task_id", "")).strip() for row in week_tasks}
    assignment_by_id = {
        str(row.get("assignment_id", "")).strip(): row
        for row in get_assignment_rows()
        if str(row.get("assignment_id", "")).strip()
    }
    week_dates: list[date] = []
    for row in week_tasks:
        try:
            week_dates.append(datetime.strptime(str(row.get("date", "")), "%Y-%m-%d").date())
        except ValueError:
            continue
    week_start = min(week_dates) if week_dates else None

    linked_assignment_ids = {
        str(dep.get("depends_on_assignment_id", "")).strip()
        for dep in get_dependency_rows()
        if str(dep.get("task_id", "")).strip() in task_ids
    }
    lines: list[str] = []
    for assignment_id in sorted(linked_assignment_ids):
        if not assignment_id:
            continue
        row = assignment_by_id.get(assignment_id)
        if not row:
            continue
        due_raw = str(row.get("due_date", "")).strip()
        due_date = None
        if due_raw:
            due_part = due_raw.split("T", maxsplit=1)[0].strip()
            try:
                due_date = datetime.strptime(due_part, "%Y-%m-%d").date()
            except ValueError:
                due_date = None
        days_to_due = ""
        if week_start and due_date:
            days_to_due = f", days_to_due_from_week_start={int((due_date - week_start).days)}"
        lines.append(
            (
                f"- assignment_id={assignment_id}, name={row.get('name', 'unknown')}, "
                f"weight_pct={row.get('weight_pct', '')}, due_date={due_raw or 'na'}{days_to_due}, "
                f"is_hurdle={row.get('is_hurdle', '')}"
            )
        )
    return lines[:6]


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
    C1 retriever: hybrid dense + BM25 search on existing indexes.

    Runtime behavior:
    - reads from `chroma_db` and BM25 JSONL
    - never writes vectors/index rows
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
        """Build optional Chroma metadata filter for course/week constraints."""
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

    def search(self, query: str, top_k: int, course_filter: str, week_filter: str) -> list[dict[str, Any]]:
        """Hybrid retrieval with optional explicit metadata filters."""
        where = self._make_where(course_filter=course_filter, week_filter=week_filter)
        dense_ids = self._dense_search(query, where=where, n_results=max(top_k, self.settings.dense_k))
        bm25_ids = self._bm25_search(
            query=query,
            course_filter=course_filter,
            week_filter=week_filter,
            n_results=max(top_k, self.settings.bm25_k),
        )
        fused_scores = _rrf_fuse(dense_ids, bm25_ids, k=max(1, self.settings.rrf_k))
        ranked_ids = sorted(fused_scores.keys(), key=lambda chunk_id: fused_scores[chunk_id], reverse=True)
        ranked_ids = ranked_ids[: max(1, top_k)]
        text_hits = [self._hydrate_hit(chunk_id, score=fused_scores[chunk_id]) for chunk_id in ranked_ids]

        if self.settings.rag_mode != RAGPipelineMode.MULTIMODAL_RETRIEVAL_MLLM:
            return text_hits

        image_scores = self._image_search(
            query=query,
            where=where,
            n_results=max(top_k, self.settings.multimodal_fusion_k),
        )
        multimodal_hits = self._multimodal_fuse_scores(text_hits=text_hits, image_scores=image_scores)
        return multimodal_hits[: max(1, top_k)]


def get_retriever() -> CourseRetriever:
    """Return singleton C1 retriever."""
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = CourseRetriever(get_settings())
    return _RETRIEVER


@tool
def kb_course_retrieval(
    query: str = "",
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
    explicit_course = course_filter.strip().lower()
    resolved_course = explicit_course
    if not resolved_course and settings.course_filter_enabled:
        resolved_course = _detect_course_filter(query)
    resolved_week = week_filter.strip().lower() or _detect_week_filter(query)

    normalized_query = query.strip()
    if not normalized_query:
        fallback_parts = ["course materials"]
        if resolved_course:
            fallback_parts.append(resolved_course.replace("-", " "))
        if resolved_week:
            fallback_parts.append(resolved_week.replace("-", " "))
        normalized_query = " ".join(fallback_parts)

    hits = get_retriever().search(
        query=normalized_query,
        top_k=max(1, int(top_k)),
        course_filter=resolved_course,
        week_filter=resolved_week,
    )

    if not hits:
        return "COURSE_RETRIEVAL_RESULTS:\nNo grounded snippets found for the current filters."

    lines: list[str] = ["C1A_RETRIEVAL_RESULTS:"]
    lines.append(
        f"Applied filters -> course={resolved_course or 'none'} | week={resolved_week or 'none'} | top_k={max(1, int(top_k))}"
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
    """
    C2 duration estimator: infer study minutes from grounded metadata and retrieved evidence.

    Args:
        task_query: User task to estimate.
        course_filter: Optional explicit course slug.
        week_filter: Optional week tag (for example, week-03).
        day_filter: Optional day constraint (for example, Tue).
        target_outcome: Optional task intent hint (review, quiz, assignment, project).
    """
    normalized_task_query = task_query.strip() or query.strip()
    if not normalized_task_query:
        return "C2_DURATION_ESTIMATE:\nquery field was missing."

    settings = get_settings()
    resolved_course = course_filter.strip().lower()
    if not resolved_course and settings.course_filter_enabled:
        resolved_course = _detect_course_filter(normalized_task_query)
    resolved_week = _resolve_week_filter(week_filter, query=normalized_task_query)
    resolved_day = _normalize_day_filter(day_filter) or _detect_day_filter(normalized_task_query)
    outcome = target_outcome.strip().lower() or "unspecified"

    candidate_docs = [
        row
        for row in get_document_rows()
        if (not resolved_course or str(row.get("course", "")).strip().lower() == resolved_course)
        and (not resolved_week or str(row.get("week", "")).strip().lower() == resolved_week)
    ][:20]
    doc_lines: list[str] = []
    for row in candidate_docs[:8]:
        doc_lines.append(
            (
                f"- doc_id={row.get('doc_id', '')}, title={row.get('title', '')}, "
                f"resource_type={row.get('resource_type', '')}, modality={row.get('modality', '')}, "
                f"difficulty={row.get('difficulty', '')}, cognitive_load={row.get('cognitive_load', '')}, "
                f"course={row.get('course', '')}, week={row.get('week', '')}"
            )
        )

    hits = get_retriever().search(
        query=normalized_task_query,
        top_k=max(3, min(8, int(settings.hybrid_k))),
        course_filter=resolved_course,
        week_filter=resolved_week,
    )
    retrieval_lines: list[str] = []
    for hit in hits[:5]:
        metadata = hit.get("metadata", {}) or {}
        snippet = str(hit.get("text", "")).replace("\n", " ").strip()
        if len(snippet) > 220:
            snippet = snippet[:220].rstrip() + "..."
        retrieval_lines.append(
            (
                f"- source={Path(str(metadata.get('source_path', 'unknown'))).name}, "
                f"resource_type={metadata.get('resource_type', '')}, difficulty_hint={metadata.get('difficulty', '')}, "
                f"cognitive_hint={metadata.get('cognitive_load', '')}, snippet={snippet}"
            )
        )
    pressure_lines = _build_assignment_pressure_lines(
        course_filter=resolved_course,
        week_filter=resolved_week,
    )

    prompt_messages = [
        SystemMessage(
            content=(
                "You estimate study duration for a personal planner.\n"
                "Use only the grounded evidence provided by the user.\n"
                "Return strict JSON only with keys:\n"
                "- estimated_minutes (integer)\n"
                "- confidence (one of: low, medium, high)\n"
                "- rationale (short string, mention key evidence signals)\n"
                "- suggested_blocking (string, e.g., '2 x 60min').\n"
                "Do not call any tools."
            )
        ),
        HumanMessage(
            content=(
                f"task_query={normalized_task_query}\n"
                f"target_outcome={outcome}\n"
                f"resolved_course={resolved_course or 'none'}\n"
                f"resolved_week={resolved_week or 'none'}\n"
                f"resolved_day={resolved_day or 'none'}\n\n"
                "documents.csv candidates:\n"
                f"{chr(10).join(doc_lines) if doc_lines else '- none'}\n\n"
                "retrieved evidence snippets:\n"
                f"{chr(10).join(retrieval_lines) if retrieval_lines else '- none'}\n\n"
                "assignment pressure summary:\n"
                f"{chr(10).join(pressure_lines) if pressure_lines else '- none'}\n"
            )
        ),
    ]
    raw_response = _render_assistant_content(get_reasoning_llm().invoke(prompt_messages).content)
    payload = _extract_first_json_object(raw_response)
    if not payload or "estimated_minutes" not in payload:
        return (
            "C2_DURATION_ESTIMATE:\n"
            "Estimator did not return valid JSON output. "
            "Ask the agent to retry the duration estimation."
        )

    estimated_minutes = _clamp_int(_safe_int(payload.get("estimated_minutes"), 0), 30, 240)
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
            f"Applied filters -> course={resolved_course or 'none'} | "
            f"week={resolved_week or 'none'} | day={resolved_day or 'none'} | outcome={outcome}"
        ),
        f"REQUIRED_MINUTES={estimated_minutes}",
        f"RESOLVED_WEEK_FILTER={resolved_week or ''}",
        f"RESOLVED_DAY_FILTER={resolved_day or ''}",
        f"ESTIMATE_JSON={json.dumps(result_payload, ensure_ascii=True)}",
    ]
    return "\n".join(lines)


@tool
def find_free_slots(
    week_filter: str,
    required_minutes: int,
    day_filter: str = "",
    min_slot_minutes: int = 30,
    day_start: str = "07:00",
    day_end: str = "22:00",
) -> str:
    """
    C3 free slot finder: compute free planner intervals in a target week/day.

    Args:
        week_filter: Dataset week tag (week-01..week-07 or easter-week).
        required_minutes: Estimated study minutes to fit.
        day_filter: Optional day (Mon/Tue/...).
        min_slot_minutes: Minimum candidate slot duration.
        day_start: Day bound start (HH:MM).
        day_end: Day bound end (HH:MM).
    """
    resolved_week = _resolve_week_filter(week_filter)
    if not resolved_week:
        return "C3_FREE_SLOTS:\nweek_filter field was missing."

    resolved_day = _normalize_day_filter(day_filter)
    required = _clamp_int(_safe_int(required_minutes, 60), 15, 360)
    min_slot = _clamp_int(_safe_int(min_slot_minutes, 30), 15, 240)
    start_min = _parse_hhmm_to_minutes(day_start)
    end_min = _parse_hhmm_to_minutes(day_end)
    if start_min is None or end_min is None or start_min >= end_min:
        return "C3_FREE_SLOTS:\nInvalid day_start/day_end bounds."

    week_rows = _filter_tasks_for_week(get_task_rows(), resolved_week)
    if resolved_day:
        week_rows = [row for row in week_rows if str(row.get("day", "")).strip() == resolved_day]
    if not week_rows:
        return (
            "C3_FREE_SLOTS:\n"
            f"Applied filters -> week={resolved_week} | day={resolved_day or 'all'}\n"
            "No planner rows matched the selected week/day."
        )

    by_date: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    parsed_dates: list[date] = []
    for row in week_rows:
        date_value = str(row.get("date", "")).strip()
        if not date_value:
            continue
        try:
            parsed_dates.append(datetime.strptime(date_value, "%Y-%m-%d").date())
        except ValueError:
            continue
    if parsed_dates:
        cursor_date = min(parsed_dates)
        max_date = max(parsed_dates)
        while cursor_date <= max_date:
            day_abbr = cursor_date.strftime("%a")
            if not resolved_day or day_abbr == resolved_day:
                by_date[(cursor_date.isoformat(), day_abbr)] = []
            cursor_date = date.fromordinal(cursor_date.toordinal() + 1)

    for row in week_rows:
        date_value = str(row.get("date", "")).strip()
        day_value = str(row.get("day", "")).strip()
        task_start = _parse_hhmm_to_minutes(str(row.get("start_time", "")))
        task_end = _parse_hhmm_to_minutes(str(row.get("end_time", "")))
        if task_start is None or task_end is None or task_end <= task_start:
            continue
        clipped_start = max(start_min, task_start)
        clipped_end = min(end_min, task_end)
        if clipped_end <= clipped_start:
            continue
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
            f"Applied filters -> week={resolved_week} | day={resolved_day or 'all'}\n"
            "No free slots found within the selected day bounds."
        )

    fitting = [slot for slot in slots if bool(slot.get("fits_required"))]
    fitting.sort(key=lambda slot: (str(slot.get("date")), int(slot.get("start", 0)), -int(slot.get("duration", 0))))
    partial = sorted(slots, key=lambda slot: int(slot.get("duration", 0)), reverse=True)

    lines = [
        "C3_FREE_SLOTS:",
        (
            f"Applied filters -> week={resolved_week} | day={resolved_day or 'all'} | "
            f"required_minutes={required} | day_bounds={day_start}-{day_end}"
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


TOOLS = [kb_course_retrieval, estimate_study_duration, find_free_slots]


SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a study planner assistant with three capabilities.\n"
        "C1 TopicSummary: summarize topics studied in a course/week using kb_course_retrieval.\n"
        "C2 DurationEstimator: estimate how many minutes a course-related task should take.\n"
        "C3 FreeSlotFinder: find open schedule slots in a target week/day given required_minutes.\n"
        "Respect dataset week tags like week-01..week-07 and easter-week.\n"
        "Routing rules:\n"
        "- If user asks to estimate time/effort/duration, call estimate_study_duration first.\n"
        "- If user asks to summarize topics/content, call kb_course_retrieval first.\n"
        "- If user asks for scheduling availability, call find_free_slots first.\n"
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
    """LLM node: routes tool calls and composes final C1/C2/C3 responses."""
    messages = state["messages"]
    user_query = _latest_user_text(messages)

    if messages and getattr(messages[-1], "type", None) == "tool":
        settings = get_settings()
        tool_name = str(getattr(messages[-1], "name", "")).strip()
        tool_result = str(messages[-1].content)
        if "query field was missing" in tool_result.lower() and user_query.strip():
            logging.getLogger(__name__).warning(
                "tool_retry missing_query -> retrying kb_course_retrieval with latest user query"
            )
            try:
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
    return {"messages": messages + [response]}


def route_after_agent(state: AgentState):
    """Route to tool execution if the model emitted tool calls."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
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
    print("\nPlanner agent (C1 + C2 + C3) ready. Type 'exit' to quit.\n")
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
