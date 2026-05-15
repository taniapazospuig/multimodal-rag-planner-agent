"""
Minimal planner agent with six capability labels:
- C1 CourseQA / TopicSummary (course materials only, via kb_course_retrieval)
- C2 DurationEstimator (estimate study time for a course-related task)
- C3 FreeSlotFinder (find open slots in a target week/day)
- C4 StudyEnvironmentRecommender (suggest where to do a task)
- C5 StudyEnvironmentFit (judge a specific environment for a task)
- C6 ResearchQA (``course=external-knowledge`` KB only, via kb_external_research_retrieval)

This module intentionally reuses already-built retrieval artifacts:
- Chroma DB in `chroma_db` (dense index)
- BM25 corpus JSONL from `Settings.text_bm25_path`

It does not perform indexing or backfill at runtime.
"""

from __future__ import annotations

import base64
import copy
import csv
import json
import logging
import re
import uuid
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
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from config import RAGPipelineMode, Settings, load_settings
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
    planner_memory: dict[str, Any]
    pending_tool_calls: list[dict[str, Any]]
    completed_tool_results: list[dict[str, str]]


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


KB_TAG_COURSE_CONTENT = "course-content"
KB_COURSE_EXTERNAL_KNOWLEDGE = "external-knowledge"


_DOC_IDS_BY_TAG_CACHE: dict[str, frozenset[str]] = {}


def get_doc_ids_with_tag(tags_value: str) -> frozenset[str]:
    """Return ``doc_id`` values from ``documents.csv`` whose ``tags`` column equals this value."""
    normalized = tags_value.strip().lower()
    if not normalized:
        return frozenset()
    cached = _DOC_IDS_BY_TAG_CACHE.get(normalized)
    if cached is not None:
        return cached
    matched: set[str] = set()
    for row in get_document_rows():
        if str(row.get("tags", "")).strip().lower() != normalized:
            continue
        doc_id = str(row.get("doc_id", "")).strip()
        if doc_id:
            matched.add(doc_id)
    frozen = frozenset(matched)
    _DOC_IDS_BY_TAG_CACHE[normalized] = frozen
    return frozen


def get_external_paper_rows() -> list[dict[str, str]]:
    """Return cached external-paper catalog rows (themes, paths)."""
    global _EXTERNAL_PAPER_ROWS
    if _EXTERNAL_PAPER_ROWS is None:
        _EXTERNAL_PAPER_ROWS = _load_csv_rows(EXTERNAL_PAPERS_CSV_PATH)
    return _EXTERNAL_PAPER_ROWS


def _paper_id_to_doc_id_map() -> dict[str, str]:
    """Map catalog paper_id (e.g. extp_001) to chunk doc_id using aligned source paths."""
    global _PAPER_ID_TO_DOC_ID
    if _PAPER_ID_TO_DOC_ID is not None:
        return _PAPER_ID_TO_DOC_ID

    doc_by_path: dict[str, str] = {}
    for row in get_document_rows():
        if str(row.get("resource_type", "")).strip() != EXTERNAL_PAPER_RESOURCE_TYPE:
            continue
        path_key = str(row.get("source_path", "")).strip().replace("\\", "/")
        if path_key:
            doc_by_path[path_key] = str(row.get("doc_id", "")).strip()

    mapping: dict[str, str] = {}
    for paper in get_external_paper_rows():
        pid = str(paper.get("paper_id", "")).strip()
        fp = str(paper.get("file_path", "")).strip().replace("\\", "/")
        if pid and fp and fp in doc_by_path:
            mapping[pid] = doc_by_path[fp]

    _PAPER_ID_TO_DOC_ID = mapping
    return mapping


def _external_doc_ids_matching_theme(hint: str) -> frozenset[str]:
    """Return doc_ids whose catalog row matches theme/title against the hint."""
    normalized = _normalize_for_match(hint)
    if not normalized:
        return frozenset()
    pmap = _paper_id_to_doc_id_map()
    out: set[str] = set()
    for paper in get_external_paper_rows():
        blob = _normalize_for_match(
            " ".join(
                [
                    str(paper.get("theme_primary", "")),
                    str(paper.get("theme_secondary", "")),
                    str(paper.get("title_short", "")),
                ]
            )
        )
        if not blob:
            continue
        hinted_tokens = [w for w in normalized.split() if len(w) > 2]
        token_match = bool(hinted_tokens) and all(t in blob for t in hinted_tokens)
        substring_match = normalized in blob or blob in normalized
        if not (substring_match or token_match):
            continue
        pid = str(paper.get("paper_id", "")).strip()
        doc_id = pmap.get(pid, "")
        if doc_id:
            out.add(doc_id)
    return frozenset(out)


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
_EXTERNAL_PAPER_ROWS: list[dict[str, str]] | None = None
_PAPER_ID_TO_DOC_ID: dict[str, str] | None = None
_STUDY_ENVIRONMENT_ROWS: list[dict[str, Any]] | None = None

DOCUMENTS_CSV_PATH = _BASE_DIR / "data" / "kb" / "metadata" / "documents.csv"
EXTERNAL_PAPERS_CSV_PATH = _BASE_DIR / "data" / "kb" / "metadata" / "external_papers.csv"
STUDY_ENVIRONMENTS_JSONL_PATH = _BASE_DIR / "data" / "kb" / "metadata" / "study_environments.jsonl"

EXTERNAL_PAPER_RESOURCE_TYPE = "external_paper"

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


def _ablation_uses_multimodal_retrieval(rag_mode: RAGPipelineMode) -> bool:
    """Whether retrieval should use image-index signals in this ablation mode."""
    return rag_mode == RAGPipelineMode.MULTIMODAL_RETRIEVAL_MLLM


def _ablation_uses_multimodal_understanding(rag_mode: RAGPipelineMode) -> bool:
    """Whether answer understanding should include image inputs in this ablation mode."""
    return rag_mode in {
        RAGPipelineMode.TEXT_RETRIEVAL_MLLM,
        RAGPipelineMode.MULTIMODAL_RETRIEVAL_MLLM,
    }


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


def _c6_ordered_extp_ids(user_query: str) -> list[str]:
    """Unique extp_* catalog ids in user text, stable order."""
    found = re.findall(r"\b(extp_\d+)\b", user_query.lower())
    return list(dict.fromkeys(found))


def _c6_paper_id_to_force(user_query: str) -> str:
    """If the user names any extp_* id, use the first for kb_external_research_retrieval.paper_id."""
    ids = _c6_ordered_extp_ids(user_query)
    return ids[0] if ids else ""


def _c6_is_single_paper_user_query(user_query: str) -> bool:
    """Exactly one distinct extp_* in the user message → single-paper scope for C6 checks."""
    return len(_c6_ordered_extp_ids(user_query)) == 1


def _retrieval_allowed_source_files(tool_result: str) -> set[str]:
    """Filenames from ``source_file=`` lines in a kb_course_retrieval / kb_external_research_retrieval blob."""
    return set(re.findall(r"source_file=(\S+)", tool_result))


def _bracket_file_citations(answer_text: str) -> list[str]:
    """Bracketed ``[name.ext]`` spans that look like filenames (one extension dot; no path slashes)."""
    return re.findall(r"\[([^\]\n/]+\.[^\]\n/]+)\]", answer_text)


def _bracket_file_citations_match_retrieval(answer_text: str, tool_result: str) -> bool:
    """Every such bracketed filename in the answer must exactly match a source_file= from this blob (C1/C6)."""
    cited = _bracket_file_citations(answer_text)
    if not cited:
        return True
    allowed = _retrieval_allowed_source_files(tool_result)
    return all(c in allowed for c in cited)


def _aimessage_with_forced_c6_paper_id(response: AIMessage, user_query: str) -> AIMessage:
    forced = _c6_paper_id_to_force(user_query)
    if not forced:
        return response
    tcalls = getattr(response, "tool_calls", None) or []
    if not tcalls:
        return response
    new_calls: list[Any] = []
    touched = False
    for tc in tcalls:
        if not isinstance(tc, dict):
            new_calls.append(tc)
            continue
        if str(tc.get("name", "")).strip() != "kb_external_research_retrieval":
            new_calls.append(tc)
            continue
        args = dict(tc.get("args") or {})
        args["paper_id"] = forced
        new_calls.append({**tc, "args": args})
        touched = True
    if not touched:
        return response
    return response.model_copy(update={"tool_calls": new_calls})


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


def _resolve_week_filter(
    raw_week_filter: str,
    query: str = "",
    *,
    infer_from_query: bool = True,
) -> str:
    """Resolve week filter from explicit tool input; optionally infer from query text."""
    candidate = _canonicalize_week_filter(raw_week_filter, allow_bare_number=True)
    if candidate:
        return candidate
    if not infer_from_query:
        return ""
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
        "kb_external_research_retrieval": ("query",),
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
    Unified retrieval over indexed ``text_chunks`` + BM25 (and planner_images fusion where configured).
    C1 restricts to rows tagged ``course-content`` in ``documents.csv``; C6 to ``course=external-knowledge``.
    Other callers omit doc allowlists when searching planner or study-environment corpora.

    Text evidence always combines dense Chroma retrieval with BM25 via reciprocal rank fusion.
    Settings.rag_mode controls whether image retrieval/fusion is applied
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
        if self.collection.count() <= 0:
            return []
        query_embedding = self.backbone.encode_text([query])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=max(1, n_results),
            where=where,
        )
        return [str(chunk_id) for chunk_id in ((results.get("ids") or [[]])[0] or [])]

    def _bm25_search(
        self,
        query: str,
        course_filter: str,
        week_filter: str,
        n_results: int,
        restrict_doc_ids: frozenset[str] | None = None,
    ) -> list[str]:
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
            if restrict_doc_ids is not None:
                doc = str(meta.get("doc_id", "")).strip()
                if doc not in restrict_doc_ids:
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
        """Retrieve planner screenshot image paths for C3 multimodal extraction.

        Week is added to the Chroma ``where`` clause only when
        ``Settings.retrieval_metadata_filter_enabled`` is true.
        """
        if self.image_collection.count() <= 0:
            return []
        use_week_meta = self.settings.retrieval_metadata_filter_enabled
        where: dict[str, Any] = {"course": "personal-planner"}
        if week_filter and use_week_meta:
            where = {"$and": [{"course": "personal-planner"}, {"week": week_filter}]}
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
        enable_multimodal_fusion: bool | None = None,
        restrict_doc_ids: frozenset[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Run dense + BM25 text retrieval fused with RRF, then optional image fusion per ``rag_mode``.

        When ``restrict_doc_ids`` is set (e.g. C1 CourseQA), fused hits are limited to chunks whose
        ``doc_id`` is in ``documents.csv`` with that cohort (typically ``course-content`` tags).
        """
        if restrict_doc_ids is not None and len(restrict_doc_ids) == 0:
            return []

        use_meta = self.settings.retrieval_metadata_filter_enabled
        eff_course = course_filter if use_meta else ""
        eff_week = week_filter if use_meta else ""
        where = self._make_where(course_filter=eff_course, week_filter=eff_week)

        dense_where: dict[str, Any] | None = where
        if restrict_doc_ids is not None:
            doc_clause = {"doc_id": {"$in": sorted(restrict_doc_ids)}}
            if dense_where is None:
                dense_where = doc_clause
            else:
                dense_where = {"$and": [dense_where, doc_clause]}

        _trace_log(
            "retrieval.search query=%r top_k=%d use_meta=%s eff_course=%r eff_week=%r "
            "restrict_doc_ids=%s where=%s",
            query,
            max(1, int(top_k)),
            use_meta,
            eff_course,
            eff_week,
            None if restrict_doc_ids is None else len(restrict_doc_ids),
            where if where is not None else "none",
        )
        dense_ids = self._dense_search(
            query,
            where=dense_where,
            n_results=max(top_k, self.settings.dense_k),
        )
        bm25_ids = self._bm25_search(
            query=query,
            course_filter=eff_course,
            week_filter=eff_week,
            n_results=max(top_k, self.settings.bm25_k),
            restrict_doc_ids=restrict_doc_ids,
        )
        _trace_log(
            "retrieval.candidates dense_bm25_rrf dense=%d bm25=%d dense_head=%s bm25_head=%s",
            len(dense_ids),
            len(bm25_ids),
            dense_ids[:3],
            bm25_ids[:3],
        )
        fused_scores = _rrf_fuse(dense_ids, bm25_ids, k=max(1, self.settings.rrf_k))

        ranked_all = sorted(
            fused_scores.keys(),
            key=lambda chunk_id: fused_scores[chunk_id],
            reverse=True,
        )
        text_hits: list[dict[str, Any]] = []
        for chunk_id in ranked_all:
            if len(text_hits) >= max(1, top_k):
                break
            hit = self._hydrate_hit(chunk_id, score=fused_scores.get(chunk_id, 0.0))
            if restrict_doc_ids is not None:
                meta = hit.get("metadata", {}) or {}
                doc = str(meta.get("doc_id", "")).strip()
                if doc not in restrict_doc_ids:
                    continue
            text_hits.append(hit)
        _trace_log(
            "retrieval.ranked kept=%s head_chunk_ids=%s",
            len(text_hits),
            [h.get("chunk_id") for h in text_hits[:3]],
        )

        multimodal_enabled = (
            self.settings.rag_mode == RAGPipelineMode.MULTIMODAL_RETRIEVAL_MLLM
            if enable_multimodal_fusion is None
            else bool(enable_multimodal_fusion)
        )
        if not multimodal_enabled:
            return text_hits

        image_where: dict[str, Any] | None = where
        if restrict_doc_ids is not None:
            doc_clause = {"doc_id": {"$in": sorted(restrict_doc_ids)}}
            if image_where is None:
                image_where = doc_clause
            else:
                image_where = {"$and": [image_where, doc_clause]}

        image_scores = self._image_search(
            query=query,
            where=image_where,
            n_results=max(top_k, self.settings.multimodal_fusion_k),
        )
        multimodal_hits = self._multimodal_fuse_scores(text_hits=text_hits, image_scores=image_scores)
        _trace_log("retrieval.multimodal final_hits=%d", len(multimodal_hits))
        return multimodal_hits[: max(1, top_k)]

    def search_external_knowledge(
        self,
        query: str,
        top_k: int,
        *,
        allowed_doc_ids: frozenset[str] | None = None,
        enable_multimodal_fusion: bool | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieval over KB chunks whose indexed metadata ``course`` is ``external-knowledge`` (C6 Research QA).

        Uses dense + BM25 fused with RRF over the external-knowledge course. ``allowed_doc_ids``
        further narrows to those ``doc_id`` values when non-``None``.
        """
        if allowed_doc_ids is not None and len(allowed_doc_ids) == 0:
            return []

        restrict_doc_ids: frozenset[str] | None
        if allowed_doc_ids is not None:
            restrict_doc_ids = frozenset(
                doc for doc in allowed_doc_ids if str(doc).strip()
            )
            if not restrict_doc_ids:
                return []
        else:
            restrict_doc_ids = None

        ext_where: dict[str, Any] = {"course": KB_COURSE_EXTERNAL_KNOWLEDGE}
        if restrict_doc_ids is not None:
            doc_clause = {"doc_id": {"$in": sorted(restrict_doc_ids)}}
            ext_where = {"$and": [ext_where, doc_clause]}

        dense_ids = self._dense_search(
            query, where=ext_where, n_results=max(top_k, self.settings.dense_k)
        )
        bm25_ids = self._bm25_search(
            query=query,
            course_filter=KB_COURSE_EXTERNAL_KNOWLEDGE,
            week_filter="",
            n_results=max(top_k, self.settings.bm25_k),
            restrict_doc_ids=restrict_doc_ids,
        )
        _trace_log(
            "retrieval.external dense_bm25_rrf dense=%d bm25=%d dense_head=%s bm25_head=%s",
            len(dense_ids),
            len(bm25_ids),
            dense_ids[:3],
            bm25_ids[:3],
        )
        fused_scores = _rrf_fuse(dense_ids, bm25_ids, k=max(1, self.settings.rrf_k))

        ranked_all = sorted(
            fused_scores.keys(),
            key=lambda chunk_id: fused_scores[chunk_id],
            reverse=True,
        )
        text_hits: list[dict[str, Any]] = []
        for chunk_id in ranked_all:
            if len(text_hits) >= max(1, top_k):
                break
            hit = self._hydrate_hit(chunk_id, score=fused_scores.get(chunk_id, 0.0))
            if restrict_doc_ids is not None:
                meta = hit.get("metadata", {}) or {}
                doc_h = str(meta.get("doc_id", "")).strip()
                if doc_h not in restrict_doc_ids:
                    continue
            text_hits.append(hit)
        _trace_log(
            "retrieval.external ranked count=%d kept=%s head_hit_ids=%s",
            len(ranked_all),
            len(text_hits),
            [h.get("chunk_id") for h in text_hits[:3]],
        )

        multimodal_enabled = (
            self.settings.rag_mode == RAGPipelineMode.MULTIMODAL_RETRIEVAL_MLLM
            if enable_multimodal_fusion is None
            else bool(enable_multimodal_fusion)
        )
        if not multimodal_enabled:
            return text_hits

        image_scores = self._image_search(
            query=query,
            where=ext_where,
            n_results=max(top_k, self.settings.multimodal_fusion_k),
        )
        multimodal_hits = self._multimodal_fuse_scores(text_hits=text_hits, image_scores=image_scores)
        _trace_log("retrieval.external.multimodal final_hits=%d", len(multimodal_hits))
        return multimodal_hits[: max(1, top_k)]


def get_retriever() -> CourseRetriever:
    """Return singleton text/multimodal retriever (course + external KB)."""
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
    C1 CourseQA retrieval (internal): grounded snippets limited to KB documents tagged ``course-content``.
    When the assistant cites materials with ``[filename.ext]`` brackets, each tag must exactly copy a
    ``source_file=`` basename from these results (same spelling and punctuation).

    Args:
        query: User query/topic to retrieve evidence for.
        top_k: Number of evidence snippets to return.
        course_filter: Optional explicit course slug filter.
        week_filter: Optional explicit week filter (for example, week-03).
    """
    settings = get_settings()
    meta_on = settings.retrieval_metadata_filter_enabled
    resolved_course = _resolve_course_filter(
        course_filter,
        query,
        filter_enabled=meta_on,
    )
    resolved_week = _resolve_week_filter(week_filter, query=query, infer_from_query=meta_on)

    normalized_query = query.strip()
    if not normalized_query:
        return "COURSE_RETRIEVAL_RESULTS:\nquery field was missing."
    restrict_course_docs = get_doc_ids_with_tag(KB_TAG_COURSE_CONTENT)
    _trace_log(
        "c1.kb_course_retrieval query=%r tool_course=%r tool_week=%r resolved_course=%r resolved_week=%r "
        "metadata_filter=%s course_content_doc_scope=%s",
        normalized_query,
        course_filter,
        week_filter,
        resolved_course,
        resolved_week,
        meta_on,
        len(restrict_course_docs),
    )

    # resolved_* feed retrieval only when metadata filtering is enabled (CourseRetriever.search).
    hits = get_retriever().search(
        query=normalized_query,
        top_k=max(1, int(top_k)),
        course_filter=resolved_course,
        week_filter=resolved_week,
        restrict_doc_ids=restrict_course_docs,
    )
    _trace_log("c1.retrieval hits=%d", len(hits))

    if not hits:
        return "COURSE_RETRIEVAL_RESULTS:\nNo grounded snippets found for the current filters."

    lines: list[str] = ["C1A_RETRIEVAL_RESULTS:"]
    applied_parts = [
        f"metadata_filter={'on' if meta_on else 'off'}",
    ]
    if meta_on:
        applied_parts.append(f"course={resolved_course or 'none'}")
        applied_parts.append(f"week={resolved_week or 'none'}")
    applied_parts.extend(
        [
            f"corpus_tags={KB_TAG_COURSE_CONTENT}",
            f"top_k={max(1, int(top_k))}",
            f"rag_mode={settings.rag_mode.value}",
        ]
    )
    lines.append("Applied filters -> " + " | ".join(applied_parts))
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
def kb_external_research_retrieval(
    query: str,
    top_k: int = 6,
    paper_id: str = "",
    theme_hint: str = "",
) -> str:
    """
    C6 Research QA (internal): grounded snippets from KB chunks with ``course=external-knowledge``.

    Use for evidence on task prioritization, study techniques and learning science,
    cognitive load / energy scheduling, breaks and recovery, habit formation, etc.

    Args:
        query: Research question or concepts to retrieve (required).
        top_k: Number of evidence snippets to return.
        paper_id: Optional catalog id from ``external_papers.csv`` (e.g. extp_004).
        theme_hint: Optional substring to bias toward themes in the catalog (e.g. spaced repetition, habits).
    """
    settings = get_settings()
    normalized_query = query.strip()
    if not normalized_query:
        return "C6_RESEARCH_QA_RETRIEVAL_RESULTS:\nquery field was missing."

    catalog_note = ""
    allow: frozenset[str] | None = None

    pid = str(paper_id).strip().lower()
    if pid:
        doc_for_paper = _paper_id_to_doc_id_map().get(pid, "")
        if not doc_for_paper:
            return (
                "C6_RESEARCH_QA_RETRIEVAL_RESULTS:\nUnknown paper_id. "
                "Use ids from external_papers.csv (e.g. extp_001)."
            )
        allow = frozenset({doc_for_paper})

    th = str(theme_hint).strip()
    if th:
        theme_docs = _external_doc_ids_matching_theme(th)
        if theme_docs:
            if allow is not None:
                inter = frozenset(doc for doc in allow if doc in theme_docs)
                if not inter:
                    return (
                        "C6_RESEARCH_QA_RETRIEVAL_RESULTS:\npaper_id does not match "
                        "theme_hint according to external_papers.csv; broaden or drop one filter."
                    )
                allow = inter
            else:
                allow = theme_docs
        else:
            catalog_note = (
                "Catalog note: theme_hint matched no rows in external_papers.csv; "
                f"searched full {KB_COURSE_EXTERNAL_KNOWLEDGE} corpus."
            )

    _trace_log(
        "c6.kb_external_research_retrieval query=%r paper_id=%r theme_hint=%r allow=%s",
        normalized_query,
        paper_id,
        theme_hint,
        None if allow is None else sorted(allow),
    )

    hits = get_retriever().search_external_knowledge(
        query=normalized_query,
        top_k=max(1, int(top_k)),
        allowed_doc_ids=allow,
    )
    _trace_log("c6.retrieval hits=%d", len(hits))

    if not hits:
        return (
            "C6_RESEARCH_QA_RETRIEVAL_RESULTS:\nNo grounded snippets found for "
            f"course={KB_COURSE_EXTERNAL_KNOWLEDGE} with current filters."
        )

    lines: list[str] = ["C6_RESEARCH_QA_RETRIEVAL_RESULTS:"]
    if catalog_note:
        lines.append(catalog_note)
    lines.append(
        f"Applied scope -> course={KB_COURSE_EXTERNAL_KNOWLEDGE}"
        f" | paper_id_filter={pid or 'none'} | theme_hint={th or 'none'}"
        f" | top_k={max(1, int(top_k))}"
        f" | rag_mode={settings.rag_mode.value}"
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
                f"doc_id={metadata.get('doc_id', '?')}"
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
    meta_on = settings.retrieval_metadata_filter_enabled
    resolved_course = _resolve_course_filter(
        course_filter,
        normalized_task_query,
        filter_enabled=meta_on,
    )
    resolved_week = _resolve_week_filter(
        week_filter, query=normalized_task_query, infer_from_query=meta_on
    )
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
        restrict_doc_ids=get_doc_ids_with_tag(KB_TAG_COURSE_CONTENT),
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
        _ablation_uses_multimodal_retrieval(settings.rag_mode),
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

    mllm_understanding_on = _ablation_uses_multimodal_understanding(settings.rag_mode)
    if mllm_understanding_on and image_paths:
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
        mllm_understanding_on and bool(image_paths),
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
    ]
    applied_parts = [
        f"metadata_filter={'on' if meta_on else 'off'}",
        f"mode={mode_name}",
    ]
    if meta_on:
        applied_parts.append(f"course={resolved_course or 'none'}")
        applied_parts.append(f"week={resolved_week or 'none'}")
    applied_parts.append(f"day={resolved_day or 'none'}")
    applied_parts.append(f"outcome={outcome}")
    lines.append("Applied filters -> " + " | ".join(applied_parts))
    lines.extend(
        [
            f"REQUIRED_MINUTES={estimated_minutes}",
            f"RESOLVED_WEEK_FILTER={resolved_week or ''}",
            f"RESOLVED_DAY_FILTER={resolved_day or ''}",
            f"ESTIMATE_JSON={json.dumps(result_payload, ensure_ascii=True)}",
        ]
    )
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
    enable_multimodal_retrieval: bool,
) -> list[dict[str, Any]]:
    """Extract planner events from indexed planner screenshot OCR snippets."""
    logger = logging.getLogger(__name__)
    settings = get_settings()
    normalized_week = week_filter.strip().lower()
    logger.info(
        "c2_debug screenshot_extract start query=%r week_filter=%r day_filter=%r top_k=%d include_images=%s multimodal_retrieval=%s",
        _truncate_for_log(query, 180),
        normalized_week,
        day_filter,
        max(6, int(top_k)),
        include_images,
        enable_multimodal_retrieval,
    )
    snippet_lines: list[str] = []
    image_paths: list[str] = []
    hits = get_retriever().search(
        query=query,
        top_k=max(6, int(top_k)),
        course_filter="personal-planner",
        week_filter=normalized_week,
        enable_multimodal_fusion=enable_multimodal_retrieval,
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
    if enable_multimodal_retrieval:
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
            f"Applied filters -> mode={mode_name} | "
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
    settings = get_settings()
    rag_mode = settings.rag_mode
    c3_mode_name = rag_mode.value
    multimodal_retrieval_on = _ablation_uses_multimodal_retrieval(rag_mode)
    mllm_understanding_on = _ablation_uses_multimodal_understanding(rag_mode)
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
        top_k=max(12, int(settings.hybrid_k) * 3),
        include_images=mllm_understanding_on,
        enable_multimodal_retrieval=multimodal_retrieval_on,
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
    """Query indexed study location photos in planner_images and return source_path scores."""
    retriever = get_retriever()
    if retriever.image_collection.count() <= 0:
        return {}
    query_embedding = retriever.backbone.encode_text([query])[0]
    where = {"course": "study-environment"}
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
    Shared helper: C4 (recommend) vs C5 (assess fit) study-environment tools.
    """
    targeted_mode = mode == "assess"
    header = "C5_STUDY_ENVIRONMENT" if targeted_mode else "C4_STUDY_ENVIRONMENT"
    raw_query = query.strip() or task_query.strip()
    normalized_query = raw_query
    if not normalized_query:
        return f"{header}:\nquery field was missing."
    env_rows = get_study_environment_rows()
    if not env_rows:
        return f"{header}:\nNo study environments metadata found."

    settings = get_settings()
    text_scores, text_excerpts = _query_study_environment_text_scores(
        normalized_query,
        top_k=max(8, int(top_k) * 3),
    )
    multimodal_retrieval_on = _ablation_uses_multimodal_retrieval(settings.rag_mode)
    mllm_understanding_on = _ablation_uses_multimodal_understanding(settings.rag_mode)
    image_scores = (
        _query_study_environment_image_scores(normalized_query, top_k=max(6, int(top_k) * 2))
        if multimodal_retrieval_on
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
        if multimodal_retrieval_on:
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
        return f"{header}:\nNo candidate environments found."

    targeted_environment_id = (
        _resolve_environment_hint(environment_hint or raw_query, env_rows) if targeted_mode else ""
    )
    if targeted_mode and not targeted_environment_id:
        return (
            f"{header}:\n"
            "Could not resolve the requested environment. "
            "Provide environment_hint using environment_id or environment_type."
        )
    image_grounding_guidance = (
        "When image evidence is available, explicitly ground your reasoning in observable "
        "image characteristics (for example: visible natural light, crowd density, desk/chair setup, "
        "indoor vs outdoor context, and distraction cues).\n"
        if mllm_understanding_on
        else ""
    )
    evidence_mode = (
        "multimodal_index+images"
        if multimodal_retrieval_on
        else ("text_index+images" if mllm_understanding_on else "text_index_only")
    )
    llm_candidates = candidates[: max(3, min(8, int(top_k) * 2))]
    recommendation_selection_rules = (
        ""
        if targeted_mode
        else (
            "Recommendation mode: choose environment_id by comparing the task_query to each candidate's "
            "environment_type, coarse_geo, and retrieval evidence (and images when present). "
            "Weigh tradeoffs (focus vs collaboration, quiet vs background noise, indoor screen work vs "
            "outdoor/light movement, privacy, group work, exam conditions, etc.). "
            "Do not default to private_study_room (env_03) or any single type—select it only when the task and "
            "evidence clearly favor a quiet solo enclosed space. If another candidate is a better fit, pick that one.\n"
        )
    )
    llm_prompt = [
        SystemMessage(
            content=(
                "You are analyzing study-environment suitability for a task.\n"
                "You only have sparse metadata (environment_type/coarse_geo) plus retrieval evidence.\n"
                "Infer likely task fit factors (internet connection, natural light, interruptions, "
                "noise, seating comfort) from available evidence. In multimodal mode, use image evidence when present.\n"
                f"{recommendation_selection_rules}"
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
                f"evidence_mode={evidence_mode}\n"
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
        "observable_image_evidence": image_evidence_items if mllm_understanding_on else [],
        "verdict": verdict,
        "assessment_mode": "targeted" if targeted_mode else "recommendation",
        "confidence": confidence,
    }

    lines = [
        f"{header}:",
        (
            f"Applied mode -> rag_mode={settings.rag_mode.value} | "
            f"evidence_mode={evidence_mode} | top_k={max(1, int(top_k))}"
        ),
        f"query={normalized_query}",
        f"ASSESSMENT_MODE={'targeted' if targeted_mode else 'recommendation'}",
        f"TARGET_ENVIRONMENT_ID={targeted_environment_id or ''}",
        f"MULTIMODAL_IMAGE_GROUNDING_REQUIRED={1 if mllm_understanding_on else 0}",
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

    if mllm_understanding_on:
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
    C4 environment recommender: choose where a task should be done.

    Ablation behavior:
    - text_only: uses study_environments.jsonl + indexed text retrieval for study-location photos.
    - text_retrieval_mllm: keeps text retrieval, but passes candidate images to MLLM for final understanding.
    - multimodal_retrieval_mllm: additionally uses planner_images retrieval for candidate ranking and MLLM input.
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
    C5 targeted assessment: judge whether a specific environment is good for the task.

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
    kb_external_research_retrieval,
    estimate_study_duration,
    find_free_slots,
    recommend_study_environment,
    assess_study_environment_fit,
]


SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a study planner assistant with six capability labels C1–C5 and C6.\n"
        "C1 CourseQA / TopicSummary: summarize course/week material using kb_course_retrieval "
        "(only docs tagged ``course-content`` in ``documents.csv``—never syllabus-unrelated KB). "
        "If you use ``[filename.ext]`` evidence brackets, copy basenames only from retrieval ``source_file=`` lines.\n"
        "C6 ResearchQA: answer from indexed research using kb_external_research_retrieval "
        "(``course=external-knowledge`` papers: task prioritization, study techniques, cognitive load "
        "and scheduling energy, breaks and recovery, habit formation, behavior change). Optionally set "
        "theme_hint from the user wording and paper_id if they name an extp_* catalog id. "
        "If you use ``[filename.ext]`` evidence brackets, copy basenames only from retrieval ``source_file=`` lines.\n"
        "C2 DurationEstimator: estimate task minutes with estimate_study_duration.\n"
        "C3 FreeSlotFinder: find open schedule slots with find_free_slots.\n"
        "C4 StudyEnvironmentRecommender: use recommend_study_environment to choose where to work. "
        "Match the task to environment characteristics; there is no single best place for every task.\n"
        "C5 StudyEnvironmentFit: use assess_study_environment_fit to judge a specific environment for a task.\n"
        "Respect dataset week tags like week-01..week-07 and easter-week.\n"
        "Routing rules:\n"
        "- If user asks to estimate duration, call estimate_study_duration.\n"
        "- If user asks what was covered / course topics / lecture content, call kb_course_retrieval first (C1).\n"
        "- If user asks about research-backed study advice, prioritization psychology, spaced practice, recall, habits, breaks, fatigue, routines, "
        "or similar—not specific course syllabus content—call kb_external_research_retrieval first (C6).\n"
        "- If user asks for free slots, call find_free_slots.\n"
        "- If user asks where to do a task or asks for best study place/environment, call recommend_study_environment first (C4).\n"
        "- If user asks 'Is it a good idea to do this task in [environment]?', call assess_study_environment_fit first (C5) "
        "and preserve the environment mention in environment_hint.\n"
        "- If the user needs several planner capabilities in one message, you may emit multiple tool_calls in one "
        "assistant turn; they run **in order**, one tool per graph step, so each later call can rely on earlier "
        "tool outputs already present in the transcript.\n"
        "Always include required tool arguments."
    )
)


def _build_llm(settings: Settings):
    """Create OpenAI backend and bind available tools."""
    from langchain_openai import ChatOpenAI

    if not settings.openai_api_key:
        raise ValueError(
            "OpenAI API key is missing. Set OPENAI_API_KEY."
        )
    return ChatOpenAI(
        model=settings.openai_model,
        temperature=0,
        api_key=settings.openai_api_key,
    ).bind_tools(TOOLS)


def _build_reasoning_llm(settings: Settings):
    """Create a plain OpenAI client without bound tools."""
    from langchain_openai import ChatOpenAI

    if not settings.openai_api_key:
        raise ValueError(
            "OpenAI API key is missing. Set OPENAI_API_KEY."
        )
    return ChatOpenAI(
        model=settings.openai_model,
        temperature=0,
        api_key=settings.openai_api_key,
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
    """Latest non-empty HumanMessage string; used only for internal verifier heuristics."""
    for message in reversed(messages):
        if not isinstance(message, HumanMessage):
            continue
        content = message.content
        if isinstance(content, str):
            text = content.strip()
            if text:
                return text
    return ""


def _active_user_query(state: AgentState) -> str:
    """Latest user text, falling back to memory when ToolNode has replaced message history."""
    direct = _latest_user_text(state.get("messages", []))
    if direct:
        return direct
    mem = _get_planner_memory(state)
    return str(mem.get("active_user_query", "")).strip()


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


def _user_has_explicit_week_signal(user_query: str) -> bool:
    """Detect whether user text explicitly constrains a week-like time scope."""
    normalized = _normalize_for_match(user_query)
    if not normalized:
        return False
    if _detect_week_filter(normalized):
        return True
    relative_signals = ("this week", "next week", "tonight", "today", "tomorrow")
    return any(signal in normalized for signal in relative_signals)


def _query_has_explicit_calendar_date(user_query: str) -> bool:
    """True if the user names a concrete calendar day (not a bare weekday)."""
    normalized = _normalize_for_match(user_query)
    if not normalized:
        return False
    if re.search(r"\b20\d{2}-\d{2}-\d{2}\b", normalized):
        return True
    if re.search(r"\b\d{1,2}/\d{1,2}/(20)?\d{2}\b", normalized):
        return True
    # "31 mar", "mar 31", "31st march"
    months = (
        "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
    )
    if re.search(rf"\b\d{{1,2}}(st|nd|rd|th)?\s+({months})\b", normalized):
        return True
    if re.search(rf"\b({months})\s+\d{{1,2}}(st|nd|rd|th)?\b", normalized):
        return True
    return False


def _get_planner_memory(state: AgentState) -> dict[str, Any]:
    raw = state.get("planner_memory")
    return dict(raw) if isinstance(raw, dict) else {}


def _format_planner_memory_for_prompt(mem: dict[str, Any]) -> str:
    """Compact system text so the model can reuse resolved scope across steps and REPL turns."""
    if not mem:
        return ""
    lines: list[str] = []
    cap = str(mem.get("capability", "")).strip().upper()
    if cap:
        lines.append(f"Last planner capability: {cap}")
    course = str(mem.get("resolved_course", "")).strip()
    week = str(mem.get("resolved_week", "")).strip()
    day = str(mem.get("resolved_day", "")).strip()
    scope: list[str] = []
    if course:
        scope.append(f"course={course}")
    if week:
        scope.append(f"week={week}")
    if day:
        scope.append(f"day={day}")
    if scope:
        lines.append("Resolved scope: " + ", ".join(scope))
    tool = str(mem.get("last_tool_name", "")).strip()
    if tool:
        lines.append(f"Last tool: {tool}")
    if not lines:
        return ""
    return (
        "Planner memory (session continuity; internal context—not a user quotation):\n"
        + "\n".join(f"- {line}" for line in lines)
    )


def _parse_applied_filters_from_blob(text: str) -> dict[str, str]:
    """Best-effort course/week/day from tool transcript lines (Applied filters -> ...)."""
    course_m = re.search(r"\bcourse=([^\s|]+)", text, re.I)
    week_m = re.search(r"\bweek=([^\s|]+)", text, re.I)
    day_m = re.search(r"\bday=([^\s|]+)", text, re.I)

    def grab(match):
        if not match:
            return ""
        v = match.group(1).strip().lower()
        return "" if v in {"", "none", "unknown", "?"} else v

    return {
        "resolved_course": grab(course_m),
        "resolved_week": grab(week_m),
        "resolved_day": grab(day_m),
    }


def _tool_call_args_for_tool_message(messages: list[BaseMessage], tool_msg: BaseMessage) -> dict[str, Any]:
    """Args from the AIMessage tool call that produced this ToolMessage (matches tool_call_id when present)."""
    if getattr(tool_msg, "type", None) != "tool":
        return {}
    want_id = str(getattr(tool_msg, "tool_call_id", "") or "")
    tool_name = str(getattr(tool_msg, "name", "") or "").strip()
    idx = next((i for i, m in enumerate(messages) if m is tool_msg), -1)
    if idx < 0:
        return {}
    for i in range(idx - 1, -1, -1):
        m = messages[i]
        calls = getattr(m, "tool_calls", None) or []
        if not calls:
            continue
        fallback: dict[str, Any] | None = None
        for call in calls:
            if not isinstance(call, dict):
                continue
            if str(call.get("name", "")).strip() != tool_name:
                continue
            args = call.get("args")
            if not isinstance(args, dict):
                continue
            cid = str(call.get("id", ""))
            if want_id and cid and cid == want_id:
                return dict(args)
            fallback = dict(args)
        if fallback is not None:
            return fallback
        break
    return {}


_CAPABILITY_BY_TOOL: dict[str, str] = {
    "kb_course_retrieval": "c1",
    "kb_external_research_retrieval": "c6",
    "estimate_study_duration": "c2",
    "find_free_slots": "c3",
    "recommend_study_environment": "c4",
    "assess_study_environment_fit": "c5",
}


def _build_planner_memory_patch(
    tool_name: str,
    tool_result: str,
    call_args: dict[str, Any],
) -> dict[str, Any]:
    parsed = _parse_applied_filters_from_blob(tool_result)
    rc = str(call_args.get("course_filter", "")).strip().lower()
    rw = str(call_args.get("week_filter", "")).strip().lower()
    rd = str(call_args.get("day_filter", "")).strip().lower()
    patch: dict[str, Any] = {
        "capability": _CAPABILITY_BY_TOOL.get(tool_name, ""),
        "last_tool_name": tool_name,
        "last_tool_summary": _truncate_for_log(tool_result, 240),
        "resolved_course": rc or parsed["resolved_course"],
        "resolved_week": rw or parsed["resolved_week"],
        "resolved_day": rd or parsed["resolved_day"],
    }
    if tool_name == "estimate_study_duration":
        minutes = _required_minutes_from_text(tool_result)
        if minutes > 0:
            patch["last_estimated_minutes"] = minutes
    return patch


def _c3_has_time_anchor(
    user_query: str,
    tool_call_args: dict[str, Any],
    tool_result: str,
) -> bool:
    """
    Prefer structured scope from the tool call / resolved filters; fall back to user text
    (week phrases, calendar date). Bare weekday in user text alone is not enough.
    """
    wf = str(tool_call_args.get("week_filter", "")).strip().lower()
    df = str(tool_call_args.get("day_filter", "")).strip().lower()
    if wf and wf not in {"", "none", "unknown", "?"}:
        return True
    if df and df not in {"", "none", "unknown", "?", "all"}:
        return True

    parsed = _parse_applied_filters_from_blob(tool_result)
    wk = parsed.get("resolved_week", "")
    if wk:
        return True
    day_out = parsed.get("resolved_day", "")
    if day_out and day_out not in {"", "all", "none", "unknown", "?"}:
        return True

    if _user_has_explicit_week_signal(user_query) or _query_has_explicit_calendar_date(user_query):
        return True
    return False


def _c2_query_is_underspecified(user_query: str) -> bool:
    """Heuristic: duration request lacks concrete task scope/material detail."""
    normalized = _normalize_for_match(user_query)
    if not normalized:
        return True
    has_week_scope = _user_has_explicit_week_signal(normalized)
    has_specific_material = any(
        keyword in normalized
        for keyword in (
            "lecture",
            "tutorial",
            "assignment",
            "project",
            "exam",
            "quiz",
            "lab",
            "homework",
            "notes",
            "slides",
            "practice",
            "question",
            "report",
            "deliverable",
            "applied class",
        )
    )
    return not has_week_scope and not has_specific_material


def _json_assigned_on_line(text: str, key: str) -> Any | None:
    """
    Parse JSON from the first stripped line starting with ``key=<json>``.

    Tool transcripts use ``ESTIMATE_JSON=``, ``SLOTS_JSON=``, ``RECOMMENDATION_JSON=``
    (single-line ``json.dumps(..., ensure_ascii=True)`` payloads).
    """
    prefix = f"{key}="
    for raw in text.splitlines():
        s = raw.strip()
        if not s.startswith(prefix):
            continue
        tail = s[len(prefix) :].strip()
        if not tail:
            return None
        try:
            return json.loads(tail)
        except json.JSONDecodeError:
            return None
    return None


def _estimate_json_blob_invalid(text: str) -> str:
    """Return a reason code if ``ESTIMATE_JSON`` is missing/unparseable/invalid; else ``''``."""
    parsed = _json_assigned_on_line(text, "ESTIMATE_JSON")
    if parsed is None:
        return "invalid_estimate_json_parse"
    if not isinstance(parsed, dict):
        return "invalid_estimate_json_shape"
    minutes = _safe_int(parsed.get("estimated_minutes"), 0)
    if minutes <= 0:
        return "invalid_estimate_minutes_json"
    req = re.search(r"REQUIRED_MINUTES=(\d+)", text)
    if req and int(req.group(1)) != minutes:
        return "estimate_required_minutes_json_mismatch"
    conf = str(parsed.get("confidence", "")).strip().lower()
    if conf and conf not in {"low", "medium", "high"}:
        return "invalid_estimate_confidence_json"
    return ""


def _slots_json_blob_invalid(text: str) -> str:
    """Return a reason code if ``SLOTS_JSON`` is invalid for a successful C3 blob; else ``''``."""
    parsed = _json_assigned_on_line(text, "SLOTS_JSON")
    if parsed is None:
        return "invalid_slots_json_parse"
    if not isinstance(parsed, list) or len(parsed) == 0:
        return "invalid_slots_json_shape"
    for item in parsed:
        if not isinstance(item, dict):
            return "invalid_slots_json_item"
        for field in ("date", "day", "start", "end", "duration"):
            if field not in item:
                return "invalid_slots_json_fields"
        try:
            start_m = int(item["start"])
            end_m = int(item["end"])
            duration = int(item["duration"])
        except (TypeError, ValueError):
            return "invalid_slots_json_types"
        if end_m <= start_m or duration <= 0:
            return "invalid_slots_json_bounds"
        if duration != end_m - start_m:
            return "invalid_slots_json_duration_mismatch"
    return ""


def _recommendation_json_blob_invalid(text: str) -> str:
    """Return a reason code if ``RECOMMENDATION_JSON`` is invalid; else ``''``."""
    parsed = _json_assigned_on_line(text, "RECOMMENDATION_JSON")
    if parsed is None:
        return "invalid_recommendation_json_parse"
    if not isinstance(parsed, dict):
        return "invalid_recommendation_json_shape"
    env_id = str(parsed.get("environment_id", "")).strip()
    if not env_id:
        return "missing_environment_id_json"
    verdict = str(parsed.get("verdict", "")).strip().lower()
    if verdict not in {"yes", "maybe", "no"}:
        return "invalid_verdict_json"
    mode = str(parsed.get("assessment_mode", "")).strip().lower()
    if mode not in {"targeted", "recommendation"}:
        return "invalid_assessment_mode_json"
    fr = parsed.get("factor_rationale", [])
    if not isinstance(fr, list):
        return "invalid_factor_rationale_json"
    oie = parsed.get("observable_image_evidence", [])
    if not isinstance(oie, list):
        return "invalid_observable_image_evidence_json"
    conf = str(parsed.get("confidence", "")).strip().lower()
    if conf and conf not in {"high", "medium", "low"}:
        return "invalid_recommendation_confidence_json"
    return ""


def _tool_output_is_weak(
    tool_name: str,
    tool_result: str,
    user_query: str = "",
    *,
    tool_call_args: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Minimal post-tool verification for required payload fields and evidence strength."""
    text = tool_result.strip()
    lower = text.lower()
    weak_markers = (
        "query field was missing",
        "no grounded snippets found",
        "did not return valid json",
        "insufficient planner evidence",
        "no planner rows matched",
        "no candidate environments found",
        "could not resolve the requested environment",
        "unknown paper_id",
        "theme_hint according to external_papers.csv",
    )
    if any(marker in lower for marker in weak_markers):
        return True, "weak_evidence"

    if tool_name == "kb_course_retrieval":
        has_ranked_hits = re.search(r"(?m)^\d+\.\s+score=", text) is not None
        if "C1A_RETRIEVAL_RESULTS:" not in text or not has_ranked_hits:
            return True, "missing_retrieval_hits"
        # Require either explicit user constraints or non-empty resolved filters for broad topic-summary prompts.
        applied_match = re.search(r"Applied filters ->[^\n]*course=([^\s|]+)[^\n]*week=([^\s|]+)", text)
        resolved_course = (applied_match.group(1).strip().lower() if applied_match else "")
        resolved_week = (applied_match.group(2).strip().lower() if applied_match else "")
        no_course_week_filters = (
            resolved_course in {"", "none", "unknown", "?"}
            and resolved_week in {"", "none", "unknown", "?"}
        )
        query_has_scope = bool(_detect_course_filter(user_query)) or _user_has_explicit_week_signal(user_query)
        if no_course_week_filters and not query_has_scope:
            return True, "broad_query_without_scope_filters"
        return False, ""

    if tool_name == "kb_external_research_retrieval":
        has_ranked_hits = re.search(r"(?m)^\d+\.\s+score=", text) is not None
        if "C6_RESEARCH_QA_RETRIEVAL_RESULTS:" not in text or not has_ranked_hits:
            return True, "missing_retrieval_hits"
        if _c6_is_single_paper_user_query(user_query):
            files = re.findall(r"source_file=(\S+)", text)
            distinct = {f for f in files if f and f != "?"}
            if len(distinct) > 1:
                return True, "multi_source_single_paper_scope"
            expect = _c6_ordered_extp_ids(user_query)[0]
            got = str((tool_call_args or {}).get("paper_id", "")).strip().lower()
            if got != expect:
                return True, "paper_id_mismatch_single_paper"
        return False, ""

    if tool_name == "estimate_study_duration":
        if "C2_DURATION_ESTIMATE:" not in text or "ESTIMATE_JSON=" not in text:
            return True, "missing_estimate_json"
        match = re.search(r"REQUIRED_MINUTES=(\d+)", text)
        if not match or int(match.group(1)) <= 0:
            return True, "non_positive_minutes"
        week_match = re.search(r"Applied filters ->[^\n]*\bweek=([^\s|]+)", text)
        resolved_week = week_match.group(1).strip().lower() if week_match else ""
        if resolved_week in {"", "none", "unknown", "?"}:
            rw_line = re.search(r"(?m)^RESOLVED_WEEK_FILTER=(.*)$", text)
            if rw_line:
                resolved_week = rw_line.group(1).strip().lower()
        if resolved_week in {"", "none", "unknown", "?"}:
            wf_raw = str((tool_call_args or {}).get("week_filter", "")).strip()
            resolved_week = (
                _canonicalize_week_filter(wf_raw, allow_bare_number=True).strip().lower()
                if wf_raw
                else ""
            )
        outcome_match = re.search(r"Applied filters ->[^\n]*outcome=([^\s|]+)", text)
        resolved_outcome = outcome_match.group(1).strip().lower() if outcome_match else ""
        if (
            resolved_week in {"", "none", "unknown", "?"}
            and resolved_outcome in {"", "none", "unknown", "?", "unspecified"}
            and _c2_query_is_underspecified(user_query)
        ):
            return True, "underspecified_duration_request"
        bad_json = _estimate_json_blob_invalid(text)
        if bad_json:
            return True, bad_json
        return False, ""

    if tool_name == "find_free_slots":
        if "C3_FREE_SLOTS:" not in text or "SLOTS_JSON=" not in text:
            return True, "missing_slots_json"
        if not _c3_has_time_anchor(user_query, tool_call_args or {}, text):
            return True, "no_user_time_scope"
        bad_json = _slots_json_blob_invalid(text)
        if bad_json:
            return True, bad_json
        return False, ""

    if tool_name == "recommend_study_environment":
        if "C4_STUDY_ENVIRONMENT:" not in text or "RECOMMENDATION_JSON=" not in text:
            return True, "missing_environment_json"
        bad_json = _recommendation_json_blob_invalid(text)
        if bad_json:
            return True, bad_json
        return False, ""

    if tool_name == "assess_study_environment_fit":
        if "C5_STUDY_ENVIRONMENT:" not in text or "RECOMMENDATION_JSON=" not in text:
            return True, "missing_environment_json"
        bad_json = _recommendation_json_blob_invalid(text)
        if bad_json:
            return True, bad_json
        return False, ""

    return False, ""


def _clarifying_question_for_tool(tool_name: str, reason: str = "") -> str:
    """Short clarification when verification fails (no user-quote; keeps copy generic)."""
    if tool_name == "kb_course_retrieval":
        return (
            "I could not find strong grounded course evidence. "
            "Could you specify the course and week (for example, operations-research, week-03)?"
        )
    if tool_name == "kb_external_research_retrieval":
        if reason == "multi_source_single_paper_scope":
            return (
                "Retrieval spanned more than one paper but your message targets a single extp_* paper. "
                "Try again with one catalog id or broaden the question."
            )
        if reason == "paper_id_mismatch_single_paper":
            return (
                "The retrieval call did not use the extp_* id from your message. "
                "Please repeat the question; the agent should scope to that paper only."
            )
        return (
            "I could not find strong snippets in the external-knowledge research corpus. "
            "Could you rephrase with more specific keywords (topic or paper_id extp_* if relevant)?"
        )
    if tool_name == "estimate_study_duration":
        return (
            "I could not produce a reliable duration estimate. "
            "Could you clarify task scope and target outcome (for example, revision depth or deliverable)?"
        )
    if tool_name == "find_free_slots":
        return (
            "I could not verify reliable planner evidence for free-slot computation. "
            "Could you provide a specific week and optional day (for example, week-03, Tue)?"
        )
    if tool_name == "recommend_study_environment":
        return (
            "I need a bit more detail to recommend a study environment. "
            "Could you include the task type and any priority constraints (quiet, internet, duration)?"
        )
    if tool_name == "assess_study_environment_fit":
        return (
            "I need a bit more detail to assess environment fit. "
            "Could you include the task type, which environment you mean, and any priority constraints?"
        )
    return "I need a bit more detail to give a reliable grounded answer. Could you clarify?"


def _kb_tool_result_after_retry(
    tool_name: str,
    tool_result: str,
    user_query: str,
    settings: Settings,
) -> str:
    """Re-run KB tools with the latest user text when the model omitted query or got empty snippets."""
    lower_tool_result = tool_result.lower()
    missing_kb_query = (
        tool_name in {"kb_course_retrieval", "kb_external_research_retrieval"}
        and (
            "query field was missing" in lower_tool_result
            or ("field required" in lower_tool_result and "query" in lower_tool_result)
        )
    )
    kb_no_snippets = tool_name in {
        "kb_course_retrieval",
        "kb_external_research_retrieval",
    } and ("no grounded snippets found" in lower_tool_result)
    if not ((missing_kb_query or kb_no_snippets) and user_query.strip()):
        return tool_result
    logging.getLogger(__name__).warning(
        "tool_retry %s -> retrying with latest user query only "
        "(missing_query=%s empty_snippets=%s)",
        tool_name,
        bool(missing_kb_query),
        bool(kb_no_snippets),
    )
    try:
        _trace_log("agent_node retry %s with query=%r", tool_name, user_query.strip())
        if tool_name == "kb_course_retrieval":
            return str(
                kb_course_retrieval.invoke(
                    {
                        "query": user_query.strip(),
                        "top_k": max(1, settings.hybrid_k),
                    }
                )
            )
        payload: dict[str, Any] = {
            "query": user_query.strip(),
            "top_k": max(1, settings.hybrid_k),
        }
        fpid = _c6_paper_id_to_force(user_query)
        if fpid:
            payload["paper_id"] = fpid
        return str(kb_external_research_retrieval.invoke(payload))
    except Exception as exc:
        logging.getLogger(__name__).warning("tool_retry failed: %s", exc)
        return tool_result


def _tool_calls_dict_list(raw: Any) -> list[dict[str, Any]]:
    """Deep-copy tool call dicts from an AIMessage for safe queueing."""
    out: list[dict[str, Any]] = []
    for c in (raw or []):
        if isinstance(c, dict):
            out.append(copy.deepcopy(c))
    return out


def _tool_messages_since_latest_user(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Tool outputs produced for the current user turn, in transcript order."""
    start = 0
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            start = index + 1
            break
    return [m for m in messages[start:] if getattr(m, "type", None) == "tool"]


def _completed_tool_results(state: AgentState) -> list[dict[str, str]]:
    """Completed tool transcripts carried outside LangGraph's message replacement semantics."""
    output: list[dict[str, str]] = []
    for item in state.get("completed_tool_results") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        content = str(item.get("content", "")).strip()
        if name and content:
            output.append({"name": name, "content": content})
    return output


def _required_minutes_from_text(content: str) -> int:
    """Extract a C2 estimate from a tool transcript."""
    match = re.search(r"(?m)^REQUIRED_MINUTES=(\d+)\s*$", content)
    if match:
        return _safe_int(match.group(1), 0)
    estimate_payload = _json_assigned_on_line(content, "ESTIMATE_JSON")
    if isinstance(estimate_payload, dict):
        minutes = _safe_int(estimate_payload.get("estimated_minutes"), 0)
        if minutes > 0:
            return minutes
    return 0


def _latest_required_minutes(messages: list[BaseMessage]) -> int:
    """Most recent REQUIRED_MINUTES value from completed tool transcripts."""
    for message in reversed(messages):
        if getattr(message, "type", None) != "tool":
            continue
        minutes = _required_minutes_from_text(str(message.content))
        if minutes > 0:
            return minutes
    return 0


def _latest_required_minutes_from_completed(completed: list[dict[str, str]]) -> int:
    """Most recent C2 estimate from completed tool transcripts stored in state."""
    for item in reversed(completed):
        if item.get("name") != "estimate_study_duration":
            continue
        minutes = _required_minutes_from_text(str(item.get("content", "")))
        if minutes > 0:
            return minutes
    return 0


def _minutes_requested_in_query(user_query: str) -> int:
    """Extract a simple duration hint from user text for C3 when no C2 result exists yet."""
    normalized = _normalize_for_match(user_query)
    hour_match = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:hour|hours|hr|hrs|h)\b", normalized)
    if hour_match:
        try:
            return max(1, int(float(hour_match.group(1)) * 60))
        except ValueError:
            pass
    minute_match = re.search(r"\b(\d{1,4})\s*(?:minute|minutes|min|mins|m)\b", normalized)
    if minute_match:
        return max(1, _safe_int(minute_match.group(1), 60))
    return 0


def _make_tool_call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Create a LangChain-compatible tool call dict for deterministic queue fallback."""
    return {
        "name": name,
        "args": args,
        "id": f"call_planner_{uuid.uuid4().hex[:16]}",
        "type": "tool_call",
    }


def _infer_requested_tool_names(user_query: str) -> list[str]:
    """Heuristic fallback for obvious multi-capability requests in one user message."""
    normalized = _normalize_for_match(user_query)
    if not normalized:
        return []

    requested: list[str] = []

    def add(name: str) -> None:
        if name not in requested:
            requested.append(name)

    if re.search(r"\b(how long|duration|estimate|take|time needed|study time)\b", normalized):
        add("estimate_study_duration")
    if re.search(r"\b(free slot|free slots|available|availability|when can|find time|fit .*in|schedule)\b", normalized):
        add("find_free_slots")
    if re.search(r"\b(where|place|environment|location|room|space)\b", normalized) and re.search(
        r"\b(study|work|do|complete|revise|review|focus)\b", normalized
    ):
        add("recommend_study_environment")
    if re.search(r"\b(good idea|suitable|fit|appropriate|okay|ok)\b", normalized) and re.search(
        r"\b(in|at)\b", normalized
    ):
        if "recommend_study_environment" in requested:
            requested.remove("recommend_study_environment")
        add("assess_study_environment_fit")
    research_request = bool(
        re.search(
            r"\b(research-backed|evidence-based|learning science|study technique|spaced|retrieval practice|recall|habit|breaks|fatigue|prioriti[sz]|cognitive load)\b",
            normalized,
        )
        or (
            re.search(r"\b(research|evidence)\b", normalized)
            # Avoid false positives when the user mentions the "operations-research" course name.
            # `_normalize_for_match()` preserves hyphens, so match both "operations research" and "operations-research".
            and not re.search(r"\boperations[\s-]?research\b", normalized)
            and re.search(r"\b(advice|recommend|strategy|strategies|why|effective|better|guidance)\b", normalized)
        )
    )
    explicit_course_material_request = bool(
        re.search(
            r"\b(covered|lecture|lectures|slides|course material|course materials|summarize)\b",
            normalized,
        )
        or re.search(r"\b(course|week|lecture|slide|material|covered)\s+topics?\b", normalized)
        or re.search(r"\btopics?\s+(covered|in\s+(?:the\s+)?(?:course|week|lecture|slides?|materials?))\b", normalized)
    )
    if explicit_course_material_request:
        add("kb_course_retrieval")
    if research_request:
        add("kb_external_research_retrieval")

    priority = {
        "kb_course_retrieval": 10,
        "kb_external_research_retrieval": 20,
        "estimate_study_duration": 30,
        "find_free_slots": 40,
        "recommend_study_environment": 50,
        "assess_study_environment_fit": 50,
    }
    return sorted(requested, key=lambda name: priority.get(name, 999))


def _default_tool_call_for_name(name: str, user_query: str, messages: list[BaseMessage]) -> dict[str, Any]:
    """Build conservative arguments for a missing tool in a detected multi-tool query."""
    settings = get_settings()
    week = _detect_week_filter(user_query)
    day = _detect_day_filter(user_query)
    course = _detect_course_filter(user_query)

    if name == "kb_course_retrieval":
        args: dict[str, Any] = {"query": user_query, "top_k": max(1, settings.hybrid_k)}
        if course:
            args["course_filter"] = course
        if week:
            args["week_filter"] = week
        return _make_tool_call(name, args)
    if name == "kb_external_research_retrieval":
        args = {"query": user_query, "top_k": max(1, settings.hybrid_k)}
        paper_id = _c6_paper_id_to_force(user_query)
        if paper_id:
            args["paper_id"] = paper_id
        return _make_tool_call(name, args)
    if name == "estimate_study_duration":
        args = {"task_query": user_query, "query": user_query}
        if course:
            args["course_filter"] = course
        if week:
            args["week_filter"] = week
        if day:
            args["day_filter"] = day
        return _make_tool_call(name, args)
    if name == "find_free_slots":
        required = _latest_required_minutes(messages) or _minutes_requested_in_query(user_query) or 60
        args = {
            "week_filter": week,
            "day_filter": day,
            "required_minutes": required,
        }
        return _make_tool_call(name, args)
    if name == "assess_study_environment_fit":
        return _make_tool_call(name, {"task_query": user_query, "query": user_query, "environment_hint": user_query})
    return _make_tool_call(name, {"task_query": user_query, "query": user_query})


def _augment_tool_calls_for_multi_intent(
    tool_calls: list[dict[str, Any]],
    user_query: str,
    messages: list[BaseMessage],
) -> list[dict[str, Any]]:
    """Append deterministic missing tool calls when the user clearly asked for several capabilities."""
    inferred = _infer_requested_tool_names(user_query)
    if len(inferred) <= 1:
        return tool_calls
    existing = [str(call.get("name", "")).strip() for call in tool_calls if isinstance(call, dict)]
    augmented = list(tool_calls)
    for name in inferred:
        if name in existing:
            continue
        augmented.append(_default_tool_call_for_name(name, user_query, messages))
        existing.append(name)
    if len(augmented) > len(tool_calls):
        logging.getLogger(__name__).info(
            "sequential_tools augmented_missing=%s",
            [name for name in existing if name not in [str(c.get("name", "")).strip() for c in tool_calls]],
        )
    return augmented


def _normalize_deferred_tool_call_args(
    call: dict[str, Any],
    user_query: str,
    messages: list[BaseMessage],
    completed_tool_results: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Fill args whose values are only known after previous sequential tools have run."""
    name = str(call.get("name", "")).strip()
    args = dict(call.get("args") or {})
    if name == "find_free_slots":
        incoming_required = _safe_int(args.get("required_minutes", 0), 0)
        message_estimated_required = _latest_required_minutes(messages)
        completed_estimated_required = _latest_required_minutes_from_completed(completed_tool_results or [])
        estimated_required = completed_estimated_required or message_estimated_required
        query_required = _minutes_requested_in_query(user_query)
        if not str(args.get("week_filter", "")).strip():
            args["week_filter"] = _detect_week_filter(user_query)
        if not str(args.get("day_filter", "")).strip():
            detected_day = _detect_day_filter(user_query)
            if detected_day:
                args["day_filter"] = detected_day
        if estimated_required > 0:
            args["required_minutes"] = estimated_required
        elif incoming_required <= 0:
            args["required_minutes"] = query_required or 60
        logging.getLogger(__name__).info(
            "sequential_tools c3_duration_handoff incoming_required=%s "
            "message_estimate_required=%s completed_estimate_required=%s "
            "latest_estimate_required=%s query_required=%s final_required=%s "
            "week_filter=%r day_filter=%r",
            incoming_required,
            message_estimated_required,
            completed_estimated_required,
            estimated_required,
            query_required,
            args.get("required_minutes"),
            args.get("week_filter", ""),
            args.get("day_filter", ""),
        )
    if name in {"estimate_study_duration", "recommend_study_environment", "assess_study_environment_fit"}:
        if not str(args.get("task_query", "")).strip() and not str(args.get("query", "")).strip():
            args["task_query"] = user_query
            args["query"] = user_query
    if name in {"kb_course_retrieval", "kb_external_research_retrieval"} and not str(args.get("query", "")).strip():
        args["query"] = user_query
    return {**call, "args": args}


def agent_node(state: AgentState):
    """LLM node: routes tool calls and composes final C1–C6 responses."""
    messages = state["messages"]
    user_query = _active_user_query(state)
    routing_turn = not (messages and getattr(messages[-1], "type", None) == "tool")
    _trace_log("agent_node enter last_message_type=%s user_query=%r", getattr(messages[-1], "type", "unknown"), user_query)

    if messages and getattr(messages[-1], "type", None) == "tool":
        settings = get_settings()
        tm = messages[-1]
        tool_name = str(getattr(tm, "name", "")).strip()
        tool_result = str(tm.content)
        tool_result = _kb_tool_result_after_retry(tool_name, tool_result, user_query, settings)
        _trace_log(
            "agent_node tool_result name=%s preview=%r",
            tool_name,
            _truncate_for_log(tool_result, 180),
        )

        current_tool_records = _completed_tool_results(state)
        current_tool_messages = _tool_messages_since_latest_user(messages)
        if len(current_tool_records) > 1:
            tool_sections: list[str] = []
            for index, tool_record in enumerate(current_tool_records, start=1):
                name = str(tool_record.get("name", "unknown_tool")).strip()
                content = str(tool_record.get("content", ""))
                tool_sections.append(f"TOOL_{index} name={name}\n{content}")
            prompt_messages = [
                SystemMessage(
                    content=(
                        "Produce one concise answer that integrates every completed planner tool result.\n"
                        "Use a short section for each distinct user need in the same order the tools ran.\n"
                        "Do not ignore earlier tool outputs just because a later tool ran. "
                        "Ground claims only in the provided tool transcripts."
                    )
                ),
                HumanMessage(
                    content=(
                        f'User request: "{user_query}"\n\n'
                        "Completed tool transcripts:\n\n"
                        + "\n\n".join(tool_sections)
                    )
                ),
            ]
        elif len(current_tool_messages) > 1:
            tool_sections = []
            for index, tool_message in enumerate(current_tool_messages, start=1):
                name = str(getattr(tool_message, "name", "") or "unknown_tool").strip()
                content = str(tool_message.content)
                if tool_message is tm and tool_result != content:
                    content = tool_result
                tool_sections.append(f"TOOL_{index} name={name}\n{content}")
            prompt_messages = [
                SystemMessage(
                    content=(
                        "Produce one concise answer that integrates every completed planner tool result.\n"
                        "Use a short section for each distinct user need in the same order the tools ran.\n"
                        "Do not ignore earlier tool outputs just because a later tool ran. "
                        "Ground claims only in the provided tool transcripts."
                    )
                ),
                HumanMessage(
                    content=(
                        f'User request: "{user_query}"\n\n'
                        "Completed tool transcripts:\n\n"
                        + "\n\n".join(tool_sections)
                    )
                ),
            ]
        elif tool_name == "estimate_study_duration":
            c2_payload = (
                f'User request: "{user_query}"\n\n'
                "Duration estimate:\n"
                f"{tool_result}"
            )
            prompt_messages = [
                SystemMessage(
                    content=(
                        "Produce a concise C2 duration estimate answer.\n"
                        "Output format:\n"
                        "- Estimated effort: minutes and suggested blocking\n"
                        "- Confidence and rationale grounded in provided evidence\n"
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
            prompt_messages = [
                SystemMessage(
                    content=(
                        "Produce a concise C3 slot-finder answer.\n"
                        "Summarize top slot options in bullets and mention if a split is needed.\n"
                        "Anchor time only on dates/week tag from Free slots; avoid relative week labels unless the user said them."
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
            include_images = _ablation_uses_multimodal_understanding(settings.rag_mode)
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
                    "mllm_input_c4_c5 tool=%s mode=%s candidate_images=%d attached_images=%d",
                    tool_name,
                    settings.rag_mode.value,
                    len(image_paths),
                    attached_count,
                )
                c4_user_payload = multimodal_content
            else:
                c4_user_payload = c4_payload_text
            prompt_messages = [
                SystemMessage(
                    content=(
                        (
                            "Produce a concise C5 environment-fit answer.\n"
                            if targeted_mode
                            else "Produce a concise C4 environment recommendation answer.\n"
                        )
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
                            "If assessment_mode is recommendation, select exactly one environment from the evidence by "
                            "comparing the user's request to each option's type, location, and evidence (and images if present). "
                            "Do not always recommend the private study room (env_03); pick the environment that best matches "
                            "the task (e.g. collaboration vs solo focus, noise tolerance, need for quiet, outdoor/light breaks).\n"
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
            include_images = _ablation_uses_multimodal_understanding(settings.rag_mode)
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
            if tool_name == "kb_external_research_retrieval":
                retrieval_guide = (
                    "Summarize actionable, research-grounded guidance for planners and learners from the excerpts. "
                    "Stay close to retrieved wording; distinguish established findings vs tentative claims. "
                    "If overlaps or contradictions appear across snippets, say so briefly.\n"
                    "Relevance: Lead with bullets that directly answer the question; deprioritize or omit excerpts "
                    "from loosely related papers when they do not help.\n"
                    "Output format:\n"
                    "- 3–6 short bullets tuned to what the student asked\n"
                    "- When you add evidence brackets, use only tags that exactly copy a source_file= filename "
                    "from the retrieved lines above (same spelling and punctuation). Do not bracket any other "
                    "filename or invented short name; if none apply, omit bracket tags."
                )
            else:
                retrieval_guide = (
                    "Given retrieved snippets for one course/week, list the main studied topics only. "
                    "Do not mention unrelated details. If evidence is weak, say so.\n"
                    "Output format:\n"
                    "- 3-6 topic bullets\n"
                    "- When you add evidence brackets, use only tags that exactly copy a source_file= filename "
                    "from the retrieved lines above (same spelling and punctuation). Do not bracket any other "
                    "filename or invented short name; if none apply, omit bracket tags."
                )
            prompt_messages = [
                SystemMessage(content=retrieval_guide),
                HumanMessage(content=user_payload),
            ]
    else:
        prompt_messages = messages

    mem_text = _format_planner_memory_for_prompt(_get_planner_memory(state))
    if mem_text.strip():
        prompt_messages = [SystemMessage(content=mem_text)] + prompt_messages

    response = get_llm().invoke(prompt_messages)
    if getattr(response, "tool_calls", None):
        response = _aimessage_with_forced_c6_paper_id(response, user_query)
    tcalls = _tool_calls_dict_list(getattr(response, "tool_calls", None))
    if routing_turn:
        tcalls = _augment_tool_calls_for_multi_intent(tcalls, user_query, messages)
        if tcalls:
            response = AIMessage(content="", tool_calls=tcalls)
    if getattr(response, "tool_calls", None):
        tool_names = [str(call.get("name", "")).strip() for call in response.tool_calls if isinstance(call, dict)]
        logging.getLogger(__name__).info(
            "agent_loop_debug model_requested_tools count=%d names=%s",
            len(response.tool_calls),
            tool_names,
        )
    else:
        logging.getLogger(__name__).info("agent_loop_debug model_requested_tools count=0")
        trailing = (
            messages[-1]
            if messages and getattr(messages[-1], "type", None) == "tool"
            else None
        )
        if trailing is not None and str(getattr(trailing, "name", "")).strip() in (
            "kb_course_retrieval",
            "kb_external_research_retrieval",
        ) and len(_tool_messages_since_latest_user(messages)) <= 1:
            answer_text = _render_assistant_content(response.content)
            if not _bracket_file_citations_match_retrieval(answer_text, str(trailing.content)):
                response = AIMessage(
                    content=(
                        "I couldn't match every file citation in my reply to this retrieval: each bracketed "
                        "filename must exactly copy a source_file= value from the evidence block, with no other "
                        "filenames in brackets."
                    )
                )
    _trace_log("agent_node llm_response has_tool_calls=%s", bool(getattr(response, "tool_calls", None)))
    if getattr(response, "tool_calls", None):
        _trace_log("agent_node tool_calls=%s", getattr(response, "tool_calls", None))
    _log_missing_required_tool_args(getattr(response, "tool_calls", None))

    out: dict[str, Any] = {"messages": messages + [response]}
    if routing_turn:
        out["planner_memory"] = {**_get_planner_memory(state), "active_user_query": user_query}
        out["completed_tool_results"] = []
    tcalls = _tool_calls_dict_list(getattr(response, "tool_calls", None))
    if len(tcalls) > 1:
        first = tcalls[0]
        response = response.model_copy(update={"tool_calls": [first]})
        out["messages"] = messages + [response]
        out["pending_tool_calls"] = tcalls[1:]
        logging.getLogger(__name__).info(
            "sequential_tools first_hop=%s queued=%d",
            str(first.get("name", "")).strip(),
            len(tcalls) - 1,
        )
    else:
        out["pending_tool_calls"] = []
    return out


def deferred_tools_node(state: AgentState) -> dict[str, Any]:
    """Pop the next queued tool call and append a synthetic AIMessage so ToolNode runs one tool per hop."""
    pending = list(state.get("pending_tool_calls") or [])
    if not pending:
        return {}
    rest = pending[1:]
    head = copy.deepcopy(pending[0])
    if not isinstance(head, dict) or not str(head.get("name", "")).strip():
        logging.getLogger(__name__).warning("deferred_tools skipping invalid queue head")
        return {"pending_tool_calls": rest}
    user_query = _active_user_query(state)
    head = _normalize_deferred_tool_call_args(
        head,
        user_query,
        state["messages"],
        completed_tool_results=_completed_tool_results(state),
    )
    synthetic = AIMessage(content="", tool_calls=[head])
    logging.getLogger(__name__).info(
        "sequential_tools deferred name=%s remaining_in_queue=%d",
        str(head.get("name", "")).strip(),
        len(rest),
    )
    return {
        "messages": state["messages"] + [synthetic],
        "pending_tool_calls": rest,
    }


def verifier_node(state: AgentState):
    """
    Post-tool verifier node.

    If tool output is weak/malformed, return a clarifying AI message instead of
    continuing to final synthesis.
    """
    messages = state["messages"]
    if not messages or getattr(messages[-1], "type", None) != "tool":
        return {}

    tool_name = str(getattr(messages[-1], "name", "")).strip()
    tool_result = str(messages[-1].content)
    user_query = _active_user_query(state)
    # Run KB auto-retry here (before weak checks) so recovery is not skipped when
    # the transcript would otherwise fail weak_markers in _tool_output_is_weak.
    retried = _kb_tool_result_after_retry(
        tool_name, tool_result, user_query, get_settings()
    )
    if retried != tool_result:
        prev = messages[-1]
        messages = [
            *messages[:-1],
            ToolMessage(
                content=retried,
                tool_call_id=str(getattr(prev, "tool_call_id", "") or ""),
                name=str(getattr(prev, "name", "") or tool_name),
            ),
        ]
        tool_result = retried
        logging.getLogger(__name__).info(
            "verifier kb_retry applied tool=%s changed_chars=%d",
            tool_name,
            abs(len(retried) - len(str(prev.content))),
        )

    call_args = _tool_call_args_for_tool_message(messages, messages[-1])
    mem_patch = _build_planner_memory_patch(tool_name, tool_result, call_args)
    merged_memory = {**_get_planner_memory(state), **mem_patch}
    completed = _completed_tool_results(state) + [
        {"name": tool_name, "content": tool_result}
    ]

    is_weak, reason = _tool_output_is_weak(
        tool_name,
        tool_result,
        user_query=user_query,
        tool_call_args=call_args,
    )
    logging.getLogger(__name__).info(
        "agent_loop_debug verifier tool=%s weak=%s reason=%s",
        tool_name,
        is_weak,
        reason or "ok",
    )
    if not is_weak:
        logging.getLogger(__name__).info(
            "sequential_tools completed_tool_results count=%d names=%s latest_estimate_required=%s",
            len(completed),
            [item.get("name", "") for item in completed],
            _latest_required_minutes_from_completed(completed),
        )
        return {
            "messages": messages,
            "planner_memory": merged_memory,
            "completed_tool_results": completed,
        }

    question = _clarifying_question_for_tool(tool_name, reason)
    return {
        "messages": messages + [AIMessage(content=question)],
        "planner_memory": merged_memory,
        "pending_tool_calls": [],
        "completed_tool_results": completed,
    }


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


def route_after_verifier(state: AgentState):
    """If verifier added a clarifying AI message, end; else run more queued tools or return to the agent."""
    last = state["messages"][-1]
    if getattr(last, "type", None) == "ai":
        logging.getLogger(__name__).info("agent_loop_debug verifier_route=end")
        return END
    pending = list(state.get("pending_tool_calls") or [])
    if pending:
        logging.getLogger(__name__).info(
            "agent_loop_debug verifier_route=deferred_tools remaining=%d",
            len(pending),
        )
        return "deferred_tools"
    logging.getLogger(__name__).info("agent_loop_debug verifier_route=agent")
    return "agent"


tool_node = ToolNode(TOOLS)

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_node("verifier", verifier_node)
graph.add_node("deferred_tools", deferred_tools_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
graph.add_conditional_edges(
    "verifier",
    route_after_verifier,
    {"agent": "agent", "deferred_tools": "deferred_tools", END: END},
)
graph.add_edge("tools", "verifier")
graph.add_edge("deferred_tools", "tools")
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
    print("\nPlanner agent ready. Type 'exit' to quit.\n")
    session_memory: dict[str, Any] = {}
    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in {"exit", "quit"}:
            break
        if not user_text:
            # Empty line would create HumanMessage("") which LangChain drops; providers
            # may reject requests with no user content.
            continue

        result = app.invoke(
            {
                "messages": [
                    SYSTEM_PROMPT,
                    HumanMessage(content=user_text),
                ],
                "planner_memory": dict(session_memory),
                "pending_tool_calls": [],
                "completed_tool_results": [],
            }
        )
        merged = result.get("planner_memory")
        if isinstance(merged, dict):
            session_memory = dict(merged)
        last = result["messages"][-1]
        print(f"\nAgent: {_render_assistant_content(last.content)}\n")
