"""Central runtime settings for the planner agent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum


class RAGPipelineMode(str, Enum):
    """Supported RAG pipeline modes."""
    TEXT_ONLY = "text_only"
    TEXT_RETRIEVAL_MLLM = "text_retrieval_mllm"
    MULTIMODAL_RETRIEVAL_MLLM = "multimodal_retrieval_mllm"


class TextRetrievalStrategy(str, Enum):
    """Supported text retrieval strategies (``TEXT_RETRIEVAL_STRATEGY``)."""
    HYBRID = "hybrid"
    DENSE_ONLY = "dense_only"
    BM25_ONLY = "bm25_only"


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the planner agent."""
    gemini_model: str
    gemini_api_key: str | None

    # Retrieval
    rag_mode: RAGPipelineMode
    text_retrieval_strategy: TextRetrievalStrategy
    open_clip_model: str
    open_clip_pretrained: str
    text_collection_name: str
    text_bm25_path: str
    text_context_mode: str
    dense_k: int
    bm25_k: int
    hybrid_k: int
    rrf_k: int
    multimodal_fusion_alpha: float
    multimodal_fusion_k: int
    # When True: infer course/week from tool args + query (optional LLM course enum),
    # apply them as Chroma ``where`` + BM25 post-filter, and echo resolved course/week in
    # tool transcripts. When False: no query-side week/course inference; retrieval is not
    # narrowed by course/week metadata. See CourseRetriever.search.
    retrieval_metadata_filter_enabled: bool
    debug_trace_enabled: bool
    mllm_max_images: int
    mllm_max_image_edge: int


DEFAULTS = {
    "GEMINI_MODEL": "gemini-3.1-flash-lite",
    "PLANNER_RAG_MODE": RAGPipelineMode.TEXT_RETRIEVAL_MLLM.value,
    "TEXT_RETRIEVAL_STRATEGY": TextRetrievalStrategy.HYBRID.value,
    "OPEN_CLIP_MODEL": "ViT-B-32",
    "OPEN_CLIP_PRETRAINED": "laion2b_s34b_b79k",
    "TEXT_COLLECTION_NAME": "text_chunks",
    "TEXT_BM25_PATH": "data/kb/02_index/bm25_corpus.jsonl",
    "TEXT_CONTEXT_MODE": "metadata",
    "TEXT_DENSE_K": 12,
    "TEXT_BM25_K": 12,
    "TEXT_HYBRID_K": 6,
    "TEXT_RRF_K": 60,
    "MULTIMODAL_FUSION_ALPHA": 0.7,
    "MULTIMODAL_FUSION_K": 8,
    "RETRIEVAL_METADATA_FILTER_ENABLED": True,
    "DEBUG_TRACE_ENABLED": False,
    "MLLM_MAX_IMAGES": 4,
    "MLLM_MAX_IMAGE_EDGE": 1280,
}


def _str_env(name: str) -> str:
    return (os.environ.get(name) or str(DEFAULTS[name])).strip()


def _int_env(name: str, default: int) -> int:
    """Convert environment variable to int, with minimum 1."""
    try:
        return max(1, int((os.environ.get(name) or str(default)).strip()))
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


def _float_env(name: str, default: float, low: float, high: float) -> float:
    try:
        value = float((os.environ.get(name) or str(default)).strip())
    except ValueError:
        return default
    return max(low, min(high, value))


def _enum_env(name: str, enum_cls: type[Enum], default: str) -> Enum:
    raw = (os.environ.get(name) or default).strip().lower()
    valid = {item.value for item in enum_cls}  # type: ignore[attr-defined]
    if raw in valid:
        return enum_cls(raw)  # type: ignore[call-arg]
    return enum_cls(default)  # type: ignore[call-arg]


def load_settings() -> Settings:
    """Load settings from `.env` + environment with sane defaults."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    rag_mode = _enum_env(
        "PLANNER_RAG_MODE",
        RAGPipelineMode,
        DEFAULTS["PLANNER_RAG_MODE"],
    )
    text_retrieval_strategy = _enum_env(
        "TEXT_RETRIEVAL_STRATEGY",
        TextRetrievalStrategy,
        DEFAULTS["TEXT_RETRIEVAL_STRATEGY"],
    )
    text_context_mode = _str_env("TEXT_CONTEXT_MODE").lower()
    if text_context_mode not in {"none", "metadata"}:
        text_context_mode = str(DEFAULTS["TEXT_CONTEXT_MODE"])

    return Settings(
        gemini_model=_str_env("GEMINI_MODEL"),
        gemini_api_key=(os.environ.get("GOOGLE_API_KEY") or "").strip() or None,
        rag_mode=rag_mode,
        text_retrieval_strategy=text_retrieval_strategy,
        open_clip_model=_str_env("OPEN_CLIP_MODEL"),
        open_clip_pretrained=_str_env("OPEN_CLIP_PRETRAINED"),
        text_collection_name=_str_env("TEXT_COLLECTION_NAME"),
        text_bm25_path=_str_env("TEXT_BM25_PATH"),
        text_context_mode=text_context_mode,
        dense_k=_int_env("TEXT_DENSE_K", int(DEFAULTS["TEXT_DENSE_K"])),
        bm25_k=_int_env("TEXT_BM25_K", int(DEFAULTS["TEXT_BM25_K"])),
        hybrid_k=_int_env("TEXT_HYBRID_K", int(DEFAULTS["TEXT_HYBRID_K"])),
        rrf_k=_int_env("TEXT_RRF_K", int(DEFAULTS["TEXT_RRF_K"])),
        multimodal_fusion_alpha=_float_env(
            "MULTIMODAL_FUSION_ALPHA",
            float(DEFAULTS["MULTIMODAL_FUSION_ALPHA"]),
            0.0,
            1.0,
        ),
        multimodal_fusion_k=_int_env(
            "MULTIMODAL_FUSION_K",
            int(DEFAULTS["MULTIMODAL_FUSION_K"]),
        ),
        retrieval_metadata_filter_enabled=_bool_env(
            "RETRIEVAL_METADATA_FILTER_ENABLED",
            bool(DEFAULTS["RETRIEVAL_METADATA_FILTER_ENABLED"]),
        ),
        debug_trace_enabled=_bool_env(
            "DEBUG_TRACE_ENABLED",
            bool(DEFAULTS["DEBUG_TRACE_ENABLED"]),
        ),
        mllm_max_images=_int_env("MLLM_MAX_IMAGES", int(DEFAULTS["MLLM_MAX_IMAGES"])),
        mllm_max_image_edge=_int_env(
            "MLLM_MAX_IMAGE_EDGE",
            int(DEFAULTS["MLLM_MAX_IMAGE_EDGE"]),
        ),
    )
