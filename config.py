"""Simple, readable runtime settings for the planner agent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum


# Defaults if environment variables missing:
# - LLM: gemini-3.1-flash-lite
# - Embeddings: OpenCLIP ViT-B-32 + laion2b_s34b_b79k
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"
DEFAULT_OPENCLIP_MODEL = "ViT-B-32"
DEFAULT_OPENCLIP_PRETRAINED = "laion2b_s34b_b79k"


class LLMBackend(str, Enum):
    """Supported LLM backends."""
    GEMINI = "gemini"
    OLLAMA = "ollama"


class RAGPipelineMode(str, Enum):
    """Supported RAG pipeline modes."""
    TEXT_ONLY = "text_only"
    TEXT_RETRIEVAL_MLLM = "text_retrieval_mllm"
    MULTIMODAL_RETRIEVAL_MLLM = "multimodal_retrieval_mllm"


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the planner agent."""
    # LLM
    llm_backend: LLMBackend
    gemini_model: str
    gemini_api_key: str | None

    # Kept for compatibility with planner_agent's optional Ollama branch
    ollama_base_url: str
    ollama_model: str

    # Retrieval
    rag_mode: RAGPipelineMode
    open_clip_model: str
    open_clip_pretrained: str
    text_collection_name: str
    text_bm25_path: str
    text_context_mode: str
    dense_k: int
    bm25_k: int
    hybrid_k: int
    rrf_k: int
    text_reranker_enabled: bool
    text_reranker_model: str
    text_rerank_top_n: int
    visual_rerank_enabled: bool
    visual_reranker_model: str
    visual_rerank_top_n: int
    multimodal_fusion_alpha: float
    multimodal_fusion_k: int
    course_filter_enabled: bool
    # When False, dense/BM25/image Chroma queries run with no course/week where-clause
    # (small corpora often rank better from query similarity alone). See CourseRetriever.search.
    retrieval_metadata_filter_enabled: bool
    debug_trace_enabled: bool
    mllm_max_images: int
    mllm_max_image_edge: int


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


def load_settings() -> Settings:
    """Load settings from `.env` + environment with sane defaults."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    mode_raw = (os.environ.get("PLANNER_RAG_MODE") or RAGPipelineMode.TEXT_RETRIEVAL_MLLM.value).strip().lower()
    rag_mode = (
        RAGPipelineMode(mode_raw)
        if mode_raw in {m.value for m in RAGPipelineMode}
        else RAGPipelineMode.TEXT_RETRIEVAL_MLLM
    )

    text_context_mode = (os.environ.get("TEXT_CONTEXT_MODE") or "metadata").strip().lower()
    if text_context_mode not in {"none", "metadata"}:
        text_context_mode = "metadata"
    return Settings(
        llm_backend=LLMBackend.GEMINI,
        gemini_model=(os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL).strip(),
        gemini_api_key=(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or "").strip() or None,
        ollama_base_url=(os.environ.get("OLLAMA_BASE_URL") or "http://127.0.0.1:11434").rstrip("/"),
        ollama_model=(os.environ.get("OLLAMA_MODEL") or "llava-phi3:3.8b").strip(),
        rag_mode=rag_mode,
        open_clip_model=(os.environ.get("OPEN_CLIP_MODEL") or DEFAULT_OPENCLIP_MODEL).strip(),
        open_clip_pretrained=(os.environ.get("OPEN_CLIP_PRETRAINED") or DEFAULT_OPENCLIP_PRETRAINED).strip(),
        text_collection_name=(os.environ.get("TEXT_COLLECTION_NAME") or "text_chunks").strip(),
        text_bm25_path=(os.environ.get("TEXT_BM25_PATH") or "data/kb/02_index/bm25_corpus.jsonl").strip(),
        text_context_mode=text_context_mode,
        dense_k=_int_env("TEXT_DENSE_K", 12),
        bm25_k=_int_env("TEXT_BM25_K", 12),
        hybrid_k=_int_env("TEXT_HYBRID_K", 6),
        rrf_k=_int_env("TEXT_RRF_K", 60),
        text_reranker_enabled=_bool_env("TEXT_RERANKER_ENABLED", True),
        text_reranker_model=(
            os.environ.get("TEXT_RERANKER_MODEL") or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ).strip(),
        text_rerank_top_n=_int_env("TEXT_RERANK_TOP_N", 16),
        visual_rerank_enabled=_bool_env("VISUAL_RERANK_ENABLED", False),
        visual_reranker_model=(
            os.environ.get("VISUAL_RERANKER_MODEL") or "Salesforce/blip-itm-base-coco"
        ).strip(),
        visual_rerank_top_n=_int_env("VISUAL_RERANK_TOP_N", 8),
        multimodal_fusion_alpha=_float_env("MULTIMODAL_FUSION_ALPHA", 0.7, 0.0, 1.0),
        multimodal_fusion_k=_int_env("MULTIMODAL_FUSION_K", 8),
        course_filter_enabled=_bool_env("COURSE_FILTER_ENABLED", True),
        retrieval_metadata_filter_enabled=_bool_env("RETRIEVAL_METADATA_FILTER_ENABLED", True),
        debug_trace_enabled=_bool_env("DEBUG_TRACE_ENABLED", False),
        mllm_max_images=_int_env("MLLM_MAX_IMAGES", 4),
        mllm_max_image_edge=_int_env("MLLM_MAX_IMAGE_EDGE", 1280),
    )
