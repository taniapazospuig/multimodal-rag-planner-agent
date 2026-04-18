"""Load settings from environment (and optional `.env` via python-dotenv)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum


class LLMBackend(str, Enum):
    GEMINI = "gemini"
    OLLAMA = "ollama"


class RAGPipelineMode(str, Enum):
    """Ablation arms for your study (retrieval vs understanding modalities)."""

    TEXT_ONLY = "text_only"  # text index, text-only LLM
    TEXT_RETRIEVAL_MLLM = "text_retrieval_mllm"  # text index, MLLM at read time
    MULTIMODAL_RETRIEVAL_MLLM = "multimodal_retrieval_mllm"  # CLIP-aligned index + MLLM


@dataclass(frozen=True)
class Settings:
    llm_backend: LLMBackend
    gemini_model: str
    gemini_api_key: str | None
    ollama_base_url: str
    ollama_model: str
    rag_mode: RAGPipelineMode
    open_clip_model: str
    open_clip_pretrained: str


def load_settings() -> Settings:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    backend_raw = (os.environ.get("PLANNER_LLM_BACKEND") or "gemini").strip().lower()
    backend = LLMBackend.GEMINI if backend_raw != "ollama" else LLMBackend.OLLAMA

    mode_raw = (os.environ.get("PLANNER_RAG_MODE") or RAGPipelineMode.TEXT_RETRIEVAL_MLLM.value).strip().lower()
    try:
        rag_mode = RAGPipelineMode(mode_raw)
    except ValueError:
        rag_mode = RAGPipelineMode.TEXT_RETRIEVAL_MLLM

    gemini_key = (
        os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or ""
    ).strip() or None

    return Settings(
        llm_backend=backend,
        gemini_model=(os.environ.get("GEMINI_MODEL") or "gemini-2.0-flash").strip(),
        gemini_api_key=gemini_key,
        ollama_base_url=(os.environ.get("OLLAMA_BASE_URL") or "http://127.0.0.1:11434").rstrip("/"),
        ollama_model=(os.environ.get("OLLAMA_MODEL") or "llava-phi3:3.8b").strip(),
        rag_mode=rag_mode,
        open_clip_model=(os.environ.get("OPEN_CLIP_MODEL") or "ViT-B-32").strip(),
        open_clip_pretrained=(os.environ.get("OPEN_CLIP_PRETRAINED") or "laion2b_s34b_b79k").strip(),
    )
