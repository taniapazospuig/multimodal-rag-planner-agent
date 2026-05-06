"""
Personal multimodal planner agent.

Graph is node-first (router -> capability -> retrieval planner -> tool selector ->
optional tool execution -> decomposition -> evidence formatting -> synthesis ->
verification), with structured typed AgentState handoffs between nodes.
"""

from __future__ import annotations

import csv
import json
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, List, TypedDict
from uuid import uuid4

from urllib3.exceptions import NotOpenSSLWarning
from PIL import Image
import chromadb
import torch
import open_clip
from rank_bm25 import BM25Okapi

from langchain.tools import tool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from config import RAGPipelineMode, Settings, load_settings
from text_tokenization import LexicalTokenizerConfig, tokenize_for_bm25

warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="google")


# =========================
# Agent state
# =========================


class AgentState(TypedDict):
    messages: List[BaseMessage]
    query_text: str
    route_intent: str
    route_confidence: str
    capability: str
    capability_reason: str
    retrieval_plan: str
    retrieval_sources: list[str]
    external_theme_filter: str
    required_citations: bool
    selected_tools: list[dict[str, Any]]
    tool_sequence: str
    skip_reason: str
    tool_run_status: str
    raw_tool_evidence: list[str]
    response_tasks: str
    must_include: list[str]
    unresolved_gaps: list[str]
    evidence_block: str
    draft_response: str
    verification_report: dict[str, Any]
    verify_retry_count: int


def _default_agent_state(messages: list[BaseMessage]) -> AgentState:
    "Create a fully iniitalized AgentState dictionary with default values."
    return {
        "messages": messages,
        "query_text": "",
        "route_intent": "",
        "route_confidence": "low",
        "capability": "course_grounded_qa",
        "capability_reason": "",
        "retrieval_plan": "text_only",
        "retrieval_sources": [],
        "external_theme_filter": "",
        "required_citations": True,
        "selected_tools": [],
        "tool_sequence": "none",
        "skip_reason": "",
        "tool_run_status": "empty",
        "raw_tool_evidence": [],
        "response_tasks": "",
        "must_include": [],
        "unresolved_gaps": [],
        "evidence_block": "EVIDENCE_BLOCK_START\n- [none] No tool evidence available.\nEVIDENCE_BLOCK_END",
        "draft_response": "",
        "verification_report": {"status": "pass", "notes": ""},
        "verify_retry_count": 0,
    }


# =========================
# Courses (personal KB seed)
# =========================


def load_courses(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


_BASE_DIR = Path(__file__).resolve().parent
COURSES_CSV_PATH = _BASE_DIR / "data" / "kb" / "courses.csv"
COURSES: List[dict] = load_courses(COURSES_CSV_PATH)
EXTERNAL_PAPERS_CSV_PATH = _BASE_DIR / "data" / "kb" / "metadata" / "external_papers.csv"


def load_external_papers(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


EXTERNAL_PAPERS: List[dict] = load_external_papers(EXTERNAL_PAPERS_CSV_PATH)

_SETTINGS: Settings | None = None


def get_settings() -> Settings:
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = load_settings()
    return _SETTINGS


class OpenCLIPBackbone:
    """Reusable OpenCLIP model for text and image embedding."""

    def __init__(self, settings: Settings):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess, self.tokenizer = self._load(settings)

    def _load(self, settings: Settings):
        model, _, preprocess = open_clip.create_model_and_transforms(
            settings.open_clip_model,
            pretrained=settings.open_clip_pretrained,
        )
        model = model.to(self.device).eval()
        tokenizer = open_clip.get_tokenizer(settings.open_clip_model)
        return model, preprocess, tokenizer

    def encode_text(self, texts: list[str]) -> list[list[float]]:
        with torch.no_grad():
            tokens = self.tokenizer(texts).to(self.device)
            emb = self.model.encode_text(tokens).cpu().numpy().tolist()
        return emb


def _infer_slug_from_course_row(course: dict) -> str | None:
    """Best-effort slug aligned with chunk `course` metadata (see chunks_recursive.jsonl)."""
    for key in ("course", "slug", "id"):
        raw = str(course.get(key) or "").strip()
        if raw and re.match(r"^[a-z0-9]+(?:-[a-z0-9]+)*$", raw.lower()):
            return raw.lower()
    title = str(course.get("title", "")).lower()
    if "operations" in title and "research" in title:
        return "operations-research"
    if ("high" in title and "dim" in title) or "high-dimensional" in title.replace(" ", "-"):
        return "high-dimensional-data"
    if "generative" in title:
        return "generative-ai"
    return None


def _course_filter_candidates(query: str) -> list[tuple[str, int]]:
    """Return (slug, score) pairs; higher score = stronger evidence. Used to pick a clear winner."""
    low = query.lower()
    scores: dict[str, int] = defaultdict(int)

    def add(slug: str, weight: int) -> None:
        scores[slug] += weight

    # Strong: explicit chunk / path slugs (user paste, folder paths).
    for slug in ("operations-research", "high-dimensional-data", "generative-ai"):
        if slug in low:
            add(slug, 4)

    # Catalog: course codes and full titles from courses.csv (if present).
    for course in COURSES:
        slug = _infer_slug_from_course_row(course)
        if not slug:
            continue
        code = str(course.get("code", "")).lower().strip()
        if code and len(code) >= 3 and code in low:
            add(slug, 5)
        title = str(course.get("title", "")).lower().strip()
        if len(title) >= 5 and title in low:
            add(slug, 5)
        if title:
            # Short title tokens (e.g. "COMP2701" already handled; multi-word partials).
            parts = [p for p in re.split(r"[^\w]+", title) if len(p) >= 4]
            hits = sum(1 for p in parts if p in low)
            if hits >= 2:
                add(slug, 3)

    # Phrase / synonym heuristics (weight 2 so catalog/slug beats weak overlaps).
    heuristics: list[tuple[str, tuple[str, ...]]] = [
        (
            "operations-research",
            (
                "operations research",
                "linear programming",
                "integer programming",
                "simplex method",
                "inventory theory",
                "queueing theory",
                "queuing theory",
                "network flow",
                "transportation problem",
                "assignment problem",
            ),
        ),
        (
            "high-dimensional-data",
            (
                "high dimensional data",
                "high-dimensional data",
                "high dimensional statistics",
                "high dim data",
                "high-dim data",
                "curse of dimensionality",
                "dimensionality reduction",
                "manifold learning",
            ),
        ),
        (
            "generative-ai",
            (
                "generative ai",
                "gen ai",
                "genai",
                "diffusion model",
                "variational autoencoder",
                "vae ",
                " gan ",
                "large language model",
                "llm ",
                "rag system",
                "retrieval augmented",
            ),
        ),
    ]
    for slug, phrases in heuristics:
        for ph in phrases:
            if ph.strip() in low:
                add(slug, 2)
                break

    # Token shortcuts (narrow: require word boundary via padded string for short tokens).
    padded = f" {low} "
    if re.search(r"\bhdd\b", padded):
        add("high-dimensional-data", 2)
    if re.search(r"\bcomp\s*2701\b", padded):
        add("generative-ai", 4)

    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return ranked


def _detect_course_filter(query: str) -> str | None:
    ranked = _course_filter_candidates(query)
    if not ranked:
        return None
    best_slug, best_score = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0
    # Require a winner; avoid filtering on a one-point tie with another course.
    if best_score >= 2 and best_score > second:
        return best_slug
    return None


def _rrf_fuse(
    dense_ranked_ids: list[str],
    bm25_ranked_ids: list[str],
    k: int,
) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for rank, chunk_id in enumerate(dense_ranked_ids, start=1):
        scores[chunk_id] += 1.0 / (k + rank)
    for rank, chunk_id in enumerate(bm25_ranked_ids, start=1):
        scores[chunk_id] += 1.0 / (k + rank)
    return dict(scores)


def _rank_to_rrf_score(rank: int, k: int) -> float:
    return 1.0 / (k + rank)


def _normalize_image_similarity(distance: float) -> float:
    # Chroma returns cosine distance (lower is better); convert to [0, 1] similarity.
    return max(0.0, min(1.0, 1.0 - distance))


class HybridTextRetriever:
    """Text retriever with BM25 + OpenCLIP dense search, fused via RRF."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.backbone = get_openclip_backbone()

        chroma_path = _BASE_DIR / "chroma_db"
        self.client = chromadb.PersistentClient(path=str(chroma_path))
        self.collection = self.client.get_or_create_collection(
            name=settings.text_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        bm25_path = _BASE_DIR / settings.text_bm25_path
        self.bm25_rows: list[dict] = []
        self.by_id: dict[str, dict] = {}
        self.bm25 = None
        self.lexical_tokenizer_config = LexicalTokenizerConfig()
        if bm25_path.exists():
            with bm25_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    chunk_id = str(row.get("chunk_id", ""))
                    if not chunk_id:
                        continue
                    self.bm25_rows.append(row)
                    self.by_id[chunk_id] = row

            if self.bm25_rows:
                first_meta = self.bm25_rows[0].get("metadata", {})
                lexical_meta = first_meta.get("lexical_tokenizer") if isinstance(first_meta, dict) else None
                self.lexical_tokenizer_config = LexicalTokenizerConfig.from_dict(
                    lexical_meta if isinstance(lexical_meta, dict) else None,
                    fallback=LexicalTokenizerConfig(),
                )
                corpus_tokens = [row.get("tokens", []) for row in self.bm25_rows]
                self.bm25 = BM25Okapi(corpus_tokens)

    def _dense_search(self, query: str, where: dict | None) -> list[str]:
        q = self.backbone.encode_text([query])[0]
        results = self.collection.query(
            query_embeddings=[q],
            n_results=self.settings.dense_k,
            where=where,
        )
        return [str(x) for x in ((results.get("ids") or [[]])[0] or [])]

    def _bm25_search(self, query: str, course_filter: str | None) -> list[str]:
        if self.bm25 is None or not self.bm25_rows:
            return []
        tokens = tokenize_for_bm25(query, self.lexical_tokenizer_config)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(
            range(len(scores)),
            key=lambda idx: float(scores[idx]),
            reverse=True,
        )
        out: list[str] = []
        for idx in ranked:
            if len(out) >= self.settings.bm25_k:
                break
            row = self.bm25_rows[idx]
            meta = row.get("metadata", {})
            if course_filter and str(meta.get("course")) != course_filter:
                continue
            out.append(str(row.get("chunk_id")))
        return out

    def search(self, query: str, k: int | None = None) -> list[dict]:
        course_filter = _detect_course_filter(query)
        where = {"course": course_filter} if course_filter else None

        dense_ids = self._dense_search(query, where=where)
        bm25_ids = self._bm25_search(query, course_filter=course_filter)
        fused = _rrf_fuse(dense_ids, bm25_ids, k=self.settings.rrf_k)
        out_k = k or self.settings.hybrid_k
        ranked_ids = sorted(fused.keys(), key=lambda x: fused[x], reverse=True)[:out_k]

        out: list[dict] = []
        for chunk_id in ranked_ids:
            row = self.by_id.get(chunk_id)
            if row:
                out.append(
                    {
                        "chunk_id": chunk_id,
                        "score": fused[chunk_id],
                        "retrieval_score": fused[chunk_id],
                        "text": str(row.get("text", "")),
                        "metadata": row.get("metadata", {}),
                    }
                )
                continue

            # Fallback if BM25 corpus is missing some rows.
            result = self.collection.get(ids=[chunk_id], include=["documents", "metadatas"])
            docs = result.get("documents") or []
            metas = result.get("metadatas") or []
            text = docs[0] if docs else ""
            meta = metas[0] if metas else {}
            out.append(
                {
                    "chunk_id": chunk_id,
                    "score": fused[chunk_id],
                    "retrieval_score": fused[chunk_id],
                    "text": text,
                    "metadata": meta,
                }
            )
        return out


_OPENCLIP_BACKBONE: OpenCLIPBackbone | None = None
_TEXT_RETRIEVER: HybridTextRetriever | None = None


def get_openclip_backbone() -> OpenCLIPBackbone:
    global _OPENCLIP_BACKBONE
    if _OPENCLIP_BACKBONE is None:
        _OPENCLIP_BACKBONE = OpenCLIPBackbone(get_settings())
    return _OPENCLIP_BACKBONE


def get_text_retriever() -> HybridTextRetriever:
    global _TEXT_RETRIEVER
    if _TEXT_RETRIEVER is None:
        _TEXT_RETRIEVER = HybridTextRetriever(get_settings())
    return _TEXT_RETRIEVER


class TextReranker:
    """Second-stage text reranker using a local cross-encoder."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_name = settings.text_reranker_model
        self._cross_encoder = None
        self.available = False
        self._init_model()

    def _init_model(self) -> None:
        if not self.settings.text_reranker_enabled:
            return
        try:
            from sentence_transformers import CrossEncoder

            self._cross_encoder = CrossEncoder(self.model_name)
            self.available = True
        except Exception as e:
            print(f"[TextReranker] CrossEncoder unavailable ({e}).")
            self._cross_encoder = None
            self.available = False

    def rerank(self, query: str, hits: list[dict], top_k: int) -> list[dict]:
        if not hits:
            return []
        # If text reranking is disabled, return the top k hits with their retrieval scores
        if not self.settings.text_reranker_enabled:
            out = [dict(hit) for hit in hits[:top_k]]
            for rank, row in enumerate(out, start=1):
                row["text_rerank_rank"] = rank
                row["text_rerank_score"] = float(row.get("retrieval_score", row.get("score", 0.0)))
                row["score"] = row["text_rerank_score"]
            return out

        # If text reranking is enabled but CrossEncoder is not available, raise an error
        if not self.available or self._cross_encoder is None:
            raise RuntimeError(
                "Text reranker is enabled but CrossEncoder could not be loaded. "
                "Install dependencies and verify TEXT_RERANKER_MODEL."
            )

        out: list[dict] = []
        pairs = [(query, str(hit.get("text", ""))) for hit in hits] # Build (query, document text) pairs for each hit
        scores = self._cross_encoder.predict(pairs) # Get relevance scores for each pair
        for hit, score in zip(hits, scores, strict=False):
            row = dict(hit)
            row["text_rerank_score"] = float(score) # Write relevance score
            out.append(row)
        out.sort(key=lambda x: float(x.get("text_rerank_score", 0.0)), reverse=True) # Sort all hits descending by relevance score
        for rank, row in enumerate(out, start=1):
            row["text_rerank_rank"] = rank # Assign final rank to each hit
            row["score"] = float(row.get("text_rerank_score", 0.0)) # Set final score to relevance score
        return out[:top_k]


class VisualReranker:
    """Visual reranker based on BLIP image-text matching (ITM)."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_name = settings.visual_reranker_model
        self.device = self._select_device()
        self._processor = None
        self._model = None
        self.available = False
        self._init_model()

    @staticmethod
    def _select_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _init_model(self) -> None:
        if not self.settings.visual_rerank_enabled:
            return
        try:
            from transformers import BlipForImageTextRetrieval, BlipProcessor

            self._processor = BlipProcessor.from_pretrained(self.model_name)
            self._model = BlipForImageTextRetrieval.from_pretrained(self.model_name)
            self._model = self._model.to(self.device).eval()
            self.available = True
        except Exception as e:
            print(f"[VisualReranker] BLIP ITM unavailable ({e}).")
            self._processor = None
            self._model = None
            self.available = False

    def _load_candidate_image(self, hit: dict) -> Image.Image | None:
        """Load a candidate image from absolute path or KB-relative path."""
        raw_path = str(hit.get("path", "")).strip()
        if raw_path:
            image_path = Path(raw_path)
        else:
            rel = str(hit.get("relative_path", "")).strip()
            image_path = _BASE_DIR / rel if rel else None
        if not image_path or not image_path.exists():
            return None
        try:
            with Image.open(image_path) as opened:
                return opened.convert("RGB")
        except Exception:
            return None

    def _score_pair(self, query: str, image: Image.Image) -> float:
        """Return BLIP ITM relevance score in [0, 1]."""
        if self._processor is None or self._model is None:
            raise RuntimeError("Visual reranker model is not initialized.")
        inputs = self._processor(images=image, text=query, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs, use_itm_head=True)
        logits = getattr(outputs, "itm_score", None)
        if logits is None:
            logits = outputs[0]
        probs = torch.softmax(logits, dim=-1)
        return float(probs[0, 1].item())

    def rerank(self, query: str, hits: list[dict], top_k: int) -> list[dict]:
        """Rerank image candidates with BLIP ITM; keep score fields for logging."""
        if not hits:
            return []
        if not self.settings.visual_rerank_enabled:
            out = [dict(hit) for hit in hits[:top_k]]
            for rank, row in enumerate(out, start=1):
                row["visual_rerank_rank"] = rank
                row["visual_rerank_score"] = _normalize_image_similarity(float(row.get("distance", 1.0)))
            return out

        if not self.available:
            raise RuntimeError(
                "Visual reranker is enabled but BLIP ITM could not be loaded. "
                "Install dependencies and verify VISUAL_RERANKER_MODEL."
            )

        out: list[dict] = []
        for hit in hits:
            row = dict(hit)
            image = self._load_candidate_image(row)
            if image is None:
                row["visual_rerank_score"] = 0.0
            else:
                row["visual_rerank_score"] = self._score_pair(query, image)
            out.append(row)

        out.sort(key=lambda x: float(x.get("visual_rerank_score", 0.0)), reverse=True)
        for rank, row in enumerate(out, start=1):
            row["visual_rerank_rank"] = rank
        return out[:top_k]


def _fuse_multimodal_hits(
    text_hits: list[dict],
    image_hits: list[dict],
    text_alpha: float,
    rrf_k: int,
    top_k: int,
) -> list[dict[str, Any]]:
    fused: list[dict[str, Any]] = []

    for rank, hit in enumerate(text_hits, start=1):
        fused.append(
            {
                "kind": "text",
                "item": hit,
                "score": text_alpha * _rank_to_rrf_score(rank, rrf_k),
            }
        )
    for rank, hit in enumerate(image_hits, start=1):
        fused.append(
            {
                "kind": "image",
                "item": hit,
                "score": (1.0 - text_alpha) * _rank_to_rrf_score(rank, rrf_k),
            }
        )

    fused.sort(key=lambda x: float(x["score"]), reverse=True)
    return fused[:top_k]


_TEXT_RERANKER: TextReranker | None = None
_VISUAL_RERANKER: VisualReranker | None = None


def get_text_reranker() -> TextReranker:
    global _TEXT_RERANKER
    if _TEXT_RERANKER is None:
        _TEXT_RERANKER = TextReranker(get_settings())
    return _TEXT_RERANKER


def get_visual_reranker() -> VisualReranker:
    global _VISUAL_RERANKER
    if _VISUAL_RERANKER is None:
        _VISUAL_RERANKER = VisualReranker(get_settings())
    return _VISUAL_RERANKER


# =========================
# CLIP + Chroma (image modality)
# =========================


class ImageIndex:
    """OpenCLIP image vectors in Chroma for manual and rendered PDF images."""

    def __init__(
        self,
        settings: Settings,
        image_dir: str = "data/kb/images",
        rendered_manifest_path: str = "data/kb/01_processed/rendered/pdf_pages_manifest.jsonl",
    ):
        self.image_dir = _BASE_DIR / image_dir
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.rendered_manifest_path = _BASE_DIR / rendered_manifest_path

        chroma_path = _BASE_DIR / "chroma_db"
        self.client = chromadb.PersistentClient(path=str(chroma_path))
        self.collection = self.client.get_or_create_collection(
            name="planner_images",
            metadata={"hnsw:space": "cosine"},
        )

        backbone = get_openclip_backbone()
        self.device = backbone.device
        self.model = backbone.model
        self.preprocess = backbone.preprocess
        self.tokenizer = backbone.tokenizer

        self._index_new_images()

    @staticmethod
    def _manual_image_id(path: Path) -> str:
        # Keep ids stable across runs and robust to nested paths.
        normalized = path.as_posix().replace("/", "__")
        return f"img__{normalized}"

    @staticmethod
    def _pdf_render_id(row: dict[str, str]) -> str:
        doc_id = str(row.get("doc_id", "")).strip()
        unit_id = str(row.get("unit_id", "")).strip()
        if doc_id and unit_id:
            return f"pdf__{doc_id}__{unit_id}"
        return ""

    def _collect_manual_images(self, existing_ids: set[str]) -> tuple[list[Image.Image], list[str], list[dict]]:
        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
        files = [f for f in self.image_dir.rglob("*") if f.is_file() and f.suffix.lower() in valid_exts]

        images: list[Image.Image] = []
        ids: list[str] = []
        metas: list[dict] = []
        for path in files:
            image_id = self._manual_image_id(path.relative_to(_BASE_DIR))
            if image_id in existing_ids:
                continue
            try:
                with Image.open(path) as opened:
                    img = opened.convert("RGB")
            except Exception as e:
                print(f"Skipping {path}: {e}")
                continue
            images.append(img)
            ids.append(image_id)
            metas.append(
                {
                    "source_kind": "manual_image",
                    "filename": path.name,
                    "path": str(path),
                    "relative_path": str(path.relative_to(_BASE_DIR).as_posix()),
                }
            )
        return images, ids, metas

    def _collect_rendered_pdf_pages(self, existing_ids: set[str]) -> tuple[list[Image.Image], list[str], list[dict]]:
        if not self.rendered_manifest_path.exists():
            return [], [], []

        images: list[Image.Image] = []
        ids: list[str] = []
        metas: list[dict] = []
        with self.rendered_manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                image_id = self._pdf_render_id(row)
                if not image_id or image_id in existing_ids:
                    continue

                page_image_rel = str(row.get("page_image_path", "")).strip()
                if not page_image_rel:
                    continue
                page_image_abs = _BASE_DIR / page_image_rel
                if not page_image_abs.exists():
                    continue

                try:
                    with Image.open(page_image_abs) as opened:
                        img = opened.convert("RGB")
                except Exception as e:
                    print(f"Skipping {page_image_abs}: {e}")
                    continue

                images.append(img)
                ids.append(image_id)
                metas.append(
                    {
                        "source_kind": "pdf_render",
                        "filename": page_image_abs.name,
                        "path": str(page_image_abs),
                        "relative_path": page_image_rel,
                        "doc_id": str(row.get("doc_id", "")),
                        "unit_id": str(row.get("unit_id", "")),
                        "page_number": int(row.get("page_number", 0) or 0),
                        "course": str(row.get("course", "")),
                        "week": str(row.get("week", "")),
                        "resource_type": str(row.get("resource_type", "")),
                        "source_path": str(row.get("source_path", "")),
                    }
                )
        return images, ids, metas

    def _encode_and_add(self, images: list[Image.Image], ids: list[str], metas: list[dict]) -> None:
        if not images:
            return
        with torch.no_grad():
            batch = torch.cat(
                [self.preprocess(im).unsqueeze(0).to(self.device) for im in images],
                dim=0,
            )
            emb = self.model.encode_image(batch).cpu().numpy()
        self.collection.add(embeddings=emb.tolist(), ids=ids, metadatas=metas)

    def _index_new_images(self) -> None:
        existing_ids = set(self.collection.get()["ids"] or [])
        manual_images, manual_ids, manual_metas = self._collect_manual_images(existing_ids)
        existing_ids.update(manual_ids)
        pdf_images, pdf_ids, pdf_metas = self._collect_rendered_pdf_pages(existing_ids)

        self._encode_and_add(manual_images, manual_ids, manual_metas)
        self._encode_and_add(pdf_images, pdf_ids, pdf_metas)

    def search(self, query: str, k: int = 4) -> list[dict]:
        with torch.no_grad():
            tokens = self.tokenizer([query]).to(self.device)
            q = self.model.encode_text(tokens).cpu().numpy()

        results = self.collection.query(query_embeddings=[q[0]], n_results=k)
        out: list[dict] = []
        metas = (results.get("metadatas") or [[]])[0]
        ids = (results.get("ids") or [[]])[0]
        dists = (results.get("distances") or [[]])[0]
        for image_id, meta, dist in zip(ids, metas, dists, strict=False):
            meta = meta or {}
            source_kind = str(meta.get("source_kind", "unknown"))
            filename = str(meta.get("filename", image_id))
            out.append(
                {
                    "id": image_id,
                    "distance": float(dist),
                    "source_kind": source_kind,
                    "filename": filename,
                    "path": str(meta.get("path", "")),
                    "relative_path": str(meta.get("relative_path", "")),
                    "doc_id": str(meta.get("doc_id", "")),
                    "unit_id": str(meta.get("unit_id", "")),
                    "page_number": int(meta.get("page_number", 0) or 0),
                    "course": str(meta.get("course", "")),
                    "source_path": str(meta.get("source_path", "")),
                }
            )
        return out


_IMAGE_INDEX: ImageIndex | None = None


def get_image_index() -> ImageIndex:
    global _IMAGE_INDEX
    if _IMAGE_INDEX is None:
        _IMAGE_INDEX = ImageIndex(get_settings())
    return _IMAGE_INDEX


@tool
def kb_course_qa_retrieve(query: str, top_k: int = 6, course_filter: str = "") -> str:
    """Retrieve grounded OR/HDD/GenAI course evidence with citation metadata."""
    hits = get_text_retriever().search(query, k=max(1, top_k))
    if course_filter.strip():
        wanted = course_filter.strip().lower()
        hits = [h for h in hits if str((h.get("metadata") or {}).get("course", "")).lower() == wanted]

    if not hits:
        return "No matching KB chunks found for course QA."

    lines = ["COURSE_QA_EVIDENCE:"]
    for i, hit in enumerate(hits[:top_k], start=1):
        meta = hit.get("metadata", {})
        text = str(hit.get("text", "")).replace("\n", " ").strip()
        if len(text) > 220:
            text = text[:220].rstrip() + "..."
        lines.append(
            f"{i}. [{meta.get('course','?')}|{meta.get('doc_id','?')}] "
            f"week={meta.get('week','?')} source={meta.get('source_path','?')} | {text}"
        )
    return "\n".join(lines)


@tool
def kb_weekly_plan_context(course_filters: str = "", week_range: str = "", top_k: int = 8) -> str:
    """Fetch planning context with week, difficulty, cognitive load, and dependencies."""
    planner_query = (
        "weekly study plan schedule difficulty cognitive load dependencies "
        f"{course_filters} {week_range}"
    ).strip()
    hits = get_text_retriever().search(planner_query, k=max(1, top_k))

    wanted_courses = {x.strip().lower() for x in course_filters.split(",") if x.strip()}
    if wanted_courses:
        hits = [h for h in hits if str((h.get("metadata") or {}).get("course", "")).lower() in wanted_courses]

    if not hits:
        return "No weekly planning context found."

    lines = ["WEEKLY_PLAN_CONTEXT:"]
    for i, hit in enumerate(hits[:top_k], start=1):
        meta = hit.get("metadata", {})
        text = str(hit.get("text", "")).replace("\n", " ").strip()
        if len(text) > 180:
            text = text[:180].rstrip() + "..."
        lines.append(
            f"{i}. course={meta.get('course','?')} week={meta.get('week','?')} "
            f"difficulty={meta.get('difficulty','unknown')} cognitive_load={meta.get('cognitive_load','unknown')} "
            f"dependencies={meta.get('dependencies','none')} doc_id={meta.get('doc_id','?')} | {text}"
        )
    return "\n".join(lines)


@tool
def kb_multimodal_retrieve(query: str, top_k_text: int = 6, top_k_image: int = 4) -> str:
    """Retrieve fused text+image evidence for find-that-topic/diagram requests."""
    text_hits = get_text_retriever().search(query, k=max(1, top_k_text))
    lines = ["MULTIMODAL_EVIDENCE:", "TEXT_HITS:"]
    for i, hit in enumerate(text_hits[:top_k_text], start=1):
        meta = hit.get("metadata", {})
        text = str(hit.get("text", "")).replace("\n", " ").strip()
        if len(text) > 170:
            text = text[:170].rstrip() + "..."
        lines.append(
            f"{i}. [{meta.get('course','?')}|{meta.get('doc_id','?')}] "
            f"week={meta.get('week','?')} source={meta.get('source_path','?')} | {text}"
        )

    if get_settings().rag_mode == RAGPipelineMode.TEXT_ONLY:
        lines.append("IMAGE_HITS: disabled in text_only mode")
        return "\n".join(lines)

    image_hits = get_image_index().search(query, k=max(1, top_k_image))
    lines.append("IMAGE_HITS:")
    for i, hit in enumerate(image_hits[:top_k_image], start=1):
        lines.append(
            f"{i}. path={hit.get('relative_path') or hit.get('filename')} "
            f"course={hit.get('course','?')} doc_id={hit.get('doc_id','?')} page={hit.get('page_number',0)} "
            f"distance={float(hit.get('distance', 0.0)):.3f}"
        )
    return "\n".join(lines)


@tool
def kb_image_task_extract(image_refs: str, extraction_mode: str = "planner_todo") -> str:
    """Parse planner/todo screenshots and produce structured tasks with priority hints."""
    if get_settings().rag_mode == RAGPipelineMode.TEXT_ONLY:
        return "Image extraction disabled in text_only mode."

    query = image_refs.strip() or "planner todo list handwriting tasks"
    image_hits = get_image_index().search(query, k=4)
    if not image_hits:
        return "No matching images found for extraction."

    lines = [f"EXTRACTED_TASKS mode={extraction_mode}:"]
    priority_labels = ["high", "medium", "medium", "low"]
    for i, hit in enumerate(image_hits, start=1):
        source = hit.get("relative_path") or hit.get("filename")
        page = int(hit.get("page_number", 0) or 0)
        course = str(hit.get("course", "")).strip() or "unknown"
        priority = priority_labels[min(i - 1, len(priority_labels) - 1)]
        lines.append(
            f"{i}. task=Review item from {source} "
            f"priority={priority} course={course} page={page} citation=[{hit.get('doc_id','?')}]"
        )
    return "\n".join(lines)


def _parse_csv_list(raw: str) -> list[str]:
    return [x.strip() for x in (raw or "").split(",") if x.strip()]


@tool
def kb_external_paper_retrieve(
    query: str,
    theme_filter: str = "",
    paper_ids: str = "",
    top_k: int = 3,
) -> str:
    """Retrieve external papers filtered by theme_primary and/or explicit extp ids."""
    if not EXTERNAL_PAPERS:
        return "No external_papers.csv found."

    wanted_themes = {x.lower() for x in _parse_csv_list(theme_filter)}
    wanted_papers = {x.lower() for x in _parse_csv_list(paper_ids)}

    candidates = []
    for row in EXTERNAL_PAPERS:
        pid = str(row.get("paper_id", "")).strip().lower()
        theme_primary = str(row.get("theme_primary", "")).strip().lower()
        if wanted_themes and theme_primary not in wanted_themes:
            continue
        if wanted_papers and pid not in wanted_papers:
            continue
        haystack = " ".join(
            [
                str(row.get("title_short", "")),
                str(row.get("theme_primary", "")),
                str(row.get("theme_secondary", "")),
                str(row.get("notes", "")),
            ]
        ).lower()
        tokens = [t for t in re.split(r"[^\w]+", query.lower()) if len(t) >= 4]
        score = sum(token in haystack for token in tokens)
        if wanted_papers and pid in wanted_papers:
            score += 2
        candidates.append((score, row))

    if not candidates:
        return "No external papers matched filters."

    candidates.sort(key=lambda x: x[0], reverse=True)
    lines = ["EXTERNAL_PAPER_EVIDENCE:"]
    for i, (_, row) in enumerate(candidates[: max(1, top_k)], start=1):
        lines.append(
            f"{i}. [{row.get('paper_id','?')}] {row.get('title_short','?')} ({row.get('year','?')}) "
            f"theme={row.get('theme_primary','?')} secondary={row.get('theme_secondary','?')} | "
            f"{row.get('notes','')}"
        )
    return "\n".join(lines)


_BLOCKED_INTERVENTIONS = {
    "priority_reset": {
        "papers": ["extp_003"],
        "text": "Choose one high-impact task first and defer low-value busywork.",
    },
    "study_method_switch": {
        "papers": ["extp_004"],
        "text": "Switch to active recall + spaced repetition for 10-20 minutes.",
    },
    "recovery_protocol": {
        "papers": ["extp_008", "extp_009"],
        "text": "Take a short restorative break, then restart with a concrete next action.",
    },
}


@tool
def kb_blocked_intervention_lookup(block_type: str) -> str:
    """Return intervention template and default citations for blocked states."""
    key = block_type.strip().lower().replace(" ", "_")
    if key not in _BLOCKED_INTERVENTIONS:
        key = "priority_reset"
    spec = _BLOCKED_INTERVENTIONS[key]
    papers = ", ".join(spec["papers"])
    return (
        f"BLOCKED_INTERVENTION:\n"
        f"Detected block type: {key}\n"
        f"10-20 min intervention: {spec['text']}\n"
        f"Next concrete action: do one 15-minute focused attempt, then reassess.\n"
        f"Citation: {papers}"
    )


TOOLS = [
    kb_course_qa_retrieve,
    kb_weekly_plan_context,
    kb_multimodal_retrieve,
    kb_image_task_extract,
    kb_external_paper_retrieve,
    kb_blocked_intervention_lookup,
]

INTENT_TO_CAPABILITY = {
    "qa": "course_grounded_qa",
    "weekly_plan": "weekly_study_plan",
    "image_extract": "image_to_task_extraction",
    "multimodal_find": "multimodal_retrieval_assistant",
    "planning_coach": "research_grounded_planning_coach",
    "blocked": "blocked_strategy_recommender",
}
THEME_TO_PAPERS = {
    "task-prioritization": ["extp_001", "extp_002", "extp_003"],
    "study-techniques": ["extp_004", "extp_005"],
    "cognitive-load": ["extp_006"],
    "energy-focus": ["extp_007", "extp_008"],
    "recovery-strategies": ["extp_008", "extp_009"],
    "habit-formation": ["extp_010", "extp_011"],
}


def _latest_user_text(messages: list[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return str(m.content)
    return ""


def _latest_ai_message(messages: list[BaseMessage]) -> AIMessage | None:
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            return m
    return None


def _detect_route_intent(query: str) -> tuple[str, str]:
    low = query.lower()
    if re.search(r"\b(stuck|overwhelmed|procrastinating|tired|burned out|blocked)\b", low):
        return "blocked", "high"
    if re.search(r"\b(plan|schedule|week|weekly|timetable)\b", low):
        if re.search(r"\b(research-backed|evidence-based|why|paper)\b", low):
            return "planning_coach", "high"
        return "weekly_plan", "high"
    if re.search(r"\b(extract|parse)\b", low) and re.search(r"\b(image|screenshot|handwriting|todo|planner)\b", low):
        return "image_extract", "high"
    if re.search(r"\b(find|where|diagram|slide|page|figure)\b", low):
        return "multimodal_find", "medium"
    return "qa", "medium"


def _detect_block_type(query: str) -> str:
    low = query.lower()
    if re.search(r"\b(procrastinat|avoid|delay)\b", low):
        return "priority_reset"
    if re.search(r"\b(can't remember|forget|retention|memor)\b", low):
        return "study_method_switch"
    if re.search(r"\b(tired|fatigue|exhaust|drained)\b", low):
        return "recovery_protocol"
    return "priority_reset"


def _format_tool_sequence(tool_calls: list[dict[str, Any]]) -> str:
    if not tool_calls:
        return "none"
    return " -> ".join(str(call.get("name", "")) for call in tool_calls)


def _extract_tool_evidence(messages: list[BaseMessage]) -> list[str]:
    evidence: list[str] = []
    for m in messages:
        if getattr(m, "type", None) != "tool":
            continue
        text = str(getattr(m, "content", "")).strip().replace("\n", " ")
        if not text:
            continue
        if len(text) > 280:
            text = text[:280].rstrip() + "..."
        evidence.append(text)
    return evidence


def query_router_node(state: AgentState):
    query = _latest_user_text(state.get("messages", []))
    route_intent, confidence = _detect_route_intent(query)
    return {
        "query_text": query,
        "route_intent": route_intent,
        "route_confidence": confidence,
    }


def capability_router_node(state: AgentState):
    route_intent = str(state.get("route_intent", "qa"))
    capability = INTENT_TO_CAPABILITY.get(route_intent, "course_grounded_qa")
    reason = f"mapped from route intent '{route_intent}'"
    return {"capability": capability, "capability_reason": reason}


def retrieval_planner_node(state: AgentState):
    capability = str(state.get("capability", "course_grounded_qa"))
    rag_mode = get_settings().rag_mode.value

    plan = "text_plus_image" if rag_mode != RAGPipelineMode.TEXT_ONLY.value else "text_only"
    sources = ["kb_text"]
    external_theme = "none"
    required_citations = "yes"

    if capability == "multimodal_retrieval_assistant":
        plan = "text_only" if rag_mode == RAGPipelineMode.TEXT_ONLY.value else "text_plus_image"
        if plan == "text_plus_image":
            sources.append("kb_image")
    elif capability == "image_to_task_extraction":
        plan = "text_only" if rag_mode == RAGPipelineMode.TEXT_ONLY.value else "text_plus_image"
        if plan == "text_plus_image":
            sources = ["kb_image"]
    elif capability == "research_grounded_planning_coach":
        plan = "external_augmented"
        sources = ["external_papers", "kb_text"]
        external_theme = "task-prioritization,cognitive-load,energy-focus"
    elif capability == "blocked_strategy_recommender":
        plan = "external_augmented"
        sources = ["external_papers"]
        external_theme = "recovery-strategies,study-techniques,task-prioritization"

    return {
        "retrieval_plan": plan,
        "retrieval_sources": sources,
        "external_theme_filter": external_theme,
        "required_citations": required_citations == "yes",
    }


def tool_selector_node(state: AgentState):
    capability = str(state.get("capability", "course_grounded_qa"))
    query = str(state.get("query_text", "") or _latest_user_text(state.get("messages", [])))

    tool_calls: list[dict[str, Any]] = []

    def add_call(name: str, args: dict[str, Any]) -> None:
        tool_calls.append({"id": f"call_{uuid4().hex[:8]}", "name": name, "args": args})

    if capability == "course_grounded_qa":
        add_call("kb_course_qa_retrieve", {"query": query, "top_k": 6, "course_filter": ""})
    elif capability == "weekly_study_plan":
        add_call("kb_weekly_plan_context", {"course_filters": "", "week_range": "", "top_k": 8})
        add_call(
            "kb_external_paper_retrieve",
            {
                "query": query,
                "theme_filter": "task-prioritization,cognitive-load,energy-focus",
                "paper_ids": "",
                "top_k": 3,
            },
        )
    elif capability == "image_to_task_extraction":
        add_call("kb_image_task_extract", {"image_refs": query, "extraction_mode": "planner_todo"})
    elif capability == "multimodal_retrieval_assistant":
        add_call("kb_multimodal_retrieve", {"query": query, "top_k_text": 6, "top_k_image": 4})
    elif capability == "research_grounded_planning_coach":
        add_call(
            "kb_external_paper_retrieve",
            {
                "query": query,
                "theme_filter": str(state.get("external_theme_filter", "")),
                "paper_ids": "",
                "top_k": 3,
            },
        )
        add_call("kb_weekly_plan_context", {"course_filters": "", "week_range": "", "top_k": 6})
    elif capability == "blocked_strategy_recommender":
        block_type = _detect_block_type(query)
        add_call("kb_blocked_intervention_lookup", {"block_type": block_type})
        blocked_papers = ",".join(_BLOCKED_INTERVENTIONS.get(block_type, _BLOCKED_INTERVENTIONS["priority_reset"])["papers"])
        add_call(
            "kb_external_paper_retrieve",
            {
                "query": query,
                "theme_filter": "",
                "paper_ids": blocked_papers,
                "top_k": 2,
            },
        )

    if not tool_calls:
        return {
            "selected_tools": [],
            "tool_sequence": "none",
            "skip_reason": "No tool required",
        }

    ai = AIMessage(
        content=f"TOOL_SEQUENCE: {_format_tool_sequence(tool_calls)}",
        tool_calls=tool_calls,
    )
    return {
        "messages": state["messages"] + [ai],
        "selected_tools": tool_calls,
        "tool_sequence": _format_tool_sequence(tool_calls),
        "skip_reason": "",
    }


tool_executor_node = ToolNode(TOOLS)


def task_decomposer_node(state: AgentState):
    capability = str(state.get("capability", "course_grounded_qa"))
    raw_tool_evidence = _extract_tool_evidence(state.get("messages", []))
    tool_run_status = "success" if raw_tool_evidence else ("empty" if state.get("tool_sequence", "none") != "none" else "skipped")
    templates = {
        "course_grounded_qa": "1) answer directly 2) cite KB evidence",
        "weekly_study_plan": "1) build weekly blocks 2) balance difficulty/cognitive load 3) cite support",
        "image_to_task_extraction": "1) list extracted tasks 2) rank priorities 3) note assumptions",
        "multimodal_retrieval_assistant": "1) return best matches 2) explain why these 3) cite page/doc ids",
        "research_grounded_planning_coach": "1) recommendation 2) why it works 3) supporting papers",
        "blocked_strategy_recommender": "1) detect block 2) suggest 10-20 min intervention 3) next action + citation",
    }
    unresolved_gaps = []
    if state.get("tool_sequence", "none") != "none" and not raw_tool_evidence:
        unresolved_gaps.append("Tool execution returned no evidence.")

    return {
        "tool_run_status": tool_run_status,
        "raw_tool_evidence": raw_tool_evidence,
        "response_tasks": templates.get(capability, templates["course_grounded_qa"]),
        "must_include": ["citations"],
        "unresolved_gaps": unresolved_gaps,
    }


def evidence_formatter_node(state: AgentState):
    evidence_lines: list[str] = []
    for idx, content in enumerate(state.get("raw_tool_evidence", [])[:8], start=1):
        evidence_lines.append(f"- [tool_{idx}] {content}")

    if not evidence_lines:
        evidence_lines = ["- [none] No tool evidence available."]

    return {"evidence_block": "EVIDENCE_BLOCK_START\n" + "\n".join(evidence_lines) + "\nEVIDENCE_BLOCK_END"}


# =========================
# LLM factory
# =========================


def _build_base_llm(settings: Settings):
    if settings.llm_backend.value == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    if not settings.gemini_api_key:
        raise ValueError(
            "Gemini selected but no API key. Set GOOGLE_API_KEY or GEMINI_API_KEY "
            "(see `.env.example`)."
        )
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=0,
        api_key=settings.gemini_api_key,
        convert_system_message_to_human=True,
    )


def build_llm(settings: Settings):
    return _build_base_llm(settings).bind_tools(TOOLS)


def build_plain_llm(settings: Settings):
    return _build_base_llm(settings)


_LLM = None
_PLAIN_LLM = None


def get_llm():
    global _LLM
    if _LLM is None:
        _LLM = build_llm(get_settings())
    return _LLM


def get_plain_llm():
    global _PLAIN_LLM
    if _PLAIN_LLM is None:
        _PLAIN_LLM = build_plain_llm(get_settings())
    return _PLAIN_LLM


def answer_synthesizer_node(state: AgentState):
    capability = str(state.get("capability", "course_grounded_qa"))
    query = str(state.get("query_text", "") or _latest_user_text(state.get("messages", [])))
    evidence = str(state.get("evidence_block", "EVIDENCE_BLOCK_START\n- [none] no evidence\nEVIDENCE_BLOCK_END"))
    response_tasks = str(state.get("response_tasks", "answer + citations"))
    verification_report = state.get("verification_report", {}) or {}
    verify_status = str(verification_report.get("status", "pass"))
    verify_notes = str(verification_report.get("notes", ""))

    revise_instruction = ""
    if verify_status == "revise" and verify_notes:
        revise_instruction = f"\nRevision note from verifier: {verify_notes}\n"

    system_prompt = (
        "You are a personalized multimodal study-planner assistant.\n"
        "Write concise, actionable answers.\n"
        "Follow the capability-specific format exactly.\n"
        "Never invent citations.\n"
    )
    human_prompt = (
        f"Capability: {capability}\n"
        f"User query: {query}\n"
        f"Required response tasks: {response_tasks}\n"
        f"Must include: {', '.join(state.get('must_include', [])) or 'none'}\n"
        f"Unresolved gaps: {', '.join(state.get('unresolved_gaps', [])) or 'none'}\n"
        f"{evidence}\n"
        f"{revise_instruction}"
        "Return only the final user-facing response."
    )
    response = get_plain_llm().invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
    return {
        "messages": state["messages"] + [response],
        "draft_response": _render_assistant_content(response.content),
    }


def verification_node(state: AgentState):
    required_citations = bool(state.get("required_citations", True))
    draft = str(state.get("draft_response", "")).strip()
    if not draft:
        draft_msg = _latest_ai_message(state.get("messages", []))
        draft = _render_assistant_content(draft_msg.content if draft_msg else "").strip()
    retry_count = int(state.get("verify_retry_count", 0) or 0)

    has_citation = bool(
        re.search(
            r"(extp_\d{3}|doc_\d{3}|citation|citations|supporting paper)",
            draft.lower(),
        )
    )
    if not draft.strip():
        status = "abstain"
        notes = "Empty draft response."
    elif required_citations and not has_citation:
        if retry_count < 1:
            status = "revise"
            notes = "Missing explicit citations. Add citation identifiers."
        else:
            status = "abstain"
            notes = "Missing citations after retry."
    else:
        status = "pass"
        notes = "Format and citations look acceptable."

    out_messages = list(state["messages"])

    if status == "abstain":
        safe = AIMessage(
            content=(
                "I do not have enough grounded evidence to answer confidently. "
                "Please share a more specific question or relevant week/course context, and I will retry with citations."
            )
        )
        out_messages.append(safe)

    next_retry_count = retry_count + (1 if status == "revise" else 0)
    return {
        "messages": out_messages,
        "verification_report": {"status": status, "notes": notes},
        "verify_retry_count": next_retry_count,
    }


def route_after_tool_selector(state: AgentState):
    if str(state.get("tool_sequence", "none")) != "none":
        return "tool_executor"
    return "task_decomposer"


def route_after_verification(state: AgentState):
    report = state.get("verification_report", {}) or {}
    status = str(report.get("status", "pass"))
    if status == "revise":
        return "revise"
    if status == "abstain":
        return "abstain"
    return "pass"


graph = StateGraph(AgentState)
graph.add_node("query_router", query_router_node)
graph.add_node("capability_router", capability_router_node)
graph.add_node("retrieval_planner", retrieval_planner_node)
graph.add_node("tool_selector", tool_selector_node)
graph.add_node("tool_executor", tool_executor_node)
graph.add_node("task_decomposer", task_decomposer_node)
graph.add_node("evidence_formatter", evidence_formatter_node)
graph.add_node("answer_synthesizer", answer_synthesizer_node)
graph.add_node("verification", verification_node)
graph.set_entry_point("query_router")
graph.add_edge("query_router", "capability_router")
graph.add_edge("capability_router", "retrieval_planner")
graph.add_edge("retrieval_planner", "tool_selector")
graph.add_conditional_edges(
    "tool_selector",
    route_after_tool_selector,
    {"tool_executor": "tool_executor", "task_decomposer": "task_decomposer"},
)
graph.add_edge("tool_executor", "task_decomposer")
graph.add_edge("task_decomposer", "evidence_formatter")
graph.add_edge("evidence_formatter", "answer_synthesizer")
graph.add_edge("answer_synthesizer", "verification")
graph.add_conditional_edges(
    "verification",
    route_after_verification,
    {"revise": "answer_synthesizer", "pass": END, "abstain": END},
)

app = graph.compile()


def _render_assistant_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                t = item.get("text")
                if isinstance(t, str) and t.strip():
                    parts.append(t)
        if parts:
            return "\n".join(parts)
    return str(content)


if __name__ == "__main__":
    s = get_settings()
    print(
        f"\nPlanner agent ready | LLM={s.llm_backend.value} | RAG mode={s.rag_mode.value}\n"
        "Type 'exit' to quit.\n"
    )

    while True:
        user = input("You: ").strip()
        if user.lower() in {"exit", "quit"}:
            break

        result = app.invoke(
            _default_agent_state(
                [
                    HumanMessage(content=user),
                ]
            )
        )
        last_ai = _latest_ai_message(result["messages"])
        if last_ai is None:
            print("\nAgent: [No assistant response produced]\n")
            continue
        print(f"\nAgent: {_render_assistant_content(last_ai.content)}\n")
