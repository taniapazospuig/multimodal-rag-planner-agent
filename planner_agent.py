"""
Personal multimodal planner agent.

Graph is LLM-first (agent -> optional tool execution -> verification) with
structured typed AgentState handoffs between nodes.
"""

from __future__ import annotations

import csv
import base64
import io
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

_MMLM_PAYLOAD_PREFIX = "__MMLM_PAYLOAD__="


def _collect_existing_image_paths(payload: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for key in ("text_linked_images", "retrieved_image_hits"):
        rows = payload.get(key, [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            rel = str(row.get("relative_path", "")).strip()
            if not rel:
                continue
            abs_path = _BASE_DIR / rel
            if abs_path.exists():
                paths.append(str(abs_path))
    return paths


def _extract_tool_payloads(raw_outputs: list[str]) -> tuple[list[str], list[dict[str, Any]]]:
    cleaned_outputs: list[str] = []
    payloads: list[dict[str, Any]] = []
    for output in raw_outputs:
        cleaned_lines: list[str] = []
        for line in output.splitlines():
            if line.startswith(_MMLM_PAYLOAD_PREFIX):
                raw_payload = line[len(_MMLM_PAYLOAD_PREFIX):].strip()
                try:
                    payload = json.loads(raw_payload)
                    if isinstance(payload, dict):
                        payloads.append(payload)
                except json.JSONDecodeError:
                    cleaned_lines.append(line)
            else:
                cleaned_lines.append(line)
        cleaned_outputs.append("\n".join(cleaned_lines).strip())
    return cleaned_outputs, payloads


def _encode_image_as_data_url(path: str, max_edge: int) -> str | None:
    image_path = Path(path)
    if not image_path.exists():
        return None
    try:
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            width, height = image.size
            if max(width, height) > max_edge:
                scale = max_edge / float(max(width, height))
                resized = (
                    max(1, int(width * scale)),
                    max(1, int(height * scale)),
                )
                image = image.resize(resized, Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
    except Exception:
        return None
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


# =========================
# Agent state
# =========================


class AgentState(TypedDict):
    messages: List[BaseMessage]
    required_citations: bool
    selected_tools: list[dict[str, Any]]
    tool_sequence: str
    skip_reason: str
    draft_response: str
    verification_report: dict[str, Any]
    verify_retry_count: int


def _default_agent_state(messages: list[BaseMessage]) -> AgentState:
    "Create a fully initialized AgentState dictionary with default values."
    return {
        "messages": messages,
        "required_citations": True,
        "selected_tools": [],
        "tool_sequence": "none",
        "skip_reason": "",
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


def _normalize_for_match(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"[^\w\s-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _term_in_query(term: str, normalized_query: str) -> bool:
    if not term:
        return False
    # Short tokens should match as whole words (e.g. HDD, COMP2701).
    if len(term) <= 4 and " " not in term:
        return re.search(rf"\b{re.escape(term)}\b", normalized_query) is not None
    return term in normalized_query


def _course_filter_candidates(query: str) -> list[tuple[str, int]]:
    """Return (course, score) pairs from catalog matches."""
    normalized_query = _normalize_for_match(query)
    scores: dict[str, int] = defaultdict(int)

    for course in COURSES:
        slug = str(course.get("course", "")).strip().lower()
        if not slug:
            continue

        code = _normalize_for_match(str(course.get("code", "")))
        if code and _term_in_query(code, normalized_query):
            scores[slug] += 5

        title = _normalize_for_match(str(course.get("title", "")))
        if len(title) >= 5 and _term_in_query(title, normalized_query):
            scores[slug] += 5

        aliases_raw = str(course.get("aliases", "")).strip()
        if aliases_raw:
            for alias in aliases_raw.split("|"):
                alias_norm = _normalize_for_match(alias)
                if alias_norm and _term_in_query(alias_norm, normalized_query):
                    scores[slug] += 3
                    break

    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return ranked


def _detect_course_filter(
    query: str,
    enabled: bool = True,
) -> str | None:
    if not enabled:
        return None
    ranked = _course_filter_candidates(query)
    if not ranked:
        return None
    best_slug, best_score = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0
    # Apply only when confidence is strong enough to avoid over-filtering.
    if best_score >= 5 and (best_score - second) >= 2:
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
        course_filter = _detect_course_filter(
            query,
            enabled=self.settings.course_filter_enabled,
        )
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


def _append_mllm_payload(lines: list[str], payload: dict[str, Any]) -> None:
    lines.append(f"{_MMLM_PAYLOAD_PREFIX}{json.dumps(payload, ensure_ascii=False)}")


def _reranked_text_hits(
    query: str,
    final_k: int,
    course_filter: str = "",
) -> list[dict[str, Any]]:
    settings = get_settings()
    candidate_k = max(1, final_k, settings.text_rerank_top_n)
    hits = get_text_retriever().search(query, k=candidate_k)
    if course_filter.strip():
        wanted = course_filter.strip().lower()
        hits = [h for h in hits if str((h.get("metadata") or {}).get("course", "")).lower() == wanted]
    return get_text_reranker().rerank(query=query, hits=hits, top_k=max(1, final_k))


@tool
def kb_course_qa_retrieve(query: str, top_k: int = 6, course_filter: str = "") -> str:
    """Retrieve grounded OR/HDD/GenAI course evidence with citation metadata."""
    hits = _reranked_text_hits(query=query, final_k=max(1, top_k), course_filter=course_filter)

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
    settings = get_settings()
    planner_query = (
        "weekly study plan schedule difficulty cognitive load dependencies "
        f"{course_filters} {week_range}"
    ).strip()
    candidate_k = max(1, top_k, settings.text_rerank_top_n)
    stage1_hits = get_text_retriever().search(planner_query, k=candidate_k)

    wanted_courses = {x.strip().lower() for x in course_filters.split(",") if x.strip()}
    if wanted_courses:
        stage1_hits = [h for h in stage1_hits if str((h.get("metadata") or {}).get("course", "")).lower() in wanted_courses]
    hits = get_text_reranker().rerank(query=planner_query, hits=stage1_hits, top_k=max(1, top_k))

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
    """Retrieve reranked text/image evidence and weighted-RRF multimodal fusion."""
    settings = get_settings()
    text_candidate_k = max(1, top_k_text, settings.text_rerank_top_n)
    text_stage1_hits = get_text_retriever().search(query, k=text_candidate_k)
    text_hits = get_text_reranker().rerank(query=query, hits=text_stage1_hits, top_k=text_candidate_k)

    lines = ["MULTIMODAL_EVIDENCE:"]
    text_linked_images: list[dict[str, Any]] = []
    seen_text_image_paths: set[str] = set()
    for hit in text_hits:
        meta = hit.get("metadata", {}) or {}
        rel = str(meta.get("page_image_path", "")).strip()
        if not rel or rel in seen_text_image_paths:
            continue
        seen_text_image_paths.add(rel)
        text_linked_images.append(
            {
                "relative_path": rel,
                "doc_id": str(meta.get("doc_id", "")),
                "unit_id": str(meta.get("unit_id", "")),
                "course": str(meta.get("course", "")),
                "week": str(meta.get("week", "")),
            }
        )

    if settings.rag_mode == RAGPipelineMode.TEXT_ONLY:
        lines.append("MODE: text_only (image retrieval disabled)")
        lines.append("TEXT_HITS:")
        for i, hit in enumerate(text_hits[: max(1, top_k_text)], start=1):
            meta = hit.get("metadata", {})
            text = str(hit.get("text", "")).replace("\n", " ").strip()
            if len(text) > 170:
                text = text[:170].rstrip() + "..."
            lines.append(
                f"{i}. [{meta.get('course','?')}|{meta.get('doc_id','?')}] "
                f"week={meta.get('week','?')} source={meta.get('source_path','?')} "
                f"text_rerank_score={float(hit.get('text_rerank_score', hit.get('score', 0.0))):.4f} | {text}"
            )
        _append_mllm_payload(
            lines,
            {
                "tool": "kb_multimodal_retrieve",
                "mode": settings.rag_mode.value,
                "retrieval_sources": ["text"],
                "text_linked_images": [],
                "retrieved_image_hits": [],
            },
        )
        return "\n".join(lines)

    if settings.rag_mode == RAGPipelineMode.TEXT_RETRIEVAL_MLLM:
        lines.append("MODE: text_retrieval_mllm (image retrieval disabled; MLLM can inspect text-linked images)")
        lines.append("TEXT_HITS:")
        for i, hit in enumerate(text_hits[: max(1, top_k_text)], start=1):
            meta = hit.get("metadata", {})
            text = str(hit.get("text", "")).replace("\n", " ").strip()
            if len(text) > 170:
                text = text[:170].rstrip() + "..."
            lines.append(
                f"{i}. [{meta.get('course','?')}|{meta.get('doc_id','?')}] "
                f"week={meta.get('week','?')} source={meta.get('source_path','?')} "
                f"text_rerank_score={float(hit.get('text_rerank_score', hit.get('score', 0.0))):.4f} | {text}"
            )
        if text_linked_images:
            lines.append("TEXT_LINKED_IMAGES:")
            for i, image in enumerate(text_linked_images[: max(1, top_k_image)], start=1):
                lines.append(
                    f"{i}. path={image.get('relative_path')} "
                    f"doc_id={image.get('doc_id','?')} unit_id={image.get('unit_id','?')} "
                    f"course={image.get('course','?')} week={image.get('week','?')}"
                )
        _append_mllm_payload(
            lines,
            {
                "tool": "kb_multimodal_retrieve",
                "mode": settings.rag_mode.value,
                "retrieval_sources": ["text"],
                "text_linked_images": text_linked_images[: max(1, top_k_image)],
                "retrieved_image_hits": [],
            },
        )
        return "\n".join(lines)

    image_candidate_k = max(1, top_k_image, settings.visual_rerank_top_n)
    image_stage1_hits = get_image_index().search(query, k=image_candidate_k)
    image_hits = get_visual_reranker().rerank(query=query, hits=image_stage1_hits, top_k=image_candidate_k)

    fused_hits = _fuse_multimodal_hits(
        text_hits=text_hits,
        image_hits=image_hits,
        text_alpha=settings.multimodal_fusion_alpha,
        rrf_k=settings.rrf_k,
        top_k=max(1, settings.multimodal_fusion_k),
    )

    lines.append(
        "FUSED_HITS: "
        f"text_alpha={settings.multimodal_fusion_alpha:.2f} "
        f"image_alpha={(1.0 - settings.multimodal_fusion_alpha):.2f} "
        f"rrf_k={settings.rrf_k}"
    )
    for i, fused in enumerate(fused_hits, start=1):
        kind = str(fused.get("kind", "?"))
        item = fused.get("item", {}) or {}
        if kind == "text":
            meta = item.get("metadata", {})
            lines.append(
                f"{i}. kind=text fused_score={float(fused.get('score', 0.0)):.5f} "
                f"course={meta.get('course','?')} doc_id={meta.get('doc_id','?')} week={meta.get('week','?')}"
            )
        else:
            lines.append(
                f"{i}. kind=image fused_score={float(fused.get('score', 0.0)):.5f} "
                f"path={item.get('relative_path') or item.get('filename')} "
                f"doc_id={item.get('doc_id','?')} page={item.get('page_number',0)}"
            )

    lines.append("TEXT_HITS:")
    for i, hit in enumerate(text_hits[: max(1, top_k_text)], start=1):
        meta = hit.get("metadata", {})
        text = str(hit.get("text", "")).replace("\n", " ").strip()
        if len(text) > 170:
            text = text[:170].rstrip() + "..."
        lines.append(
            f"{i}. [{meta.get('course','?')}|{meta.get('doc_id','?')}] "
            f"week={meta.get('week','?')} source={meta.get('source_path','?')} "
            f"text_rerank_score={float(hit.get('text_rerank_score', hit.get('score', 0.0))):.4f} | {text}"
        )

    lines.append("IMAGE_HITS:")
    for i, hit in enumerate(image_hits[: max(1, top_k_image)], start=1):
        lines.append(
            f"{i}. path={hit.get('relative_path') or hit.get('filename')} "
            f"course={hit.get('course','?')} doc_id={hit.get('doc_id','?')} page={hit.get('page_number',0)} "
            f"distance={float(hit.get('distance', 0.0)):.3f} "
            f"visual_rerank_score={float(hit.get('visual_rerank_score', 0.0)):.4f}"
        )
    retrieved_image_hits = []
    for hit in image_hits[: max(1, top_k_image)]:
        rel = str(hit.get("relative_path", "")).strip()
        if not rel:
            continue
        retrieved_image_hits.append(
            {
                "relative_path": rel,
                "doc_id": str(hit.get("doc_id", "")),
                "page_number": int(hit.get("page_number", 0) or 0),
                "course": str(hit.get("course", "")),
            }
        )
    _append_mllm_payload(
        lines,
        {
            "tool": "kb_multimodal_retrieve",
            "mode": settings.rag_mode.value,
            "retrieval_sources": ["text", "image", "fused"],
            "text_linked_images": text_linked_images[: max(1, top_k_image)],
            "retrieved_image_hits": retrieved_image_hits,
        },
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
    _append_mllm_payload(
        lines,
        {
            "tool": "kb_image_task_extract",
            "mode": get_settings().rag_mode.value,
            "retrieval_sources": ["image"],
            "text_linked_images": [],
            "retrieved_image_hits": [
                {
                    "relative_path": str(hit.get("relative_path", "")).strip(),
                    "doc_id": str(hit.get("doc_id", "")),
                    "page_number": int(hit.get("page_number", 0) or 0),
                    "course": str(hit.get("course", "")),
                }
                for hit in image_hits
                if str(hit.get("relative_path", "")).strip()
            ],
        },
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

def _latest_user_text(messages: list[BaseMessage]) -> str:
    """Walk backwards through messages to find the most recent user message (query)."""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return str(m.content)
    return ""


def _latest_ai_message(messages: list[BaseMessage]) -> AIMessage | None:
    """Walk backwards through messages to find the most recent AI message (answer)."""
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            return m
    return None


def _format_tool_sequence(tool_calls: list[dict[str, Any]]) -> str:
    if not tool_calls:
        return "none"
    return " -> ".join(str(call.get("name", "")) for call in tool_calls)


def _normalize_tool_calls(raw_calls: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Normalize tool call dicts so ToolNode can execute them reliably."""
    out: list[dict[str, Any]] = []
    for call in raw_calls or []:
        name = str(call.get("name", "")).strip()
        if not name:
            continue

        raw_args = call.get("args", {})
        args: dict[str, Any]
        if isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args)
                args = parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                args = {}
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            args = {}

        out.append(
            {
                "id": str(call.get("id") or f"call_{uuid4().hex[:8]}"),
                "name": name,
                "args": args,
            }
        )
    return out


def _build_agent_system_prompt(state: AgentState) -> str:
    verification_report = state.get("verification_report", {}) or {}
    verify_status = str(verification_report.get("status", "pass"))
    verify_notes = str(verification_report.get("notes", ""))

    revision_note = ""
    if verify_status == "revise" and verify_notes:
        revision_note = f"\nVerifier note: {verify_notes}\nRevise your previous answer and include explicit citations.\n"

    return (
        "You are a personalized multimodal study-planner assistant with tool access.\n"
        "\n"
        "PRIMARY RULE\n"
        "- Prefer grounded answers over fast answers.\n"
        "- If retrieval would improve correctness, call tools first.\n"
        "- Never invent citations or evidence.\n"
        "\n"
        "OUTPUT CONTRACT\n"
        "- Final answers should be concise, practical, and evidence-grounded.\n"
        "- Include citation identifiers from tool outputs when evidence is used (e.g., doc_###, extp_###).\n"
        "- If evidence is insufficient, say so clearly and ask for the minimum missing detail.\n"
        "\n"
        "TOOL POLICY\n"
        "- Use kb_course_qa_retrieve for course concepts, requirements, comparisons, and factual what/why/how questions.\n"
        "- Use kb_weekly_plan_context for scheduling, weekly plans, workload balancing, sequencing, and timetables.\n"
        "- Use kb_multimodal_retrieve for find/where requests involving slides, figures, pages, diagrams, or image-backed evidence.\n"
        "- Use kb_image_task_extract for extracting todos/tasks from screenshots, planner images, handwritten notes, or photos.\n"
        "- Use kb_external_paper_retrieve for research-backed/evidence-based recommendations, rationale, and paper support.\n"
        "- Use kb_blocked_intervention_lookup when the user is blocked, overwhelmed, procrastinating, tired, burned out, or asks for next-step recovery.\n"
        "\n"
        "DEFAULT CALL SHAPES\n"
        "- kb_course_qa_retrieve(query=<user_query>, top_k=6, course_filter=\"\")\n"
        "- kb_weekly_plan_context(course_filters=\"\", week_range=\"\", top_k=8)\n"
        "- kb_multimodal_retrieve(query=<user_query>, top_k_text=6, top_k_image=4)\n"
        "- kb_image_task_extract(image_refs=<user_query>, extraction_mode=\"planner_todo\")\n"
        "- kb_external_paper_retrieve(query=<user_query>, theme_filter=\"\", paper_ids=\"\", top_k=3)\n"
        "- kb_blocked_intervention_lookup(block_type=<priority_reset|study_method_switch|recovery_protocol>)\n"
        "\n"
        "COMPOSITION RECIPES\n"
        "- Planning plus rationale: call kb_weekly_plan_context, then kb_external_paper_retrieve, then synthesize one integrated plan.\n"
        "- Multimodal explanation: call kb_multimodal_retrieve and optionally kb_course_qa_retrieve for clarification before synthesizing.\n"
        "- Blocked support with evidence: call kb_blocked_intervention_lookup and optionally kb_external_paper_retrieve for stronger rationale.\n"
        "\n"
        "AMBIGUITY / NON-KB REQUESTS\n"
        "- If the request is ambiguous and tool choice depends on missing info, ask one short clarification question.\n"
        "- For simple chit-chat or non-KB requests that do not require factual grounding, answer directly without unnecessary tool calls.\n"
        "\n"
        "QUALITY GUARDRAILS\n"
        "- Do not call irrelevant tools.\n"
        "- Do not repeat near-identical tool calls unless broadening search after empty evidence.\n"
        "- After tool results are available, provide the final answer from those results and do not call additional tools in that same turn.\n"
        f"{revision_note}"
    )


def _collect_trailing_tool_outputs(messages: list[BaseMessage]) -> list[str]:
    tool_outputs: list[str] = []
    for message in reversed(messages):
        if getattr(message, "type", None) != "tool":
            break
        tool_outputs.append(str(getattr(message, "content", "")))
    tool_outputs.reverse()
    return tool_outputs


def _build_post_tool_text_prompt(user_query: str, tool_results_text: list[str]) -> str:
    return (
        f'Original user query: "{user_query}"\n'
        "Tool results:\n"
        + "\n\n".join(tool_results_text)
    )


def _build_multimodal_human_content(
    user_query: str,
    tool_results_text: list[str],
    payloads: list[dict[str, Any]],
    settings: Settings,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rag_mode = settings.rag_mode
    retrieval_sources: set[str] = set()
    candidate_paths = _select_candidate_image_paths(payloads=payloads, rag_mode=rag_mode)
    for payload in payloads:
        sources = payload.get("retrieval_sources", [])
        if isinstance(sources, list):
            retrieval_sources.update(str(s) for s in sources if str(s).strip())

    unique_paths: list[str] = []
    seen_paths: set[str] = set()
    for path in candidate_paths:
        if path in seen_paths:
            continue
        seen_paths.add(path)
        unique_paths.append(path)

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": _build_post_tool_text_prompt(user_query, tool_results_text),
        }
    ]
    attached = 0
    failed_paths: list[str] = []
    max_images = max(1, settings.mllm_max_images)
    max_edge = max(256, settings.mllm_max_image_edge)
    for image_path in unique_paths[:max_images]:
        data_url = _encode_image_as_data_url(image_path, max_edge=max_edge)
        if not data_url:
            failed_paths.append(image_path)
            continue
        content.append(
            {
                "type": "image_url",
                "image_url": data_url,
            }
        )
        attached += 1

    trace = {
        "mode": rag_mode.value,
        "retrieval_sources": sorted(retrieval_sources),
        "attached_images": attached,
        "failed_images": len(failed_paths),
    }
    if failed_paths:
        content[0]["text"] += "\n\nImage attachment warnings:\n" + "\n".join(
            f"- failed_to_load: {Path(path).as_posix()}" for path in failed_paths
        )
    return content, trace


def _select_candidate_image_paths(
    payloads: list[dict[str, Any]],
    rag_mode: RAGPipelineMode,
) -> list[str]:
    candidate_paths: list[str] = []
    for payload in payloads:
        if rag_mode == RAGPipelineMode.TEXT_RETRIEVAL_MLLM:
            rows = payload.get("text_linked_images", [])
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, dict):
                        rel = str(row.get("relative_path", "")).strip()
                        if rel:
                            abs_path = _BASE_DIR / rel
                            if abs_path.exists():
                                candidate_paths.append(str(abs_path))
        elif rag_mode == RAGPipelineMode.MULTIMODAL_RETRIEVAL_MLLM:
            candidate_paths.extend(_collect_existing_image_paths(payload))
    unique_paths: list[str] = []
    seen_paths: set[str] = set()
    for path in candidate_paths:
        if path in seen_paths:
            continue
        seen_paths.add(path)
        unique_paths.append(path)
    return unique_paths


def run_ablation_mode_sanity_checks() -> dict[str, Any]:
    """Lightweight checks for image selection behavior across ablation modes."""
    sample_payload = {
        "retrieval_sources": ["text", "image", "fused"],
        "text_linked_images": [
            {"relative_path": "data/kb/01_processed/rendered/pdf_pages/doc_001/page_0001.png"}
        ],
        "retrieved_image_hits": [
            {"relative_path": "data/kb/01_processed/rendered/pdf_pages/doc_001/page_0002.png"}
        ],
    }
    payloads = [sample_payload]
    out = {
        "text_only_count": len(_select_candidate_image_paths(payloads, RAGPipelineMode.TEXT_ONLY)),
        "text_retrieval_mllm_count": len(
            _select_candidate_image_paths(payloads, RAGPipelineMode.TEXT_RETRIEVAL_MLLM)
        ),
        "multimodal_retrieval_mllm_count": len(
            _select_candidate_image_paths(payloads, RAGPipelineMode.MULTIMODAL_RETRIEVAL_MLLM)
        ),
    }
    out["pass"] = (
        out["text_only_count"] == 0
        and out["text_retrieval_mllm_count"] >= 1
        and out["multimodal_retrieval_mllm_count"] >= out["text_retrieval_mllm_count"]
    )
    return out


def agent_node(state: AgentState):
    messages = list(state["messages"])
    user_query = _latest_user_text(messages)
    system_prompt = _build_agent_system_prompt(state)

    if messages and getattr(messages[-1], "type", None) == "tool":
        raw_tool_results = _collect_trailing_tool_outputs(messages)
        tool_results, payloads = _extract_tool_payloads(raw_tool_results)
        non_tool_messages = [m for m in messages if getattr(m, "type", None) != "tool"]
        settings = get_settings()
        invoke_messages = non_tool_messages + [
            SystemMessage(
                content=(
                    f"{system_prompt}\n"
                    "Tool results are now available. Answer the user directly from those results.\n"
                    "Do not call more tools in this turn."
                )
            ),
        ]
        if settings.rag_mode == RAGPipelineMode.TEXT_ONLY:
            trace = {"mode": settings.rag_mode.value, "retrieval_sources": ["text"], "attached_images": 0}
            print(f"[AblationTrace] {json.dumps(trace)}")
            invoke_messages.append(HumanMessage(content=_build_post_tool_text_prompt(user_query, tool_results)))
        else:
            mm_content, trace = _build_multimodal_human_content(
                user_query=user_query,
                tool_results_text=tool_results,
                payloads=payloads,
                settings=settings,
            )
            print(f"[AblationTrace] {json.dumps(trace)}")
            invoke_messages.append(HumanMessage(content=mm_content))
    else:
        invoke_messages = messages + [SystemMessage(content=system_prompt)]

    planner_response = get_llm().invoke(invoke_messages)
    tool_calls = _normalize_tool_calls(getattr(planner_response, "tool_calls", []))
    ai_content = _render_assistant_content(getattr(planner_response, "content", "")).strip()
    ai = AIMessage(content=ai_content, tool_calls=tool_calls) if tool_calls else AIMessage(content=ai_content)
    return {
        "messages": state["messages"] + [ai],
        "selected_tools": tool_calls,
        "tool_sequence": _format_tool_sequence(tool_calls),
        "skip_reason": "" if tool_calls else "Agent returned final answer without tool call",
        "draft_response": ai_content if not tool_calls else "",
    }


tool_executor_node = ToolNode(TOOLS)


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


_LLM = None


def get_llm():
    global _LLM
    if _LLM is None:
        _LLM = build_llm(get_settings())
    return _LLM


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


def route_after_agent(state: AgentState):
    if str(state.get("tool_sequence", "none")) != "none":
        return "tool_executor"
    return "verification"


def route_after_verification(state: AgentState):
    report = state.get("verification_report", {}) or {}
    status = str(report.get("status", "pass"))
    if status == "revise":
        return "revise"
    if status == "abstain":
        return "abstain"
    return "pass"


graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tool_executor", tool_executor_node)
graph.add_node("verification", verification_node)
graph.set_entry_point("agent")
graph.add_conditional_edges(
    "agent",
    route_after_agent,
    {"tool_executor": "tool_executor", "verification": "verification"},
)
graph.add_edge("tool_executor", "agent")
graph.add_conditional_edges(
    "verification",
    route_after_verification,
    {"revise": "agent", "pass": END, "abstain": END},
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
