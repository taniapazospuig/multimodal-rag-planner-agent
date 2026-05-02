"""
Personal multimodal planner agent (starter).

LangGraph wiring mirrors `LangGraph_Agent_Demo_with_API/image_search_agent.py`:
agent node -> optional tools -> agent.

LLM: Gemini (API) or Ollama (local), selected via env — see `config.py` and `.env.example`.
Retrieval: OpenCLIP + Chroma for images; text retrieval stub for you to replace with
your text index / hybrid fusion for the three ablation modes.
"""

from __future__ import annotations

import csv
import json
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import List, TypedDict

from urllib3.exceptions import NotOpenSSLWarning
from PIL import Image
import chromadb
import torch
import open_clip
from rank_bm25 import BM25Okapi

from langchain.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from config import RAGPipelineMode, Settings, load_settings

warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="google")


# =========================
# Agent state
# =========================


class AgentState(TypedDict):
    messages: List[BaseMessage]


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

_SETTINGS: Settings | None = None


def get_settings() -> Settings:
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = load_settings()
    return _SETTINGS


@tool
def search_courses(query: str) -> str:
    """Search curated course rows (Operations Research, High-Dim Data, Generative AI)."""
    if not COURSES:
        return (
            "No courses.csv found. Add `data/kb/courses.csv` (see example in repo) "
            "with your real metadata, deadlines, and links."
        )

    query_tokens = query.lower().split()
    scored: list[tuple[int, dict]] = []
    for course in COURSES:
        text = " ".join(str(v) for v in course.values()).lower()
        score = sum(token in text for token in query_tokens)
        if score > 0:
            scored.append((score, course))

    if not scored:
        return "No matching course rows."

    scored.sort(key=lambda x: x[0], reverse=True)
    lines = []
    for _, c in scored[:8]:
        lines.append(
            f"{c.get('code', '?')}: {c.get('title', '')} — {c.get('notes', '')}"
        )
    return "\n".join(lines)


def _peek_indexed_text_context_mode(collection, bm25_rows: list[dict]) -> str | None:
    """Read context_mode stamped at index time (see scripts/index_text_chunks.py)."""
    for row in bm25_rows[:32]:
        meta = row.get("metadata") or {}
        cm = meta.get("context_mode")
        if cm in ("none", "metadata"):
            return str(cm)
    try:
        sample = collection.get(limit=1, include=["metadatas"])
        metas = sample.get("metadatas") or []
        if metas and metas[0]:
            cm = (metas[0] or {}).get("context_mode")
            if cm in ("none", "metadata"):
                return str(cm)
    except Exception:
        pass
    return None


def _format_text_context_mode_banner(settings: Settings, indexed_mode: str | None) -> str:
    env_mode = settings.text_context_mode
    if indexed_mode is None:
        return (
            f"[text context mode: env={env_mode}; indexed mode unknown — "
            "re-run scripts/index_text_chunks.py so metadata records context_mode]"
        )
    if indexed_mode != env_mode:
        return (
            f"[text context mode: index={indexed_mode} env={env_mode} "
            "(embeddings and stored text follow the index; align .env or re-index)]"
        )
    return f"[text context mode={indexed_mode}]"


@tool
def retrieve_context(query: str) -> str:
    settings = get_settings()
    mode = settings.rag_mode
    text_retriever = get_text_retriever()
    text_hits = text_retriever.search(query)
    if not text_hits:
        return (
            "No text index found yet. Run:\n"
            "python scripts/index_text_chunks.py\n"
            "Then retry your question."
        )

    lines = [
        f"[retrieval mode={mode.value}]",
        _format_text_context_mode_banner(settings, text_retriever.indexed_text_context_mode),
    ]
    lines.append("Top text context (BM25 + OpenCLIP fused with RRF):")
    for i, hit in enumerate(text_hits, start=1):
        meta = hit.get("metadata", {})
        score = float(hit.get("score", 0.0))
        text = str(hit.get("text", "")).replace("\n", " ").strip()
        if len(text) > 280:
            text = text[:280].rstrip() + "..."
        lines.append(
            f"{i}. score={score:.4f} | course={meta.get('course','?')} | week={meta.get('week','?')} | "
            f"doc_id={meta.get('doc_id','?')} | source={meta.get('source_path','?')}\n"
            f"   {text}"
        )

    if mode == RAGPipelineMode.MULTIMODAL_RETRIEVAL_MLLM:
        image_hits = get_image_index().search(query, k=3)
        if image_hits:
            lines.append("Top image context (OpenCLIP image index):")
            for i, (name, dist) in enumerate(image_hits, start=1):
                lines.append(f"{i}. {name} (distance {dist:.3f})")

    return "\n".join(lines)


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


def _simple_tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


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
                corpus_tokens = [row.get("tokens", []) for row in self.bm25_rows]
                self.bm25 = BM25Okapi(corpus_tokens)

        self.indexed_text_context_mode = _peek_indexed_text_context_mode(
            self.collection,
            self.bm25_rows,
        )

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
        tokens = _simple_tokenize(query)
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

    def search(self, query: str) -> list[dict]:
        course_filter = _detect_course_filter(query)
        where = {"course": course_filter} if course_filter else None

        dense_ids = self._dense_search(query, where=where)
        bm25_ids = self._bm25_search(query, course_filter=course_filter)
        fused = _rrf_fuse(dense_ids, bm25_ids, k=self.settings.rrf_k)
        ranked_ids = sorted(fused.keys(), key=lambda x: fused[x], reverse=True)[: self.settings.hybrid_k]

        out: list[dict] = []
        for chunk_id in ranked_ids:
            row = self.by_id.get(chunk_id)
            if row:
                out.append(
                    {
                        "chunk_id": chunk_id,
                        "score": fused[chunk_id],
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


# =========================
# CLIP + Chroma (image modality)
# =========================


class ImageIndex:
    """OpenCLIP image vectors in Chroma; text queries use the same CLIP text tower."""

    def __init__(self, settings: Settings, image_dir: str = "data/kb/images"):
        self.image_dir = _BASE_DIR / image_dir
        self.image_dir.mkdir(parents=True, exist_ok=True)

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

    def _index_new_images(self) -> None:
        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
        files = [f for f in self.image_dir.iterdir() if f.suffix.lower() in valid_exts]
        existing_ids = set(self.collection.get()["ids"] or [])

        new_images: list[Image.Image] = []
        new_ids: list[str] = []
        new_metas: list[dict] = []

        for path in files:
            file_id = f"img_{path.name}"
            if file_id in existing_ids:
                continue
            try:
                img = Image.open(path).convert("RGB")
                new_images.append(img)
                new_ids.append(file_id)
                new_metas.append({"filename": path.name, "path": str(path)})
            except Exception as e:
                print(f"Skipping {path.name}: {e}")

        if not new_images:
            return

        with torch.no_grad():
            batch = torch.cat(
                [self.preprocess(im).unsqueeze(0).to(self.device) for im in new_images],
                dim=0,
            )
            emb = self.model.encode_image(batch).cpu().numpy()
        self.collection.add(embeddings=emb.tolist(), ids=new_ids, metadatas=new_metas)

    def search(self, query: str, k: int = 4) -> list[tuple[str, float]]:
        with torch.no_grad():
            tokens = self.tokenizer([query]).to(self.device)
            q = self.model.encode_text(tokens).cpu().numpy()

        results = self.collection.query(query_embeddings=[q[0]], n_results=k)
        out: list[tuple[str, float]] = []
        metas = (results.get("metadatas") or [[]])[0]
        dists = (results.get("distances") or [[]])[0]
        for meta, dist in zip(metas, dists, strict=False):
            if meta and "filename" in meta:
                out.append((meta["filename"], float(dist)))
        return out


_IMAGE_INDEX: ImageIndex | None = None


def get_image_index() -> ImageIndex:
    global _IMAGE_INDEX
    if _IMAGE_INDEX is None:
        _IMAGE_INDEX = ImageIndex(get_settings())
    return _IMAGE_INDEX


@tool
def search_images(query: str) -> str:
    """Retrieve planner-related images (calendar screenshots, desk photos, whiteboards)."""
    if get_settings().rag_mode == RAGPipelineMode.TEXT_ONLY:
        return "Image search disabled in TEXT_ONLY ablation mode."

    results = get_image_index().search(query, k=4)
    if not results:
        return (
            "No indexed images. Drop files under `data/kb/images/` "
            "(.jpg / .png / .webp) and retry."
        )
    return "\n".join(f"{i+1}. {name} (distance {dist:.3f})" for i, (name, dist) in enumerate(results))


TOOLS = [search_courses, retrieve_context, search_images]


# =========================
# LLM factory
# =========================


def build_llm(settings: Settings):
    if settings.llm_backend.value == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        ).bind_tools(TOOLS)

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
    ).bind_tools(TOOLS)


_LLM = None


def get_llm():
    global _LLM
    if _LLM is None:
        _LLM = build_llm(get_settings())
    return _LLM


SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a personal university study and planning assistant.\n"
        "You have tools for courses, retrieval (stub), and image search.\n"
        "Use search_courses for degree/course structure questions.\n"
        "Use retrieve_context when the user needs facts from their KB (notes, PDFs, planners).\n"
        "Use search_images for visual memory: timetables, sketchnotes, environment cues.\n"
        "After tool results, answer directly. If tools return stubs, say what is missing honestly."
    )
)


def agent_node(state: AgentState):
    messages: List[BaseMessage] = state["messages"]

    last_user_msg = None
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_user_msg = m.content
            break

    if messages and getattr(messages[-1], "type", None) == "tool":
        tool_result = messages[-1].content
        non_tool = [m for m in messages if getattr(m, "type", None) != "tool"]
        messages = non_tool + [
            SystemMessage(
                content=(
                    "You have received tool results.\n"
                    f'The original user question was: "{last_user_msg}"\n'
                    "Answer the user directly using the tool results below.\n"
                    "Do NOT call any more tools."
                )
            ),
            HumanMessage(content=f"Tool results:\n{tool_result}"),
        ]
    else:
        messages = messages + [
            SystemMessage(
                content="Before answering, decide whether a tool would improve accuracy."
            )
        ]

    response = get_llm().invoke(messages)
    return {"messages": state["messages"] + [response]}


tool_node = ToolNode(TOOLS)


def route_after_agent(state: AgentState):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

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
            {
                "messages": [
                    SYSTEM_PROMPT,
                    HumanMessage(content=user),
                ]
            }
        )
        last = result["messages"][-1]
        print(f"\nAgent: {_render_assistant_content(last.content)}\n")
