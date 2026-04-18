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
import warnings
from pathlib import Path
from typing import List, TypedDict

from urllib3.exceptions import NotOpenSSLWarning
from PIL import Image
import chromadb
import torch
import open_clip

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


@tool
def calc(expression: str) -> str:
    """Evaluate a simple arithmetic expression (for study time / grade estimates)."""
    allowed = set("0123456789+-*/(). %")
    if any(ch not in allowed for ch in expression):
        return "Error: invalid characters"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"


@tool
def retrieve_context(query: str) -> str:
    """
    Placeholder for your text / hybrid retriever over OCR text, captions, and metadata.

    Set `PLANNER_RAG_MODE` to switch ablation behaviour when you implement:
    - text_only: text embeddings only
    - text_retrieval_mllm: same index; MLLM consumes retrieved text (+ optional user image)
    - multimodal_retrieval_mllm: fuse CLIP image hits with text hits
    """
    mode = get_settings().rag_mode
    return (
        f"[stub retrieval | mode={mode.value}] No index wired yet for query: {query!r}. "
        "Implement Chroma text collection + metadata filters (course, week, modality)."
    )


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

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            settings.open_clip_model,
            pretrained=settings.open_clip_pretrained,
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(settings.open_clip_model)

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

        results = self.collection.query(query_embeddings=q, n_results=k)
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


TOOLS = [search_courses, calc, retrieve_context, search_images]


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
        "You have tools for courses, arithmetic, retrieval (stub), and image search.\n"
        "Use calc for any numeric reasoning.\n"
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
