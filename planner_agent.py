"""
Minimal planner agent with a single capability:
C1 TopicSummary (topics studied in course/week from retrieved evidence).

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


COURSES_CSV_PATH = _BASE_DIR / "data" / "kb" / "metadata" / "courses.csv"
COURSES = load_courses(COURSES_CSV_PATH)

_SETTINGS: Settings | None = None
_LLM = None
_RETRIEVER: "CourseRetriever | None" = None
_OPENCLIP_BACKBONE: "OpenCLIPBackbone | None" = None


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
    explicit_course = course_filter.strip().lower()
    resolved_course = explicit_course
    if not resolved_course and settings.course_filter_enabled:
        resolved_course = _detect_course_filter(query)
    resolved_week = week_filter.strip().lower() or _detect_week_filter(query)

    hits = get_retriever().search(
        query=query.strip(),
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


TOOLS = [kb_course_retrieval]


SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a C1 TopicSummary assistant.\n"
        "C1 definition: Summarize key topics studied in a given course/week from retrieved evidence.\n"
        "Use kb_course_retrieval to gather evidence first."
    )
)


def _build_llm(settings: Settings):
    """Create LLM backend and bind available tools."""
    if settings.llm_backend == LLMBackend.OLLAMA:
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        ).bind_tools(TOOLS)

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


def get_llm():
    """Return singleton LLM instance."""
    global _LLM
    if _LLM is None:
        _LLM = _build_llm(get_settings())
    return _LLM


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
    """LLM node: chooses tool first, then produces user-facing topic summary."""
    messages = state["messages"]
    user_query = _latest_user_text(messages)

    if messages and getattr(messages[-1], "type", None) == "tool":
        settings = get_settings()
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
        image_paths = _extract_image_paths(tool_result)
        include_images = settings.rag_mode in {
            RAGPipelineMode.TEXT_RETRIEVAL_MLLM,
            RAGPipelineMode.MULTIMODAL_RETRIEVAL_MLLM,
        }
        non_tool_messages = [m for m in messages if getattr(m, "type", None) != "tool"]
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
        prompt_messages = messages + [SYSTEM_PROMPT]

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
    print("\nPlanner agent (C1 only) ready. Type 'exit' to quit.\n")
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
