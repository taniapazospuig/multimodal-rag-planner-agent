# Personal Multimodal Planner RAG (starter)

Scaffold for a **personalised, multimodal study planner** with LangGraph, **OpenCLIP + ChromaDB** for image retrieval, and a **configurable LLM** (Gemini API or local Ollama). This mirrors the layout of `LangGraph_Agent_Demo_with_API/` while leaving hooks for your three RAG ablations and richer KB metadata.

## Quickstart

```bash
cd Personal_Multimodal_Planner_RAG
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env: set PLANNER_LLM_BACKEND and the corresponding API URL / key.
python planner_agent.py
```

Environment variables (see `.env.example`):

- `PLANNER_LLM_BACKEND`: `gemini` (default) or `ollama`
- `PLANNER_RAG_MODE`: `text_only` | `text_retrieval_mllm` | `multimodal_retrieval_mllm` (controls whether `search_images` is allowed)
- `GEMINI_MODEL`, `GOOGLE_API_KEY` / `GEMINI_API_KEY` for Gemini
- `OLLAMA_BASE_URL`, `OLLAMA_MODEL` for Ollama (e.g. `llava-phi3:3.8b`)

## Repo layout

- `planner_agent.py` — LangGraph agent, tools, CLIP/Chroma image index
- `config.py` — settings and `RAGPipelineMode` enum for ablations
- `data/kb/courses.csv` — seed rows for your three courses (replace with real data)
- `data/kb/images/` — image KB drop zone
- `chroma_db/` — local persistent Chroma (gitignored)

## Choosing Ollama (LLaVA-Phi3) vs Gemini Flash‑Lite

**Use Gemini Flash‑Lite (API)** when you want stronger multimodal reading of messy notes and slides, faster iteration, and no local GPU requirement. It is usually the better default for **understanding** retrieved text plus optional images in one pass, which matters for conditions (2) and (3) in your study.

**Use Ollama LLaVA‑Phi3 locally** when **privacy** matters (raw course PDFs and handwriting staying on machine), you need **offline** demos, or you want **frozen, local** inference for reproducibility without API drift. Expect more hallucination on dense tables and smaller text unless you crop/zoom images.

**For a clean ablation paper**, keep the **reader model fixed** across pipelines (2) and (3) so you measure retrieval, not “which MLLM is stronger.” You can still run a **separate** sensitivity analysis swapping Gemini vs LLaVA.

## Next implementation steps

1. **Ingestion**: OCR / PDF text pipeline → chunked text embeddings in Chroma (separate collection from images). Store rich metadata: `course`, `week`, `modality`, `cognitive_load`, `dependencies`, etc.
2. **Hybrid retrieval**: BM25 or sparse + dense fusion; optional cross-encoder rerank on top‑k.
3. **Three pipelines**: Branch on `RAGPipelineMode` — text-only answers vs passing image paths to the MLLM vs CLIP-fused context.
4. **Evaluation**: Held-out question set per course; metrics such as answer correctness, citation overlap, schedule constraint satisfaction, latency, cost; plain LLM and no-retrieval baselines.
5. **GitHub**: `git init`, add remote, ensure `.env` and `data/kb/private/` never commit secrets or copyrighted LMS PDFs (keep a `private/` tree gitignored and document what you store there).
