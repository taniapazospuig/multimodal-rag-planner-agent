.PHONY: ingest planner planner-filter planner-nofilter prep-and-run

RAG_MODE ?= text_retrieval_mllm

# A/B retrieval: metadata filters narrow Chroma + BM25 to resolved course/week (see config.RETRIEVAL_METADATA_FILTER_ENABLED).
# - planner / planner-filter: filtering ON (default baseline)
# - planner-nofilter: filtering OFF (rank over whole KB)

ingest:
	bash scripts/run_full_ingestion.sh

planner-filter:
	RETRIEVAL_METADATA_FILTER_ENABLED=1 PLANNER_RAG_MODE=$(RAG_MODE) python planner_agent.py

planner-nofilter:
	RETRIEVAL_METADATA_FILTER_ENABLED=0 PLANNER_RAG_MODE=$(RAG_MODE) python planner_agent.py

prep-and-run: ingest planner
