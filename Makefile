.PHONY: ingest planner prep-and-run

RAG_MODE ?= text_retrieval_mllm

ingest:
	bash scripts/run_full_ingestion.sh

planner:
	PLANNER_RAG_MODE=$(RAG_MODE) python planner_agent.py

prep-and-run: ingest planner
