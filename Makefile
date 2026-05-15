.PHONY: help ingest run run-eval \
	run-filter run-nofilter ablate-filter \
	run-text-only run-text-mllm run-multimodal-mllm ablate-multimodal \
	ablate-all prep-and-run

# Evaluation knobs (can be overridden on CLI):
# make run-eval JUDGE_MODEL=gpt-4.1-mini
FILTER ?= 1
MODE ?= text_retrieval_mllm
JUDGE_MODEL ?= gpt-4.1-mini
PYTHON ?= python3

help:
	@echo "Targets:"
	@echo "  make run                              # run with config.py/.env defaults"
	@echo "  make run-eval JUDGE_MODEL=..."
	@echo "  make ablate-filter                    # metadata filter on vs off"
	@echo "  make ablate-multimodal                # text_only vs text_mllm vs multimodal_mllm"
	@echo "  make ablate-all                       # run all ablations sequentially"

ingest:
	bash scripts/run_full_ingestion.sh

# Default run: no overrides (single source of truth = config.py + .env)
run:
	python planner_agent.py

# Run the benchmark suite and write report-ready results under eval/results/.
run-eval:
	EVAL_JUDGE_MODEL=$(JUDGE_MODEL) $(PYTHON) eval/run_benchmark.py

# 1) Metadata filter ablation
run-filter:
	RETRIEVAL_METADATA_FILTER_ENABLED=1 python planner_agent.py

run-nofilter:
	RETRIEVAL_METADATA_FILTER_ENABLED=0 python planner_agent.py

ablate-filter: run-filter run-nofilter

# 2) Pipeline mode ablation
run-text-only:
	PLANNER_RAG_MODE=text_only python planner_agent.py

run-text-mllm:
	PLANNER_RAG_MODE=text_retrieval_mllm python planner_agent.py

run-multimodal-mllm:
	PLANNER_RAG_MODE=multimodal_retrieval_mllm python planner_agent.py

ablate-multimodal: run-text-only run-text-mllm run-multimodal-mllm

ablate-all: ablate-filter ablate-multimodal

prep-and-run: ingest run
