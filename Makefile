.PHONY: ingest planner prep-and-run

ingest:
	bash scripts/run_full_ingestion.sh

planner:
	python planner_agent.py

prep-and-run: ingest planner
