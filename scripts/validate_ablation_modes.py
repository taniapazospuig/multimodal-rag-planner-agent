"""Quick sanity checks for ablation mode image-understanding behavior."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from planner_agent import run_ablation_mode_sanity_checks
except ModuleNotFoundError:
    run_ablation_mode_sanity_checks = None


def main() -> int:
    if run_ablation_mode_sanity_checks is None:
        planner_source = (REPO_ROOT / "planner_agent.py").read_text(encoding="utf-8")
        result = {
            "fallback_check": True,
            "has_text_only_branch": "RAGPipelineMode.TEXT_ONLY" in planner_source,
            "has_text_retrieval_mllm_branch": "RAGPipelineMode.TEXT_RETRIEVAL_MLLM" in planner_source,
            "has_multimodal_branch": "RAGPipelineMode.MULTIMODAL_RETRIEVAL_MLLM" in planner_source,
            "has_payload_marker": "__MMLM_PAYLOAD__=" in planner_source,
            "has_image_attachment": "\"type\": \"image_url\"" in planner_source,
        }
        result["pass"] = all(
            bool(result[k])
            for k in (
                "has_text_only_branch",
                "has_text_retrieval_mllm_branch",
                "has_multimodal_branch",
                "has_payload_marker",
                "has_image_attachment",
            )
        )
    else:
        result = run_ablation_mode_sanity_checks()
    print("Ablation sanity checks:", result)
    return 0 if result.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
