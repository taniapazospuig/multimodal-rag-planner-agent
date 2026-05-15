"""Run benchmark_v1.json against planner_agent variants.

Outputs report-ready raw rows and aggregate tables under eval/results/.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class Variant:
    name: str
    rag_mode: str | None
    metadata_filter_enabled: bool | None
    plain_llm: bool = False


VARIANTS = [
    Variant("plain_llm", None, None, plain_llm=True),
    Variant("text_only_filter_on", "text_only", True),
    Variant("text_retrieval_mllm_filter_on", "text_retrieval_mllm", True),
    Variant("multimodal_retrieval_mllm_filter_on", "multimodal_retrieval_mllm", True),
    Variant("text_only_filter_off", "text_only", False),
    Variant("text_retrieval_mllm_filter_off", "text_retrieval_mllm", False),
    Variant("multimodal_retrieval_mllm_filter_off", "multimodal_retrieval_mllm", False),
]

FINAL_AGENT_VARIANT = "multimodal_retrieval_mllm_filter_on"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_source_file_doc_map(documents_csv: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with documents_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            doc_id = str(row.get("doc_id", "")).strip()
            source_path = str(row.get("source_path", "")).strip()
            if doc_id and source_path:
                mapping[Path(source_path).name] = doc_id
    return mapping


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _tool_call_name(tool_call: Any) -> str:
    """Resolve a tool name from dict or provider-specific tool_call objects."""
    if tool_call is None:
        return ""
    if isinstance(tool_call, dict):
        name = tool_call.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
        fn = tool_call.get("function")
        if isinstance(fn, dict):
            inner = fn.get("name")
            if isinstance(inner, str) and inner.strip():
                return inner.strip()
    name_attr = getattr(tool_call, "name", None)
    if isinstance(name_attr, str) and name_attr.strip():
        return name_attr.strip()
    fn_obj = getattr(tool_call, "function", None)
    if fn_obj is not None:
        fname = getattr(fn_obj, "name", None)
        if isinstance(fname, str) and fname.strip():
            return fname.strip()
    return ""


def _is_tool_message(planner_agent: Any, message: Any) -> bool:
    if getattr(message, "type", None) == "tool":
        return True
    tool_cls = getattr(planner_agent, "ToolMessage", None)
    return bool(tool_cls is not None and isinstance(message, tool_cls))


def _completed_tool_results_as_dicts(raw: Any) -> list[dict[str, str]]:
    """Normalize ``AgentState.completed_tool_results`` entries for transcript merge."""
    if not isinstance(raw, list):
        return []
    output: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        content = str(item.get("content", "")).strip()
        if name and content:
            output.append({"name": name, "content": content})
    return output


def _merge_tool_transcripts(
    from_messages: list[dict[str, str]],
    from_completed: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Append completed-tool blobs not already present (same name+content) from message ToolMessages."""
    seen: set[tuple[str, str]] = {(d["name"], d["content"]) for d in from_messages}
    merged = list(from_messages)
    for item in from_completed:
        key = (item["name"], item["content"])
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def _tool_message_as_dict(planner_agent: Any, message: Any) -> dict[str, str] | None:
    if not _is_tool_message(planner_agent, message):
        return None
    name = str(getattr(message, "name", "") or "").strip()
    content = str(getattr(message, "content", "") or "").strip()
    if not name or not content:
        return None
    return {"name": name, "content": content}


def _append_unique_tool_message(
    output: list[dict[str, str]],
    seen: set[tuple[str, str]],
    item: dict[str, str],
) -> None:
    key = (item["name"], item["content"])
    if key in seen:
        return
    seen.add(key)
    output.append(item)


def _capture_graph_trace(
    planner_agent: Any,
    payload: Any,
    *,
    requested_tools: list[str],
    seen_tool_calls: set[tuple[str, str]],
    tool_messages: list[dict[str, str]],
    seen_tool_messages: set[tuple[str, str]],
) -> str:
    """Capture tool calls/results from LangGraph stream updates or final state payloads."""
    final_answer = ""
    if isinstance(payload, dict):
        completed = _completed_tool_results_as_dicts(payload.get("completed_tool_results"))
        for item in completed:
            _append_unique_tool_message(tool_messages, seen_tool_messages, item)

        raw_messages = payload.get("messages")
        messages = raw_messages if isinstance(raw_messages, list) else []
        for message in messages:
            for tool_call in getattr(message, "tool_calls", None) or []:
                name = _tool_call_name(tool_call)
                if not name:
                    continue
                call_id = ""
                args_repr = ""
                if isinstance(tool_call, dict):
                    call_id = str(tool_call.get("id", "") or "")
                    try:
                        args_repr = json.dumps(tool_call.get("args", {}), sort_keys=True, default=str)
                    except TypeError:
                        args_repr = str(tool_call.get("args", ""))
                key = (name, call_id or args_repr)
                if key not in seen_tool_calls:
                    seen_tool_calls.add(key)
                    requested_tools.append(name)

            item = _tool_message_as_dict(planner_agent, message)
            if item:
                _append_unique_tool_message(tool_messages, seen_tool_messages, item)
            elif getattr(message, "type", None) == "ai" and not (getattr(message, "tool_calls", None) or []):
                content = render_content(getattr(message, "content", ""))
                if content.strip():
                    final_answer = content

        for key, value in payload.items():
            if key in {"messages", "completed_tool_results"}:
                continue
            if isinstance(value, dict):
                nested = _capture_graph_trace(
                    planner_agent,
                    value,
                    requested_tools=requested_tools,
                    seen_tool_calls=seen_tool_calls,
                    tool_messages=tool_messages,
                    seen_tool_messages=seen_tool_messages,
                )
                if nested.strip():
                    final_answer = nested
    return final_answer


def extract_retrieved_doc_ids(tool_messages: list[dict[str, str]], source_file_doc_map: dict[str, str]) -> list[str]:
    doc_ids: list[str] = []
    token_pattern = re.compile(
        r"\bdoc_id=(doc_\d+)\b|\b(doc_\d+)\b|\bsource(?:_file)?=([^\s,]+)"
    )
    for message in tool_messages:
        content = message.get("content", "")
        for doc_id_match, bare_doc_id_match, source_file_match in token_pattern.findall(content):
            doc_id = doc_id_match or bare_doc_id_match
            if doc_id:
                doc_ids.append(doc_id)
                continue
            mapped_doc_id = source_file_doc_map.get(Path(source_file_match).name)
            if mapped_doc_id:
                doc_ids.append(mapped_doc_id)
    return dedupe_preserve_order(doc_ids)


def recall_at_k(retrieved: list[str], gold: list[str], k: int = 6) -> float:
    if not gold:
        return 0.0
    retrieved_set = set(retrieved[:k])
    return len(retrieved_set.intersection(gold)) / len(set(gold))


def mrr_at_k(retrieved: list[str], gold: list[str], k: int = 6) -> float:
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    for index, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in gold_set:
            return 1.0 / index
    return 0.0


def render_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text)
        if parts:
            return "\n".join(parts)
    return str(content)


def extract_first_json_object(raw_text: str) -> dict[str, Any] | None:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def set_variant_env(variant: Variant) -> None:
    if variant.rag_mode is None:
        os.environ.pop("PLANNER_RAG_MODE", None)
    else:
        os.environ["PLANNER_RAG_MODE"] = variant.rag_mode
    if variant.metadata_filter_enabled is None:
        os.environ.pop("RETRIEVAL_METADATA_FILTER_ENABLED", None)
    else:
        os.environ["RETRIEVAL_METADATA_FILTER_ENABLED"] = "1" if variant.metadata_filter_enabled else "0"


def load_planner_agent(variant: Variant):
    set_variant_env(variant)
    planner_agent = importlib.import_module("planner_agent")
    planner_agent = importlib.reload(planner_agent)
    reset_planner_singletons(planner_agent)
    return planner_agent


def reset_planner_singletons(planner_agent: Any) -> None:
    for name in (
        "_SETTINGS",
        "_LLM",
        "_ESTIMATOR_LLM",
        "_RETRIEVER",
        "_OPENCLIP_BACKBONE",
        "_DOCUMENT_ROWS",
        "_EXTERNAL_PAPER_ROWS",
        "_PAPER_ID_TO_DOC_ID",
        "_STUDY_ENVIRONMENT_ROWS",
    ):
        if hasattr(planner_agent, name):
            setattr(planner_agent, name, None)
    if hasattr(planner_agent, "_DOC_IDS_BY_TAG_CACHE"):
        planner_agent._DOC_IDS_BY_TAG_CACHE.clear()


def run_plain_llm_case(planner_agent: Any, case: dict[str, Any]) -> tuple[str, list[dict[str, str]], list[str]]:
    messages: list[Any] = [
        planner_agent.SystemMessage(
            content=(
                "Answer the user's study-planner question directly. "
                "You do not have access to tools or the private knowledge base."
            )
        )
    ]
    final_answer = ""
    for turn in case.get("turns", []):
        messages.append(planner_agent.HumanMessage(content=str(turn)))
        response = planner_agent.get_reasoning_llm().invoke(messages)
        final_answer = render_content(response.content)
        messages.append(response)
    return final_answer, [], []


def run_agent_case(planner_agent: Any, case: dict[str, Any]) -> tuple[str, list[dict[str, str]], list[str]]:
    planner_memory: dict[str, Any] = {}
    final_answer = ""
    all_tool_messages: list[dict[str, str]] = []
    seen_tool_messages: set[tuple[str, str]] = set()
    requested_tools: list[str] = []
    seen_tool_calls: set[tuple[str, str]] = set()

    for turn in case.get("turns", []):
        # Match planner_agent.py's interactive path: each user turn starts a fresh
        # provider-visible transcript and carries continuity through planner_memory.
        # This avoids feeding orphaned ToolMessages into the next evaluated turn.
        input_state = {
            "messages": [
                planner_agent.SYSTEM_PROMPT,
                planner_agent.HumanMessage(content=str(turn)),
            ],
            "planner_memory": dict(planner_memory),
            "pending_tool_calls": [],
            "completed_tool_results": [],
        }
        turn_answer = ""
        try:
            for update in planner_agent.app.stream(input_state, stream_mode="updates"):
                captured = _capture_graph_trace(
                    planner_agent,
                    update,
                    requested_tools=requested_tools,
                    seen_tool_calls=seen_tool_calls,
                    tool_messages=all_tool_messages,
                    seen_tool_messages=seen_tool_messages,
                )
                if captured.strip():
                    turn_answer = captured
                if isinstance(update, dict):
                    for payload in update.values():
                        if isinstance(payload, dict) and isinstance(payload.get("planner_memory"), dict):
                            planner_memory = dict(payload["planner_memory"])
        except TypeError:
            result = planner_agent.app.invoke(input_state)
            turn_answer = _capture_graph_trace(
                planner_agent,
                result,
                requested_tools=requested_tools,
                seen_tool_calls=seen_tool_calls,
                tool_messages=all_tool_messages,
                seen_tool_messages=seen_tool_messages,
            )
            merged_memory = result.get("planner_memory")
            if isinstance(merged_memory, dict):
                planner_memory = dict(merged_memory)
        if turn_answer.strip():
            final_answer = turn_answer

    return final_answer, all_tool_messages, requested_tools


def judge_answer(planner_agent: Any, case: dict[str, Any], answer: str, judge_model: str) -> dict[str, Any]:
    from langchain_openai import ChatOpenAI

    settings = planner_agent.get_settings()
    if not settings.openai_api_key:
        raise ValueError("OpenAI API key is missing. Set OPENAI_API_KEY.")
    judge = ChatOpenAI(
        model=judge_model,
        temperature=0,
        api_key=settings.openai_api_key,
    )
    prompt = [
        planner_agent.SystemMessage(
            content=(
                "You are an evaluation judge for a personal multimodal RAG planner. "
                "Score only whether the final answer satisfies the rubric hints. "
                "Do not reward style. Do not require exact wording if the answer is semantically correct. "
                "Return strict JSON only with keys: success, score, reason. "
                "success must be boolean. score must be 0 or 1. reason must be short."
            )
        ),
        planner_agent.HumanMessage(
            content=json.dumps(
                {
                    "case_id": case.get("id"),
                    "family": case.get("family"),
                    "turns": case.get("turns", []),
                    "final_answer": answer,
                    "rubric_hints": case.get("rubric_hints", []),
                    "expected_tools_suggested": case.get("expected_tools_suggested", []),
                    "gold_environment_ids_optional": case.get("gold_environment_ids_optional", []),
                },
                ensure_ascii=True,
            )
        ),
    ]
    raw = render_content(judge.invoke(prompt).content)
    payload = extract_first_json_object(raw) or {}
    success = bool(payload.get("success", False))
    score_raw = payload.get("score", 1 if success else 0)
    try:
        score = 1 if int(score_raw) >= 1 else 0
    except (TypeError, ValueError):
        score = 1 if success else 0
    return {
        "success": bool(success),
        "score": score,
        "reason": str(payload.get("reason", raw)).strip()[:500],
        "raw": raw,
    }


def make_result_row(
    *,
    case: dict[str, Any],
    variant: Variant,
    answer: str,
    tool_messages: list[dict[str, str]],
    requested_tools: list[str],
    source_file_doc_map: dict[str, str],
    latency_seconds: float,
    judge_latency_seconds: float,
    judge_result: dict[str, Any] | None,
    error: str,
) -> dict[str, Any]:
    retrieved_doc_ids = [] if variant.plain_llm else extract_retrieved_doc_ids(tool_messages, source_file_doc_map)
    gold_documents = [str(doc) for doc in case.get("gold_documents", [])]
    return {
        "case_id": case.get("id"),
        "family": case.get("family"),
        "variant": variant.name,
        "is_final_agent": variant.name == FINAL_AGENT_VARIANT,
        "rag_mode": variant.rag_mode or "none",
        "metadata_filter_enabled": variant.metadata_filter_enabled,
        "turns": case.get("turns", []),
        "gold_documents": gold_documents,
        "retrieved_doc_ids": retrieved_doc_ids,
        "recall_at_6": recall_at_k(retrieved_doc_ids, gold_documents, k=6),
        "mrr_at_6": mrr_at_k(retrieved_doc_ids, gold_documents, k=6),
        "answer_success": bool((judge_result or {}).get("success", False)),
        "answer_score": int((judge_result or {}).get("score", 0)),
        "judge_reason": str((judge_result or {}).get("reason", "")),
        "judge_raw": str((judge_result or {}).get("raw", "")),
        "latency_seconds": round(latency_seconds, 4),
        "judge_latency_seconds": round(judge_latency_seconds, 4),
        "tool_call_count": len(requested_tools),
        "requested_tools": requested_tools,
        "expected_tools_suggested": case.get("expected_tools_suggested", []),
        "tool_messages": tool_messages,
        "answer": answer,
        "error": error,
    }


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def summarize_rows(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(item) for item in group_keys)
        groups.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(part) for part in item[0])):
        summary = {name: value for name, value in zip(group_keys, key)}
        latencies = [float(row.get("latency_seconds", 0.0)) for row in group]
        summary.update(
            {
                "n_cases": len(group),
                "answer_success_rate": round(mean([float(row.get("answer_score", 0)) for row in group]), 4),
                "mean_recall_at_6": round(mean([float(row.get("recall_at_6", 0.0)) for row in group]), 4),
                "mean_mrr_at_6": round(mean([float(row.get("mrr_at_6", 0.0)) for row in group]), 4),
                "mean_latency_seconds": round(mean(latencies), 4),
                "median_latency_seconds": round(median(latencies), 4),
                "mean_tool_call_count": round(mean([float(row.get("tool_call_count", 0)) for row in group]), 4),
                "error_count": sum(1 for row in group if row.get("error")),
            }
        )
        summaries.append(summary)
    return summaries


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._\n"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = [str(row.get(column, "")) for column in columns]
        lines.append("| " + " | ".join(value.replace("|", "\\|") for value in values) + " |")
    return "\n".join(lines) + "\n"


def write_report_tables(path: Path, by_variant: list[dict[str, Any]], by_family: list[dict[str, Any]]) -> None:
    cols = [
        "variant",
        "n_cases",
        "answer_success_rate",
        "mean_recall_at_6",
        "mean_mrr_at_6",
        "mean_latency_seconds",
        "mean_tool_call_count",
        "error_count",
    ]
    variant_map = {row["variant"]: row for row in by_variant}
    plain_vs_final = [
        variant_map[name]
        for name in ("plain_llm", FINAL_AGENT_VARIANT)
        if name in variant_map
    ]
    multimodal = [
        variant_map[name]
        for name in (
            "text_only_filter_on",
            "text_retrieval_mllm_filter_on",
            "multimodal_retrieval_mllm_filter_on",
        )
        if name in variant_map
    ]
    metadata = [
        row
        for row in by_variant
        if row.get("variant") != "plain_llm"
    ]
    family_cols = ["family"] + cols
    content = [
        "# Planner Agent Evaluation Tables",
        "",
        "## Plain LLM vs Final Agent",
        markdown_table(plain_vs_final, cols),
        "## Multimodal Ablation",
        markdown_table(multimodal, cols),
        "## Metadata Filtering Ablation",
        markdown_table(metadata, cols),
        "## Family Breakdown",
        markdown_table(by_family, family_cols),
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def select_cases(cases: list[dict[str, Any]], smoke: bool) -> list[dict[str, Any]]:
    if not smoke:
        return cases
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for case in cases:
        family = str(case.get("family", ""))
        if family in seen:
            continue
        seen.add(family)
        output.append(case)
    return output


def run_benchmark(args: argparse.Namespace) -> None:
    benchmark_path = (PROJECT_ROOT / args.benchmark).resolve() if not Path(args.benchmark).is_absolute() else Path(args.benchmark)
    out_dir = (PROJECT_ROOT / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark = load_json(benchmark_path)
    cases = select_cases(list(benchmark.get("cases", [])), smoke=args.smoke)
    variants = [variant for variant in VARIANTS if not args.variant or variant.name == args.variant]
    source_file_doc_map = load_source_file_doc_map(PROJECT_ROOT / "data" / "kb" / "metadata" / "documents.csv")
    rows: list[dict[str, Any]] = []

    for variant in variants:
        print(f"[eval] variant={variant.name}", flush=True)
        planner_agent = load_planner_agent(variant)
        for case in cases:
            for repeat_index in range(args.repeats):
                print(f"[eval] case={case.get('id')} repeat={repeat_index + 1}/{args.repeats}", flush=True)
                answer = ""
                tool_messages: list[dict[str, str]] = []
                requested_tools: list[str] = []
                judge_result: dict[str, Any] | None = None
                error = ""
                agent_latency = 0.0
                judge_latency = 0.0
                agent_start = time.perf_counter()
                try:
                    if variant.plain_llm:
                        answer, tool_messages, requested_tools = run_plain_llm_case(planner_agent, case)
                    else:
                        answer, tool_messages, requested_tools = run_agent_case(planner_agent, case)
                    agent_latency = time.perf_counter() - agent_start
                    judge_start = time.perf_counter()
                    judge_result = judge_answer(planner_agent, case, answer, args.judge_model)
                    judge_latency = time.perf_counter() - judge_start
                except Exception as exc:
                    if agent_latency == 0.0:
                        agent_latency = time.perf_counter() - agent_start
                    error = f"{type(exc).__name__}: {exc}"
                row = make_result_row(
                    case=case,
                    variant=variant,
                    answer=answer,
                    tool_messages=tool_messages,
                    requested_tools=requested_tools,
                    source_file_doc_map=source_file_doc_map,
                    latency_seconds=agent_latency,
                    judge_latency_seconds=judge_latency,
                    judge_result=judge_result,
                    error=error,
                )
                if args.repeats > 1:
                    row["repeat_index"] = repeat_index
                rows.append(row)

    by_variant = summarize_rows(rows, ["variant"])
    by_family = summarize_rows(rows, ["family", "variant"])
    write_jsonl(out_dir / "runs.jsonl", rows)
    write_csv(out_dir / "summary_by_variant.csv", by_variant)
    write_csv(out_dir / "summary_by_family.csv", by_family)
    write_report_tables(out_dir / "report_tables.md", by_variant, by_family)
    print(f"[eval] wrote {out_dir}", flush=True)


def self_test() -> None:
    assert recall_at_k(["doc_001", "doc_002"], ["doc_001", "doc_002"], 6) == 1.0
    assert recall_at_k(["doc_001"], ["doc_001", "doc_002"], 6) == 0.5
    assert recall_at_k([], ["doc_001"], 6) == 0.0
    assert mrr_at_k(["doc_999", "doc_002"], ["doc_002"], 6) == 0.5
    assert mrr_at_k(["doc_999"], ["doc_002"], 6) == 0.0
    fake_map = {"source.pdf": "doc_123"}
    extracted = extract_retrieved_doc_ids(
        [
            {"name": "kb_course_retrieval", "content": "1. source_file=source.pdf\n2. doc_id=doc_456"},
            {"name": "kb_course_retrieval", "content": "3. source_file=source.pdf"},
        ],
        fake_map,
    )
    assert extracted == ["doc_123", "doc_456"], extracted

    assert _tool_call_name({"name": "kb_course_retrieval"}) == "kb_course_retrieval"
    assert _tool_call_name({"function": {"name": "find_free_slots"}}) == "find_free_slots"

    class _NamedToolCall:
        __slots__ = ("name",)

        def __init__(self, name: str) -> None:
            self.name = name

    assert _tool_call_name(_NamedToolCall("estimate_study_duration")) == "estimate_study_duration"

    merged = _merge_tool_transcripts(
        [{"name": "kb_course_retrieval", "content": "doc_id=doc_001"}],
        _completed_tool_results_as_dicts(
            [
                {"name": "kb_course_retrieval", "content": "doc_id=doc_001"},
                {"name": "kb_course_retrieval", "content": "doc_id=doc_002"},
            ]
        ),
    )
    assert merged == [
        {"name": "kb_course_retrieval", "content": "doc_id=doc_001"},
        {"name": "kb_course_retrieval", "content": "doc_id=doc_002"},
    ]
    assert extract_retrieved_doc_ids(merged, {}) == ["doc_001", "doc_002"]

    estimate_style = [
        {
            "name": "estimate_study_duration",
            "content": "rationale mentions doc_002\n- source=source.pdf, score=0.1",
        }
    ]
    assert extract_retrieved_doc_ids(estimate_style, fake_map) == ["doc_002", "doc_123"]

    completed_only = _completed_tool_results_as_dicts(
        [{"name": "kb_external_research_retrieval", "content": "1. source_file=source.pdf doc_id=doc_888"}]
    )
    assert extract_retrieved_doc_ids(completed_only, fake_map) == ["doc_123", "doc_888"]

    print("[self-test] ok")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="eval/benchmark_v1.json")
    parser.add_argument("--out-dir", default="eval/results")
    parser.add_argument("--judge-model", default=os.getenv("EVAL_JUDGE_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--smoke", action="store_true", help="Run one case per benchmark family.")
    parser.add_argument("--variant", choices=[variant.name for variant in VARIANTS])
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    run_benchmark(args)


if __name__ == "__main__":
    main()
