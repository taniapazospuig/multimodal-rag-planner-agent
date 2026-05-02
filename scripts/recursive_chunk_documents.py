#!/usr/bin/env python3
"""
Create recursive text chunks from extracted KB units.

Inputs:
  data/kb/01_processed/extracted/documents_extracted.jsonl
  data/kb/01_processed/rendered/pdf_pages_manifest.jsonl (optional but recommended)

Outputs:
  data/kb/01_processed/chunked/chunks_recursive.jsonl
  data/kb/01_processed/chunked/chunk_schema.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTRACTED_PATH = PROJECT_ROOT / "data/kb/01_processed/extracted/documents_extracted.jsonl"
PAGES_MANIFEST_PATH = PROJECT_ROOT / "data/kb/01_processed/rendered/pdf_pages_manifest.jsonl"
OUT_PATH = PROJECT_ROOT / "data/kb/01_processed/chunked/chunks_recursive.jsonl"
SCHEMA_PATH = PROJECT_ROOT / "data/kb/01_processed/chunked/chunk_schema.json"

SEPARATORS = ["\n\n", "\n", ". ", "; ", ": ", " ", ""]
TECHNICAL_KEYWORDS = {
    "attention",
    "embedding",
    "transformer",
    "objective",
    "constraint",
    "optimization",
    "algorithm",
    "gradient",
    "retrieval",
    "rag",
    "multimodal",
    "indexing",
    "loss",
}


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def estimate_tokens(text: str) -> int:
    # Deterministic token estimate without external tokenizer dependency.
    return len(re.findall(r"[A-Za-z0-9_]+|[^\s]", text))


def merge_small_chunks(chunks: list[str], min_keep_tokens: int) -> list[str]:
    if not chunks:
        return []

    merged: list[str] = []
    i = 0
    while i < len(chunks):
        cur = chunks[i]
        cur_tokens = estimate_tokens(cur)
        if cur_tokens >= min_keep_tokens or len(chunks) == 1:
            merged.append(cur)
            i += 1
            continue

        if i + 1 < len(chunks):
            next_chunk = chunks[i + 1]
            merged.append(normalize_text(f"{cur} {next_chunk}"))
            i += 2
        elif merged:
            merged[-1] = normalize_text(f"{merged[-1]} {cur}")
            i += 1
        else:
            merged.append(cur)
            i += 1

    return [c for c in merged if c]


def _contains_technical_signal(text: str) -> bool:
    low = text.lower()
    return any(k in low for k in TECHNICAL_KEYWORDS)


def _is_low_value_slide_title_chunk(row: dict, text: str, tokens: int) -> bool:
    if str(row.get("resource_type", "")) != "annotated_lecture_slides":
        return False
    if tokens >= 30:
        return False
    if _contains_technical_signal(text):
        return False
    low = text.lower()
    title_hints = (
        "course overview",
        "outline",
        "thanks",
        "questions",
        "tentative syllabus",
        "teaching team",
        "what's new",
        "what’s new",
    )
    if any(h in low for h in title_hints):
        return True
    # Conservative fallback for short title-like slide strings.
    return text.count("\n") <= 2 and text.count(".") <= 1 and tokens < 20


def postprocess_chunk_texts(row: dict, chunk_texts: list[str]) -> list[str]:
    # Drop low-value tiny title chunks for annotated lecture slides.
    out: list[str] = []
    for text in chunk_texts:
        tokens = estimate_tokens(text)
        if _is_low_value_slide_title_chunk(row, text, tokens):
            continue
        normalized = normalize_text(text)
        if normalized:
            out.append(normalized)
    return out


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_page_image_lookup(rows: list[dict]) -> dict[tuple[str, str], str]:
    lookup: dict[tuple[str, str], str] = {}
    for row in rows:
        doc_id = str(row.get("doc_id", ""))
        unit_id = str(row.get("unit_id", ""))
        page_image_path = str(row.get("page_image_path", ""))
        if doc_id and unit_id and page_image_path:
            lookup[(doc_id, unit_id)] = page_image_path
    return lookup


def chunk_unit_text(
    text: str,
    target_tokens: int,
    overlap_tokens: int,
    min_keep_tokens: int,
    row: dict,
) -> list[str]:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency langchain-text-splitters. "
            "Install it with: python3 -m pip install langchain-text-splitters"
        ) from exc

    splitter = RecursiveCharacterTextSplitter(
        separators=SEPARATORS,
        chunk_size=target_tokens,
        chunk_overlap=overlap_tokens,
        length_function=estimate_tokens,
        is_separator_regex=False,
    )
    split_chunks = splitter.split_text(normalize_text(text))
    cleaned = [normalize_text(c) for c in split_chunks if normalize_text(c)]
    merged = merge_small_chunks(cleaned, min_keep_tokens=min_keep_tokens)
    post = postprocess_chunk_texts(row=row, chunk_texts=merged)
    return [c for c in post if normalize_text(c)]


def write_schema(path: Path) -> None:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "ChunkedDocumentUnit",
        "description": "One retrieval-ready chunk from extracted KB units.",
        "type": "object",
        "required": [
            "chunk_id",
            "chunk_index",
            "token_count_est",
            "doc_id",
            "source_path",
            "course",
            "week",
            "modality",
            "resource_type",
            "unit_id",
            "text",
            "extraction_method",
        ],
        "properties": {
            "chunk_id": {"type": "string", "minLength": 1},
            "chunk_index": {"type": "integer", "minimum": 0},
            "token_count_est": {"type": "integer", "minimum": 1},
            "doc_id": {"type": "string", "minLength": 1},
            "source_path": {"type": "string", "minLength": 1},
            "course": {"type": "string", "minLength": 1},
            "week": {"type": "string", "minLength": 1},
            "modality": {"type": "string", "minLength": 1},
            "resource_type": {"type": "string", "minLength": 1},
            "unit_id": {"type": "string", "minLength": 1},
            "text": {"type": "string", "minLength": 1},
            "extraction_method": {"type": "string", "minLength": 1},
            "image_path": {"type": ["string", "null"]},
            "page_image_path": {"type": ["string", "null"]},
        },
        "additionalProperties": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recursive chunking for extracted KB units.")
    parser.add_argument("--target-tokens", type=int, default=450)
    parser.add_argument("--overlap-tokens", type=int, default=60)
    parser.add_argument("--min-keep-tokens", type=int, default=120)
    parser.add_argument("--max-docs", type=int, default=0)
    args = parser.parse_args()

    extracted_rows = load_jsonl(EXTRACTED_PATH)
    page_rows = load_jsonl(PAGES_MANIFEST_PATH)
    page_lookup = build_page_image_lookup(page_rows)

    allowed_doc_ids: set[str] | None = None
    if args.max_docs > 0:
        doc_ids_in_order: list[str] = []
        seen: set[str] = set()
        for row in extracted_rows:
            doc_id = str(row.get("doc_id", ""))
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                doc_ids_in_order.append(doc_id)
        allowed_doc_ids = set(doc_ids_in_order[: args.max_docs])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_schema(SCHEMA_PATH)

    token_counts: list[int] = []
    chunk_ids_seen: set[str] = set()
    duplicate_chunk_ids = 0
    skipped_empty_chunks = 0
    modality_counts: Counter[str] = Counter()
    course_counts: Counter[str] = Counter()
    source_doc_ids: set[str] = set()

    written_chunks = 0
    with OUT_PATH.open("w", encoding="utf-8") as out_f:
        for row in extracted_rows:
            doc_id = str(row.get("doc_id", ""))
            if allowed_doc_ids is not None and doc_id not in allowed_doc_ids:
                continue

            source_doc_ids.add(doc_id)
            base_text = normalize_text(str(row.get("text", "")))
            if not base_text:
                continue

            chunk_texts = chunk_unit_text(
                text=base_text,
                target_tokens=args.target_tokens,
                overlap_tokens=args.overlap_tokens,
                min_keep_tokens=args.min_keep_tokens,
                row=row,
            )
            for idx, text in enumerate(chunk_texts):
                text = normalize_text(text)
                if not text:
                    skipped_empty_chunks += 1
                    continue

                token_count = estimate_tokens(text)
                if token_count <= 0:
                    skipped_empty_chunks += 1
                    continue

                chunk_id = f"{doc_id}__{row.get('unit_id', 'unit')}__chunk_{idx:04d}"
                if chunk_id in chunk_ids_seen:
                    duplicate_chunk_ids += 1
                    continue
                chunk_ids_seen.add(chunk_id)

                record = {
                    "chunk_id": chunk_id,
                    "chunk_index": idx,
                    "token_count_est": token_count,
                    "doc_id": doc_id,
                    "source_path": row.get("source_path"),
                    "course": row.get("course"),
                    "week": row.get("week"),
                    "modality": row.get("modality"),
                    "resource_type": row.get("resource_type"),
                    "unit_id": row.get("unit_id"),
                    "text": text,
                    "extraction_method": row.get("extraction_method"),
                    "image_path": row.get("image_path"),
                    "page_image_path": page_lookup.get((doc_id, str(row.get("unit_id", "")))),
                }
                out_f.write(json.dumps(record, ensure_ascii=True) + "\n")
                written_chunks += 1
                token_counts.append(token_count)
                modality_counts[str(row.get("modality", "unknown"))] += 1
                course_counts[str(row.get("course", "unknown"))] += 1

    print(f"Wrote chunks: {written_chunks}")
    print(f"Source docs processed: {len(source_doc_ids)}")
    print(f"Output path: {OUT_PATH}")
    print(f"Schema path: {SCHEMA_PATH}")
    print(f"Skipped empty chunks: {skipped_empty_chunks}")
    print(f"Duplicate chunk ids skipped: {duplicate_chunk_ids}")

    if token_counts:
        print(
            "Token stats (estimate): "
            f"min={min(token_counts)}, avg={mean(token_counts):.1f}, max={max(token_counts)}"
        )

    print("Chunks by modality:")
    for modality, count in sorted(modality_counts.items()):
        print(f"  {modality}: {count}")

    print("Chunks by course:")
    for course, count in sorted(course_counts.items()):
        print(f"  {course}: {count}")


if __name__ == "__main__":
    main()
