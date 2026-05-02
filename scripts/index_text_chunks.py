#!/usr/bin/env python3
"""
Index chunked KB text into Chroma using OpenCLIP text embeddings.

Also writes a BM25-ready corpus file so retrieval can run as hybrid
(dense + lexical) with rank fusion.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = PROJECT_ROOT / "data/kb/01_processed/chunked/chunks_recursive.jsonl"
CHROMA_PATH = PROJECT_ROOT / "chroma_db"
BM25_OUT_PATH = PROJECT_ROOT / "data/kb/02_index/bm25_corpus.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def batched(items: list[dict], batch_size: int) -> Iterable[list[dict]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def build_context_prefix(row: dict) -> str:
    course = str(row.get("course", "unknown"))
    week = str(row.get("week", "unknown"))
    modality = str(row.get("modality", "unknown"))
    resource_type = str(row.get("resource_type", "unknown"))
    doc_id = str(row.get("doc_id", "unknown"))
    return (
        f"[course={course}] [week={week}] [modality={modality}] "
        f"[resource_type={resource_type}] [doc_id={doc_id}]"
    )


def contextual_text(row: dict, context_mode: str) -> str:
    text = str(row.get("text", "")).strip()
    if context_mode == "none":
        return text
    if context_mode == "metadata":
        # Lightweight contextualization inspired by contextual retrieval:
        # prepend deterministic metadata so chunks keep local context.
        prefix = build_context_prefix(row)
        return f"{prefix}\n{text}".strip()
    raise ValueError(f"Unsupported context_mode: {context_mode}")


def sanitize_metadata(row: dict, context_mode: str) -> dict:
    allowed = {
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
        "extraction_method",
        "image_path",
        "page_image_path",
    }
    out: dict = {}
    for key in allowed:
        val = row.get(key)
        if val is None:
            continue
        if isinstance(val, (str, int, float, bool)):
            out[key] = val
        else:
            out[key] = str(val)
    out["context_mode"] = context_mode
    return out


def simple_tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def write_bm25_corpus(path: Path, rows: list[dict], context_mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            text = contextual_text(row, context_mode=context_mode)
            record = {
                "chunk_id": str(row.get("chunk_id", "")),
                "text": text,
                "tokens": simple_tokenize(text),
                "metadata": sanitize_metadata(row, context_mode=context_mode),
            }
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Index chunked text into Chroma + BM25 corpus.")
    parser.add_argument("--chunks-path", type=Path, default=CHUNKS_PATH)
    parser.add_argument("--chroma-path", type=Path, default=CHROMA_PATH)
    parser.add_argument("--collection", type=str, default="text_chunks")
    parser.add_argument("--bm25-out-path", type=Path, default=BM25_OUT_PATH)
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--context-mode",
        type=str,
        default="metadata",
        choices=["none", "metadata"],
        help="How chunks are contextualized before embedding/BM25 indexing.",
    )
    args = parser.parse_args()

    import chromadb  # type: ignore[import-not-found]
    import open_clip  # type: ignore[import-not-found]
    import torch

    rows = load_jsonl(args.chunks_path)
    if not rows:
        raise SystemExit(f"No chunk rows found: {args.chunks_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(args.model)

    client = chromadb.PersistentClient(path=str(args.chroma_path))
    collection = client.get_or_create_collection(
        name=args.collection,
        metadata={"hnsw:space": "cosine"},
    )

    # Idempotent indexing using upsert
    indexed = 0
    for batch_rows in batched(rows, args.batch_size):
        documents = [contextual_text(row, context_mode=args.context_mode) for row in batch_rows]
        ids = [str(row.get("chunk_id")) for row in batch_rows]
        metadatas = [sanitize_metadata(row, context_mode=args.context_mode) for row in batch_rows]
        with torch.no_grad():
            tokens = tokenizer(documents).to(device)
            emb = model.encode_text(tokens).cpu().numpy().tolist()
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=emb,
        )
        indexed += len(batch_rows)

    write_bm25_corpus(args.bm25_out_path, rows, context_mode=args.context_mode)

    print(f"Indexed chunks: {indexed}")
    print(f"Chroma path: {args.chroma_path}")
    print(f"Collection: {args.collection}")
    print(f"Context mode: {args.context_mode}")
    print(f"BM25 corpus: {args.bm25_out_path}")


if __name__ == "__main__":
    main()

