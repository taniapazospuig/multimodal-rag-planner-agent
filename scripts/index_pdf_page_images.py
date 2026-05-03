#!/usr/bin/env python3
"""
Index rendered PDF page images into Chroma using OpenCLIP image embeddings.

Inputs:
  data/kb/01_processed/rendered/pdf_pages_manifest.jsonl

Outputs:
  chroma_db/ collection (default: planner_images)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data/kb/01_processed/rendered/pdf_pages_manifest.jsonl"
DEFAULT_CHROMA_PATH = PROJECT_ROOT / "chroma_db"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def batched(items: list[dict], batch_size: int) -> Iterable[list[dict]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def to_image_id(row: dict) -> str:
    doc_id = str(row.get("doc_id", "")).strip()
    unit_id = str(row.get("unit_id", "")).strip()
    if not doc_id or not unit_id:
        return ""
    return f"pdf__{doc_id}__{unit_id}"


def sanitize_metadata(row: dict) -> dict:
    return {
        "source_kind": "pdf_render",
        "filename": Path(str(row.get("page_image_path", ""))).name,
        "path": str(PROJECT_ROOT / str(row.get("page_image_path", ""))),
        "relative_path": str(row.get("page_image_path", "")),
        "doc_id": str(row.get("doc_id", "")),
        "unit_id": str(row.get("unit_id", "")),
        "page_number": int(row.get("page_number", 0) or 0),
        "course": str(row.get("course", "")),
        "week": str(row.get("week", "")),
        "resource_type": str(row.get("resource_type", "")),
        "source_path": str(row.get("source_path", "")),
        "render_scale": float(row.get("render_scale", 0.0) or 0.0),
        "image_width": int(row.get("image_width", 0) or 0),
        "image_height": int(row.get("image_height", 0) or 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Index rendered PDF pages into Chroma.")
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--chroma-path", type=Path, default=DEFAULT_CHROMA_PATH)
    parser.add_argument("--collection", type=str, default="planner_images")
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and rebuild the collection before indexing.",
    )
    args = parser.parse_args()

    import chromadb  # type: ignore[import-not-found]
    import open_clip  # type: ignore[import-not-found]
    import torch
    from PIL import Image

    rows = load_jsonl(args.manifest_path)
    if not rows:
        raise SystemExit(f"No manifest rows found: {args.manifest_path}")

    client = chromadb.PersistentClient(path=str(args.chroma_path))
    if args.force:
        try:
            client.delete_collection(args.collection)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=args.collection,
        metadata={"hnsw:space": "cosine"},
    )

    existing_ids = set(collection.get()["ids"] or [])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    model = model.to(device).eval()

    candidates: list[dict] = []
    for row in rows:
        image_id = to_image_id(row)
        if not image_id:
            continue
        if not args.force and image_id in existing_ids:
            continue
        image_rel = str(row.get("page_image_path", "")).strip()
        if not image_rel:
            continue
        image_abs = PROJECT_ROOT / image_rel
        if not image_abs.exists():
            continue
        candidates.append(
            {
                "id": image_id,
                "path": image_abs,
                "metadata": sanitize_metadata(row),
            }
        )

    indexed = 0
    for batch in batched(candidates, args.batch_size):
        images = []
        ids = []
        metadatas = []
        for item in batch:
            try:
                with Image.open(item["path"]) as opened:
                    img = opened.convert("RGB")
            except Exception:
                continue
            images.append(img)
            ids.append(item["id"])
            metadatas.append(item["metadata"])

        if not images:
            continue

        with torch.no_grad():
            tensor_batch = torch.cat(
                [preprocess(im).unsqueeze(0).to(device) for im in images],
                dim=0,
            )
            emb = model.encode_image(tensor_batch).cpu().numpy().tolist()

        collection.upsert(ids=ids, embeddings=emb, metadatas=metadatas)
        indexed += len(ids)

    print(f"Manifest rows: {len(rows)}")
    print(f"Indexed pages: {indexed}")
    print(f"Collection: {args.collection}")
    print(f"Chroma path: {args.chroma_path}")


if __name__ == "__main__":
    main()
