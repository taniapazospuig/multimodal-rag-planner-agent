#!/usr/bin/env python3
"""
Index raw KB images into Chroma using OpenCLIP image embeddings.

This pre-indexes images from selected subfolders in:
  data/kb/00_raw/

Default folders:
  - planner-screenshots
  - study-location-photos
  - todo-lists

Outputs:
  chroma_db/ collection (default: planner_images)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import chromadb  # type: ignore[import-not-found]
import open_clip  # type: ignore[import-not-found]
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data/kb/00_raw"
DEFAULT_CHROMA_PATH = PROJECT_ROOT / "chroma_db"
DEFAULT_DOCUMENTS_CSV = PROJECT_ROOT / "data/kb/metadata/documents.csv"
DEFAULT_FOLDERS = ("planner-screenshots", "study-location-photos", "todo-lists")
VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def batched(items: list[dict], batch_size: int) -> Iterable[list[dict]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def to_image_id(path: Path) -> str:
    """Generate a stable ID aligned with planner_agent's manual image IDs."""
    rel = path.relative_to(PROJECT_ROOT).as_posix().replace("/", "__")
    return f"img__{rel}"


def load_documents_metadata(documents_csv: Path) -> dict[str, dict[str, str]]:
    """Load source_path -> metadata mapping from documents.csv."""
    if not documents_csv.exists():
        raise SystemExit(f"documents.csv not found: {documents_csv}")
    out: dict[str, dict[str, str]] = {}
    with documents_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source_path = str(row.get("source_path", "")).strip().replace("\\", "/")
            if not source_path:
                continue
            out[source_path] = {
                "doc_id": str(row.get("doc_id", "")).strip(),
                "course": str(row.get("course", "")).strip(),
                "week": str(row.get("week", "")).strip(),
                "resource_type": str(row.get("resource_type", "")).strip(),
            }
    return out


def sanitize_metadata(path: Path, raw_root: Path, doc_meta: dict[str, str]) -> dict:
    rel_project = path.relative_to(PROJECT_ROOT).as_posix()
    rel_raw = path.relative_to(raw_root).as_posix()
    raw_folder = rel_raw.split("/", maxsplit=1)[0] if "/" in rel_raw else rel_raw
    return {
        "source_kind": "manual_image",
        "filename": path.name,
        "path": str(path),
        "relative_path": rel_project,
        "raw_folder": raw_folder,
        "resource_type": str(doc_meta.get("resource_type", "")).strip(),
        "doc_id": str(doc_meta.get("doc_id", "")).strip(),
        "course": str(doc_meta.get("course", "")).strip(),
        "week": str(doc_meta.get("week", "")).strip(),
    }


def collect_images(raw_root: Path, folders: list[str]) -> list[Path]:
    images: list[Path] = []
    for folder in folders:
        folder_path = raw_root / folder
        if not folder_path.exists():
            continue
        for path in folder_path.rglob("*"):
            if path.is_file() and path.suffix.lower() in VALID_EXTS:
                images.append(path)
    return sorted(images)


def parse_folders(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Index selected raw image folders into Chroma.")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--folders", type=str, default=",".join(DEFAULT_FOLDERS))
    parser.add_argument("--documents-csv", type=Path, default=DEFAULT_DOCUMENTS_CSV)
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

    folders = parse_folders(args.folders)
    if not folders:
        raise SystemExit("No folders provided. Use --folders with comma-separated folder names.")
    if not args.raw_root.exists():
        raise SystemExit(f"Raw root does not exist: {args.raw_root}")
    doc_meta_by_source = load_documents_metadata(args.documents_csv)

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

    image_paths = collect_images(args.raw_root, folders)
    candidates: list[dict] = []
    skipped_missing_metadata = 0
    for path in image_paths:
        image_id = to_image_id(path)
        if not args.force and image_id in existing_ids:
            continue
        source_key = path.relative_to(PROJECT_ROOT).as_posix()
        doc_meta = doc_meta_by_source.get(source_key)
        # Strict mode: index only files present in documents.csv mapping.
        if not doc_meta or not str(doc_meta.get("resource_type", "")).strip():
            skipped_missing_metadata += 1
            continue
        candidates.append(
            {
                "id": image_id,
                "path": path,
                "metadata": sanitize_metadata(path, args.raw_root, doc_meta),
            }
        )

    if not candidates:
        print(f"Discovered images: {len(image_paths)}")
        print("Indexed images: 0 (all already present or no matching files)")
        print(f"Collection: {args.collection}")
        print(f"Chroma path: {args.chroma_path}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    model = model.to(device).eval()

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

    print(f"Folders: {', '.join(folders)}")
    print(f"Discovered images: {len(image_paths)}")
    print(f"Skipped (missing documents.csv metadata): {skipped_missing_metadata}")
    print(f"Indexed images: {indexed}")
    print(f"Collection: {args.collection}")
    print(f"Chroma path: {args.chroma_path}")


if __name__ == "__main__":
    main()
