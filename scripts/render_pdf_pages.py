#!/usr/bin/env python3
"""
Render KB PDF pages to page images for multimodal retrieval.

Inputs:
  data/kb/metadata/documents.csv

Outputs:
  data/kb/01_processed/rendered/pdf_pages/<doc_id>/page_XXXX.png
  data/kb/01_processed/rendered/pdf_pages_manifest.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_CSV = PROJECT_ROOT / "data/kb/metadata/documents.csv"
OUT_DIR = PROJECT_ROOT / "data/kb/01_processed/rendered/pdf_pages"
MANIFEST_PATH = PROJECT_ROOT / "data/kb/01_processed/rendered/pdf_pages_manifest.jsonl"


def iter_pdf_rows(rows: list[dict[str, str]]) -> Iterable[dict[str, str]]:
    for row in rows:
        source_path = row.get("source_path", "")
        if source_path.lower().endswith(".pdf"):
            yield row


def render_pdf_doc(
    row: dict[str, str],
    scale: float,
    overwrite: bool,
) -> list[dict[str, str | int | float]]:
    try:
        import pypdfium2 as pdfium  # type: ignore
    except Exception as exc:
        raise RuntimeError("Missing dependency pypdfium2; install it first.") from exc

    source_rel = row["source_path"]
    source_abs = PROJECT_ROOT / source_rel
    if not source_abs.exists():
        return []

    try:
        pdf = pdfium.PdfDocument(str(source_abs))
    except Exception:
        return []

    doc_out_dir = OUT_DIR / row["doc_id"]
    doc_out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, str | int | float]] = []
    for idx in range(len(pdf)):
        page_number = idx + 1
        unit_id = f"page_{page_number:04d}"
        out_name = f"{unit_id}.png"
        out_abs = doc_out_dir / out_name
        out_rel = out_abs.relative_to(PROJECT_ROOT).as_posix()

        width = 0
        height = 0
        if overwrite or not out_abs.exists():
            try:
                page = pdf[idx]
                pil_image = page.render(scale=scale).to_pil()
                width, height = pil_image.size
                pil_image.save(out_abs, format="PNG")
            except Exception:
                continue
        else:
            try:
                from PIL import Image

                with Image.open(out_abs) as existing:
                    width, height = existing.size
            except Exception:
                width, height = 0, 0

        records.append(
            {
                "doc_id": row["doc_id"],
                "source_path": source_rel,
                "course": row["course"],
                "week": row["week"],
                "resource_type": row["resource_type"],
                "modality": row["modality"],
                "unit_id": unit_id,
                "page_number": page_number,
                "page_image_path": out_rel,
                "render_scale": scale,
                "image_width": width,
                "image_height": height,
            }
        )

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Render KB PDF pages to PNG images.")
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="Render scale (higher is sharper but slower/larger). Default: 2.0",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite already rendered page images.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="Optional limit for number of PDF docs to process (0 = all).",
    )
    args = parser.parse_args()

    if not DOCS_CSV.exists():
        raise SystemExit(f"documents.csv not found: {DOCS_CSV}")

    with DOCS_CSV.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    pdf_rows = list(iter_pdf_rows(rows))
    if args.max_docs > 0:
        pdf_rows = pdf_rows[: args.max_docs]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_docs = 0
    total_pages = 0
    with MANIFEST_PATH.open("w", encoding="utf-8") as out_f:
        for row in pdf_rows:
            records = render_pdf_doc(row=row, scale=args.scale, overwrite=args.force)
            if not records:
                continue
            total_docs += 1
            total_pages += len(records)
            for rec in records:
                out_f.write(json.dumps(rec, ensure_ascii=True) + "\n")

    print(f"Rendered PDF docs: {total_docs}")
    print(f"Rendered pages: {total_pages}")
    print(f"Images directory: {OUT_DIR}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
