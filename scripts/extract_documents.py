#!/usr/bin/env python3
"""
Extract all KB documents into one JSONL file for chunking.

Output:
  data/kb/01_processed/extracted/documents_extracted.jsonl
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_CSV = PROJECT_ROOT / "data/kb/metadata/documents.csv"
OUT_PATH = PROJECT_ROOT / "data/kb/01_processed/extracted/documents_extracted.jsonl"

# Keep extraction output readable when parsing imperfect PDFs.
logging.getLogger("pypdf").setLevel(logging.ERROR)

def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def cleanup_handwritten_or_text(text: str) -> str:
    """
    Conservative cleanup for Operations Research handwritten GoodNotes pages.
    Keep original meaning while reducing obvious OCR/PDF extraction artifacts.
    """
    cleaned = text
    # Split obvious number/letter boundaries in formulas and mixed tokens.
    cleaned = re.sub(r"([A-Za-z])(\d)", r"\1 \2", cleaned)
    cleaned = re.sub(r"(\d)([A-Za-z])", r"\1 \2", cleaned)
    # Normalize repeated punctuation artifacts.
    cleaned = re.sub(r"([,;:.!?])\1{1,}", r"\1", cleaned)
    # Normalize spacing around common math operators.
    cleaned = re.sub(r"\s*([=+\-*/<>])\s*", r" \1 ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return normalize_text(cleaned)


def choose_method(row: dict[str, str]) -> str:
    source_path = row["source_path"].lower()
    ext = os.path.splitext(source_path)[1]
    course = row.get("course", "").lower()

    if ext in {".md", ".txt"}:
        return "plain_text"
    if ext == ".py":
        return "code_text"
    if ext == ".ipynb":
        return "notebook_cell"
    if ext == ".rtf":
        return "rtf_text"
    if ext == ".pdf":
        # GoodNotes PDFs are exported with selectable text; OCR would duplicate it
        # and add noisy layers that hurt retrieval.
        return "pdf_text"
    if ext in {".png", ".jpg", ".jpeg", ".webp"}:
        if course == "study-environment":
            return "image_no_ocr"
        return "ocr"
    return "plain_text"


def safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def strip_rtf(raw: str) -> str:
    # Lightweight RTF cleanup that keeps visible text.
    text = re.sub(r"\\'[0-9a-fA-F]{2}", "", raw)
    text = re.sub(r"\\[a-zA-Z]+\d* ?", "", text)
    text = re.sub(r"[{}]", "", text)
    return normalize_text(text)


def extract_notebook_cells(path: Path) -> list[tuple[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    cells = data.get("cells", [])
    results: list[tuple[str, str]] = []
    for idx, cell in enumerate(cells):
        cell_type = cell.get("cell_type", "unknown")
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        text = normalize_text(str(source))
        if text:
            unit_id = f"cell_{idx:04d}_{cell_type}"
            results.append((unit_id, text))
    return results


def extract_pdf_pages(path: Path) -> tuple[dict[str, str], int]:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        return {}, 0

    try:
        reader = PdfReader(str(path), strict=False)
    except Exception:
        return {}, 0

    results: dict[str, str] = {}
    for idx, page in enumerate(reader.pages, start=1):
        text = normalize_text(page.extract_text() or "")
        if text:
            results[f"page_{idx:04d}"] = text
    return results, len(reader.pages)


def extract_image_ocr(path: Path) -> str:
    try:
        import pytesseract  # type: ignore
        from PIL import Image
    except Exception:
        return ""
    try:
        text = pytesseract.image_to_string(Image.open(path))
        return normalize_text(text)
    except Exception:
        return ""


def emit_record(
    row: dict[str, str],
    method: str,
    unit_id: str,
    text: str,
    image_path: str | None,
) -> dict[str, str | None]:
    return {
        "doc_id": row["doc_id"],
        "source_path": row["source_path"],
        "course": row["course"],
        "week": row["week"],
        "modality": row["modality"],
        "resource_type": row["resource_type"],
        "unit_id": unit_id,
        "text": text,
        "image_path": image_path,
        "extraction_method": method,
    }


def iter_records(row: dict[str, str]) -> Iterable[dict[str, str | None]]:
    method = choose_method(row)
    source_path = PROJECT_ROOT / row["source_path"]
    ext = source_path.suffix.lower()

    if method in {"plain_text", "code_text"}:
        text = normalize_text(safe_read_text(source_path))
        if text:
            yield emit_record(row, method, "full_text", text, None)
        else:
            yield emit_record(row, method, "full_text", "[EMPTY_TEXT]", None)
        return

    if method == "rtf_text":
        text = strip_rtf(safe_read_text(source_path))
        if text:
            yield emit_record(row, method, "full_text", text, None)
        else:
            yield emit_record(row, method, "full_text", "[EMPTY_RTF_TEXT]", None)
        return

    if method == "notebook_cell":
        cells = extract_notebook_cells(source_path)
        if cells:
            for unit_id, text in cells:
                yield emit_record(row, method, unit_id, text, None)
        else:
            yield emit_record(row, method, "cell_0000_unknown", "[EMPTY_NOTEBOOK]", None)
        return

    if method == "pdf_text":
        text_pages, _ = extract_pdf_pages(source_path)
        is_or_handwritten = (
            row.get("course") == "operations-research"
            and "handwritten-notes-goodnotes" in row.get("source_path", "").lower()
        )

        if text_pages:
            for unit_id in sorted(text_pages):
                text = text_pages[unit_id]
                if text and is_or_handwritten:
                    text = cleanup_handwritten_or_text(text)
                if text:
                    yield emit_record(
                        row,
                        method,
                        unit_id,
                        text,
                        f"{row['source_path']}#{unit_id}",
                    )
        else:
            fallback_text = normalize_text(
                f"PDF context only: {row.get('title','untitled')} "
                f"({row.get('course','')}, {row.get('week','')}, {row.get('resource_type','')})."
            )
            yield emit_record(row, method, "page_0001", fallback_text, f"{row['source_path']}#page_0001")
        return

    if method == "ocr" and ext in {".png", ".jpg", ".jpeg", ".webp"}:
        ocr_text = extract_image_ocr(source_path)
        combined = normalize_text(ocr_text)
        if not combined:
            combined = "[EMPTY_OCR_TEXT]"
        yield emit_record(row, method, "image_0001", combined, row["source_path"])
        return

    if method == "image_no_ocr" and ext in {".png", ".jpg", ".jpeg", ".webp"}:
        yield emit_record(row, method, "image_0001", "[IMAGE_NO_OCR]", row["source_path"])
        return

    # Final fallback
    fallback = normalize_text(safe_read_text(source_path)) if source_path.exists() else ""
    if not fallback:
        fallback = f"[UNSUPPORTED_FILE_TYPE] {source_path.name}"
    yield emit_record(row, "plain_text", "full_text", fallback, None)


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with DOCS_CSV.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    method_counts: Counter[str] = Counter()
    record_count = 0

    with OUT_PATH.open("w", encoding="utf-8") as out_f:
        for row in rows:
            method = choose_method(row)
            method_counts[method] += 1
            for record in iter_records(row):
                out_f.write(json.dumps(record, ensure_ascii=True) + "\n")
                record_count += 1

    print(f"Wrote {record_count} extracted units to: {OUT_PATH}")
    print("Documents by selected extraction method:")
    for method, count in sorted(method_counts.items()):
        print(f"  {method}: {count}")


if __name__ == "__main__":
    main()
