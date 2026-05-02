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


def choose_method(row: dict[str, str]) -> str:
    source_path = row["source_path"].lower()
    ext = os.path.splitext(source_path)[1]
    course = row["course"].lower()

    if ext in {".md", ".txt"}:
        return "plain_text"
    if ext == ".py":
        return "code_text"
    if ext == ".ipynb":
        return "notebook_cell"
    if ext == ".rtf":
        return "rtf_text"
    if ext == ".pdf":
        # PDF OCR intentionally disabled: GoodNotes exports already provide
        # selectable native text, and extra OCR introduced duplicate/noisy text.
        return "pdf_text"
    if ext in {".png", ".jpg", ".jpeg", ".webp"}:
        if course in {"study-environment"}:
            return "ocr_plus_caption"
        if course in {"personal-planner", "personal-todo"}:
            return "ocr"
        return "ocr_plus_caption"
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


def extract_pdf_page_ocr(path: Path, target_page_ids: set[str] | None = None) -> dict[str, str]:
    """
    OCR each PDF page by rendering to an image first.
    Returns a map: {unit_id -> ocr_text}
    """
    try:
        import pypdfium2 as pdfium  # type: ignore
        import pytesseract  # type: ignore
    except Exception:
        return {}

    results: dict[str, str] = {}
    try:
        pdf = pdfium.PdfDocument(str(path))
    except Exception:
        return {}

    for idx in range(len(pdf)):
        unit_id = f"page_{idx + 1:04d}"
        if target_page_ids is not None and unit_id not in target_page_ids:
            continue
        try:
            page = pdf[idx]
            # Scale 2.0 improves OCR readability without huge memory growth.
            pil_image = page.render(scale=2.0).to_pil()
            ocr_text = normalize_text(pytesseract.image_to_string(pil_image))
            if ocr_text:
                results[unit_id] = ocr_text
        except Exception:
            continue

    return results


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


def build_caption(row: dict[str, str]) -> str:
    title = row.get("title", "").strip() or "untitled"
    return (
        f"Image context: {title}. "
        f"Course/group: {row.get('course', '')}. Week: {row.get('week', '')}. "
        f"Resource type: {row.get('resource_type', '')}."
    )


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

    if method in {"pdf_text", "pdf_text_then_ocr_fallback"}:
        text_pages, page_count = extract_pdf_pages(source_path)
        ocr_pages: dict[str, str] = {}

        # OCR for PDFs is currently disabled; keep fallback route available
        # in case methods are changed later.
        if method == "pdf_text_then_ocr_fallback":
            all_page_ids = {f"page_{idx:04d}" for idx in range(1, page_count + 1)}
            missing_page_ids = all_page_ids.difference(text_pages.keys())
            if missing_page_ids:
                ocr_pages = extract_pdf_page_ocr(source_path, missing_page_ids)

        all_page_ids = sorted(set(text_pages) | set(ocr_pages))
        if all_page_ids:
            for unit_id in all_page_ids:
                pdf_text = text_pages.get(unit_id, "")
                ocr_text = ocr_pages.get(unit_id, "")
                text = pdf_text or ocr_text
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

    if method in {"ocr", "ocr_plus_caption"} and ext in {".png", ".jpg", ".jpeg", ".webp"}:
        ocr_text = extract_image_ocr(source_path)
        caption = build_caption(row) if method == "ocr_plus_caption" else ""
        combined = normalize_text(f"{ocr_text}\n\n{caption}".strip())
        if not combined:
            # Always attach row metadata on empty OCR; `caption` is blank for method "ocr".
            combined = normalize_text(
                f"Image file context: {source_path.name}. {build_caption(row)}"
            )
        yield emit_record(row, method, "image_0001", combined, row["source_path"])
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
