#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/6] Extracting documents..."
python scripts/extract_documents.py

echo "[2/6] Rendering PDF pages..."
python scripts/render_pdf_pages.py

echo "[3/6] Chunking extracted documents..."
python scripts/recursive_chunk_documents.py

echo "[4/6] Indexing text chunks..."
python scripts/index_text_chunks.py

echo "[5/6] Indexing rendered PDF page images..."
python scripts/index_pdf_page_images.py

echo "[6/6] Indexing raw image folders..."
python scripts/index_raw_images.py

echo "Ingestion pipeline completed."
