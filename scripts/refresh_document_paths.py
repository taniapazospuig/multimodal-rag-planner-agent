#!/usr/bin/env python3
"""
Refresh stale source_path entries in documents.csv after file renames.

This script preserves all existing metadata columns and values; it only updates
"source_path" when the referenced file no longer exists under data/kb/00_raw.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data/kb/00_raw"
DOCS_CSV = PROJECT_ROOT / "data/kb/metadata/documents.csv"


@dataclass
class MatchCandidate:
    path: Path
    score: float


def _normalize_stem(stem: str) -> str:
    text = stem.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    text = re.sub(r"-goodnotes$", "", text)
    text = re.sub(r"-2$", "", text)
    return text


def _similarity_score(old_stem: str, candidate_stem: str) -> float:
    old_norm = _normalize_stem(old_stem)
    cand_norm = _normalize_stem(candidate_stem)

    ratio = SequenceMatcher(None, old_norm, cand_norm).ratio()
    score = ratio

    if old_norm and old_norm in cand_norm:
        score += 0.40
        idx = cand_norm.find(old_norm)
        extra_prefix = cand_norm[:idx].strip("-")
        extra_suffix = cand_norm[idx + len(old_norm) :].strip("-")
        # Prefer names with less extra text around the original stem.
        extra_len = len(extra_prefix) + len(extra_suffix)
        score += max(0.0, 0.15 - min(extra_len, 30) * 0.005)
    if old_norm and cand_norm.endswith(old_norm):
        score += 0.25

    old_tokens = set(t for t in old_norm.split("-") if t)
    cand_tokens = set(t for t in cand_norm.split("-") if t)
    if old_tokens and old_tokens.issubset(cand_tokens):
        score += 0.20
    overlap = len(old_tokens & cand_tokens)
    score += 0.02 * overlap
    return score


def _resolve_replacement(old_rel_path: str) -> str | None:
    old_abs = PROJECT_ROOT / old_rel_path
    if old_abs.exists():
        return old_rel_path

    rel_under_raw = old_rel_path.replace("\\", "/")
    if not rel_under_raw.startswith("data/kb/00_raw/"):
        return None

    path_under_raw = rel_under_raw[len("data/kb/00_raw/") :]
    old_path = RAW_ROOT / path_under_raw
    parent = old_path.parent
    if not parent.exists():
        return None

    ext = old_path.suffix.lower()
    candidates = sorted(
        p for p in parent.iterdir() if p.is_file() and p.suffix.lower() == ext
    )
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0].relative_to(PROJECT_ROOT).as_posix()

    ranked: list[MatchCandidate] = []
    for cand in candidates:
        ranked.append(
            MatchCandidate(
                path=cand,
                score=_similarity_score(old_path.stem, cand.stem),
            )
        )
    ranked.sort(key=lambda x: x.score, reverse=True)

    if len(ranked) > 1 and abs(ranked[0].score - ranked[1].score) < 0.02:
        return None

    return ranked[0].path.relative_to(PROJECT_ROOT).as_posix()


def main() -> None:
    if not DOCS_CSV.exists():
        raise SystemExit(f"Missing documents.csv: {DOCS_CSV}")
    if not RAW_ROOT.exists():
        raise SystemExit(f"Missing raw root: {RAW_ROOT}")

    with DOCS_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if "source_path" not in fieldnames:
        raise SystemExit("documents.csv does not contain a source_path column.")

    updated = 0
    unresolved: list[str] = []
    for row in rows:
        current = str(row.get("source_path", "")).strip()
        replacement = _resolve_replacement(current)
        if replacement is None:
            unresolved.append(current)
            continue
        if replacement != current:
            row["source_path"] = replacement
            updated += 1

    if unresolved:
        sample = ", ".join(unresolved[:5])
        raise SystemExit(
            f"Could not resolve {len(unresolved)} source_path values. "
            f"Examples: {sample}"
        )

    with DOCS_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated source_path rows: {updated}")
    print(f"documents.csv: {DOCS_CSV}")


if __name__ == "__main__":
    main()
