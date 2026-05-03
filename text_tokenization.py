"""Shared lexical tokenizer for BM25 indexing and query-time retrieval."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass


DEFAULT_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "with",
    }
)

_PUNCT_TRANSLATE = str.maketrans({ch: " " for ch in string.punctuation})


@dataclass(frozen=True)
class LexicalTokenizerConfig:
    """Configuration for deterministic lexical tokenization."""

    lowercase: bool = True
    strip_punctuation: bool = True
    use_stopwords: bool = True
    use_stemming: bool = False

    def to_dict(self) -> dict[str, bool]:
        """Serialize settings for persistence in BM25 corpus rows."""
        return {
            "lowercase": self.lowercase,
            "strip_punctuation": self.strip_punctuation,
            "use_stopwords": self.use_stopwords,
            "use_stemming": self.use_stemming,
        }

    @classmethod
    def from_dict(cls, payload: dict | None, fallback: "LexicalTokenizerConfig") -> "LexicalTokenizerConfig":
        """Create config from persisted fields, defaulting to fallback values."""
        payload = payload or {}
        return cls(
            lowercase=bool(payload.get("lowercase", fallback.lowercase)),
            strip_punctuation=bool(payload.get("strip_punctuation", fallback.strip_punctuation)),
            use_stopwords=bool(payload.get("use_stopwords", fallback.use_stopwords)),
            use_stemming=bool(payload.get("use_stemming", fallback.use_stemming)),
        )


def _simple_stem(token: str) -> str:
    """Apply lightweight suffix stripping when stemming is enabled."""
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("es") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def tokenize_for_bm25(
    text: str,
    config: LexicalTokenizerConfig,
    stopwords: frozenset[str] = DEFAULT_STOPWORDS,
) -> list[str]:
    """Tokenize text with configurable normalization, stopwords, and stemming."""
    normalized = text
    if config.lowercase:
        normalized = normalized.lower()
    if config.strip_punctuation:
        normalized = normalized.translate(_PUNCT_TRANSLATE)
    tokens = re.findall(r"[a-z0-9]+", normalized)
    if config.use_stopwords:
        tokens = [tok for tok in tokens if tok not in stopwords]
    if config.use_stemming:
        tokens = [_simple_stem(tok) for tok in tokens]
    return tokens
