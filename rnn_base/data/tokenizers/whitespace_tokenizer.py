# Purpose: Provides a minimal deterministic whitespace split/join tokenizer for simple word-level sequence experiments.
"""Whitespace tokenizer.

- ``tokenize`` splits on runs of whitespace (``str.split()`` semantics, no empty tokens);
  optional lowercasing.
- ``detokenize`` joins with a single space, so original spacing/newlines are not recovered
  (lossy by design). Punctuation stays attached to words ("end." is one token).
- Output tokens are strings; map them to ids with data/vocabulary/vocabulary.py.

Requires: no third-party packages. No sibling files.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence


class WhitespaceTokenizer:
    """Deterministic whitespace split/join."""

    def __init__(self, lowercase: bool = False) -> None:
        self.lowercase = lowercase

    def tokenize(self, text: str) -> List[str]:
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}.")
        if self.lowercase:
            text = text.lower()
        return text.split()

    def tokenize_batch(self, texts: Iterable[str]) -> List[List[str]]:
        return [self.tokenize(text) for text in texts]

    def detokenize(self, tokens: Sequence[str]) -> str:
        return " ".join(tokens)
