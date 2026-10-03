# Purpose: Provides a simple regex word/punctuation tokenizer scaffold for LSTM/GRU text baselines, not a production linguistic tokenizer.
"""Simple word tokenizer.

Limitations (explicit): splits on a regex of word characters vs. single punctuation marks; no handling
of contractions, URLs, emoji sequences, CJK segmentation or language-specific rules. Use a subword
tokenizer (data/tokenizers/subword/subword_tokenizer.py) for real NLP work.

Contract:
    - ``tokenize(text) -> list[str]``; optional lowercasing is explicit.
    - ``encode`` needs a caller-supplied vocabulary object exposing ``token_to_id(str) -> int``
      (e.g. data/vocabulary/vocabulary.py); ``decode`` joins with single spaces (lossy).

Requires: stdlib ``re`` (imported lazily).
"""
from __future__ import annotations

from typing import Any, Iterable, List

DEFAULT_PATTERN = r"\w+|[^\w\s]"


class WordTokenizer:
    """Regex-based word tokenizer with explicit lowercase option."""

    def __init__(self, lowercase: bool = False, pattern: str = DEFAULT_PATTERN) -> None:
        import re

        self.lowercase = lowercase
        self.pattern = pattern
        self._regex = re.compile(pattern)

    def tokenize(self, text: str) -> List[str]:
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}.")
        if self.lowercase:
            text = text.lower()
        return self._regex.findall(text)

    def __call__(self, text: str) -> List[str]:
        return self.tokenize(text)

    def encode(self, text: str, vocab: Any) -> List[int]:
        """Tokenize and map to ids with a vocabulary exposing ``token_to_id``."""
        return [vocab.token_to_id(tok) for tok in self.tokenize(text)]

    def decode(self, ids: Iterable[int], vocab: Any) -> str:
        """Map ids back with ``vocab.id_to_token`` and join by spaces (does not restore original spacing)."""
        return " ".join(vocab.id_to_token(int(i)) for i in ids)
