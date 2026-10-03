# Purpose: Provides a simple regex word/punctuation tokenizer scaffold for word-level recurrent models (not a production linguistic tokenizer).
"""Simple word tokenizer (scaffold).

Default rule: ``\\w+|[^\\w\\s]`` -> runs of word characters, and each punctuation character
as its own token ("Hi, there!" -> ["hi", ",", "there", "!"] with lowercase=True).
With ``split_punctuation=False`` it behaves like a whitespace split.

Known limitations (adapt for real projects): no handling of contractions, URLs, numbers with
separators, hyphenation, emoji sequences or language-specific segmentation (e.g. CJK);
``detokenize`` is a lossy heuristic. For subword tokenization use a dedicated library.

``re`` is imported lazily in ``__init__``; the compiled pattern is picklable (DataLoader workers).

Requires: no third-party packages. No sibling files.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

_ATTACH_LEFT = set(".,!?;:%)]}")
_ATTACH_RIGHT = set("([{$")


class WordTokenizer:
    """Regex-based word/punctuation tokenizer with optional lowercasing."""

    def __init__(
        self,
        lowercase: bool = True,
        split_punctuation: bool = True,
        pattern: Optional[str] = None,
    ) -> None:
        import re

        if pattern is None:
            pattern = r"\w+|[^\w\s]" if split_punctuation else r"\S+"
        self.lowercase = lowercase
        self.pattern = pattern
        self._regex = re.compile(pattern)

    def tokenize(self, text: str) -> List[str]:
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}.")
        if self.lowercase:
            text = text.lower()
        return self._regex.findall(text)

    def tokenize_batch(self, texts: Iterable[str]) -> List[List[str]]:
        return [self.tokenize(text) for text in texts]

    def detokenize(self, tokens: Sequence[str]) -> str:
        """Heuristic join: no space before closing punctuation or after opening brackets."""
        pieces: List[str] = []
        for token in tokens:
            if pieces and token not in _ATTACH_LEFT and pieces[-1] not in _ATTACH_RIGHT:
                pieces.append(" ")
            pieces.append(token)
        return "".join(pieces)
