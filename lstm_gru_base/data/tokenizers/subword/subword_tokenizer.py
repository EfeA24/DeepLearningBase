# Purpose: Adapts a caller-supplied Hugging Face subword tokenizer (object or explicit file path) to a plain encode/decode contract for LSTM/GRU models.
"""Lazy subword tokenizer adapter.

Contract:
    - Wraps either a ``tokenizers.Tokenizer`` or a ``transformers`` tokenizer passed by the caller.
    - ``from_file(path)`` loads a local ``tokenizer.json`` with the ``tokenizers`` package;
      ``from_pretrained(name_or_path)`` uses ``transformers.AutoTokenizer`` and may download only
      when the caller explicitly calls it. Nothing is loaded at import time.
    - ``encode(text) -> list[int]``; ``decode(ids) -> str``; ``pad_id`` exposed if the tokenizer defines one.

Requires: ``tokenizers`` or ``transformers`` (optional, imported lazily).
"""
from __future__ import annotations

from typing import Any, Iterable, List, Optional


class SubwordTokenizer:
    """Uniform encode/decode facade over Hugging Face tokenizer objects."""

    def __init__(self, tokenizer: Any) -> None:
        if tokenizer is None:
            raise ValueError("A tokenizer object must be supplied.")
        self.tokenizer = tokenizer
        self._is_fast_backend = hasattr(tokenizer, "encode") and hasattr(tokenizer, "token_to_id") and not hasattr(
            tokenizer, "convert_tokens_to_ids"
        )

    @classmethod
    def from_file(cls, path: str) -> "SubwordTokenizer":
        """Load a serialized ``tokenizers`` JSON file from an explicit local path."""
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise ImportError("Install tokenizers to load tokenizer files: pip install tokenizers") from exc
        return cls(Tokenizer.from_file(str(path)))

    @classmethod
    def from_pretrained(cls, name_or_path: str, **kwargs: Any) -> "SubwordTokenizer":
        """Load via ``transformers.AutoTokenizer`` (may download when given a hub id)."""
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install transformers to use from_pretrained: pip install transformers") from exc
        return cls(AutoTokenizer.from_pretrained(name_or_path, **kwargs))

    @property
    def vocab_size(self) -> int:
        if self._is_fast_backend:
            return int(self.tokenizer.get_vocab_size())
        return int(len(self.tokenizer))

    @property
    def pad_id(self) -> Optional[int]:
        if self._is_fast_backend:
            padding = getattr(self.tokenizer, "padding", None)
            return None if not padding else padding.get("pad_id")
        return getattr(self.tokenizer, "pad_token_id", None)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        if self._is_fast_backend:
            return list(self.tokenizer.encode(text, add_special_tokens=add_special_tokens).ids)
        return list(self.tokenizer.encode(text, add_special_tokens=add_special_tokens))

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        id_list = [int(i) for i in ids]
        return self.tokenizer.decode(id_list, skip_special_tokens=skip_special_tokens)

    def encode_batch(self, texts: List[str], add_special_tokens: bool = True) -> List[List[int]]:
        return [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]
