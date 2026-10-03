# Purpose: Lazy adapter around a caller-supplied Hugging Face tokenizer exposing encoder-oriented batch encoding, word-id alignment and special-token metadata.
"""Hugging Face tokenizer adapter (optional dependency).

Construction:
    HFTokenizerAdapter(tokenizer)                 # wrap an already created tokenizer object
    HFTokenizerAdapter.from_pretrained(name_or_path, local_files_only=True)
        ``transformers`` is imported only here. Downloading happens only if the caller passes a hub
        id with local_files_only=False - never at import.

Outputs of ``encode_batch`` / ``encode_words``: dict of torch tensors (B, T):
    input_ids (long), attention_mask (long, 1 = valid token, 0 = pad), optional token_type_ids,
    special_tokens_mask when requested; ``encode_words`` also returns "word_ids": list (per example)
    of lists with the source-word index per token or None for special/pad tokens (fast tokenizers only).
Label alignment is NOT done here (see tasks/ner/ner_alignment.py).

Requires: transformers (pip install transformers), imported lazily; torch.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

__all__ = ["HFTokenizerAdapter"]


class HFTokenizerAdapter:
    """Thin, explicit wrapper; keeps the underlying tokenizer accessible as ``.tokenizer``."""

    def __init__(self, tokenizer: Any) -> None:
        if not callable(tokenizer):
            raise TypeError("tokenizer must be a callable Hugging Face tokenizer object")
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, name_or_path: str, local_files_only: bool = True, **kwargs: Any) -> "HFTokenizerAdapter":
        """Load via AutoTokenizer. ``local_files_only=True`` by default to prevent surprise downloads."""
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install transformers to use this adapter: pip install transformers") from exc
        return cls(AutoTokenizer.from_pretrained(name_or_path, local_files_only=local_files_only, **kwargs))

    @property
    def is_fast(self) -> bool:
        return bool(getattr(self.tokenizer, "is_fast", False))

    @property
    def pad_token_id(self) -> Optional[int]:
        return self.tokenizer.pad_token_id

    @property
    def mask_token_id(self) -> Optional[int]:
        return self.tokenizer.mask_token_id

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    @property
    def special_token_ids(self) -> List[int]:
        return list(self.tokenizer.all_special_ids)

    def encode_batch(
        self,
        texts: Sequence[str],
        text_pairs: Optional[Sequence[str]] = None,
        max_length: Optional[int] = None,
        truncation: bool = True,
        padding: str = "longest",
        return_special_tokens_mask: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize raw texts (optionally pairs) into padded tensors."""
        if text_pairs is not None and len(text_pairs) != len(texts):
            raise ValueError("texts and text_pairs must have the same length")
        encoded = self.tokenizer(
            list(texts),
            list(text_pairs) if text_pairs is not None else None,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            return_special_tokens_mask=return_special_tokens_mask,
            return_tensors="pt",
        )
        return dict(encoded)

    def encode_words(
        self,
        words_batch: Sequence[Sequence[str]],
        max_length: Optional[int] = None,
        truncation: bool = True,
        padding: str = "longest",
    ) -> Dict[str, Any]:
        """Tokenize pre-split words (NER/POS) and return tensors plus per-example ``word_ids``."""
        if not self.is_fast:
            raise ValueError("word_ids require a fast (Rust-backed) tokenizer")
        encoded = self.tokenizer(
            [list(words) for words in words_batch],
            is_split_into_words=True,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            return_tensors="pt",
        )
        word_ids = [encoded.word_ids(batch_index=i) for i in range(len(words_batch))]
        result: Dict[str, Any] = dict(encoded)
        result["word_ids"] = word_ids
        return result

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def batch_decode(self, ids_batch: Sequence[Sequence[int]], skip_special_tokens: bool = True) -> List[str]:
        return self.tokenizer.batch_decode([list(ids) for ids in ids_batch], skip_special_tokens=skip_special_tokens)
