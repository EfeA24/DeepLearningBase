# Purpose: Lazily wraps caller-supplied pretrained Hugging Face tokenizer(s) for seq2seq source/target encoding (text_target aware) without importing transformers at module load.
"""Hugging Face seq2seq tokenizer adapter.

The adapter holds an already-constructed tokenizer (any object with the HF tokenizer call
API) and optionally a separate target tokenizer. ``transformers`` is imported ONLY inside
``from_pretrained``; constructing the adapter from an existing object never imports it.

Contracts:
    encode_source(texts) -> {"input_ids", "attention_mask"} (lists, or (B, S) tensors with
        ``return_tensors="pt"``); attention_mask 1 = real token, 0 = pad. Special tokens
        (e.g. EOS) are added by the HF tokenizer itself.
    encode_target(texts) -> list[list[int]] labels (unpadded, HF special tokens included,
        typically ending with EOS). Uses ``tokenizer(text_target=...)`` when no separate
        target tokenizer is given, so language-specific target processing is applied.
    decode / batch_decode -> text with ``skip_special_tokens`` passed through.
``decoder_start_token_id`` usually lives in the MODEL config, not the tokenizer; pass it
explicitly to the collator / decoding functions.

Requires: transformers (only for ``from_pretrained``); tokenizer objects are caller-supplied.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, List, Optional


class HFSeq2SeqTokenizer:
    """Adapter around HF tokenizer object(s).

    Args:
        tokenizer: HF tokenizer used for sources (and targets if ``target_tokenizer`` is None).
        target_tokenizer: optional separate tokenizer for targets.
    """

    def __init__(self, tokenizer: Any, target_tokenizer: Optional[Any] = None) -> None:
        if not callable(tokenizer):
            raise TypeError("tokenizer must be a callable HF tokenizer object")
        if target_tokenizer is not None and not callable(target_tokenizer):
            raise TypeError("target_tokenizer must be a callable HF tokenizer object")
        self.tokenizer = tokenizer
        self.target_tokenizer = target_tokenizer

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str,
        *,
        target_name_or_path: Optional[str] = None,
        **kwargs: Any,
    ) -> "HFSeq2SeqTokenizer":
        """Load tokenizer(s) with ``AutoTokenizer.from_pretrained``.

        ``name_or_path`` may be a local directory; whether anything is downloaded is
        decided by transformers and the caller's kwargs (e.g. ``local_files_only=True``).
        """
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Install transformers to use HFSeq2SeqTokenizer.from_pretrained: pip install transformers"
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
        target = None
        if target_name_or_path is not None:
            target = AutoTokenizer.from_pretrained(target_name_or_path, **kwargs)
        return cls(tokenizer, target)

    @property
    def pad_token_id(self) -> Optional[int]:
        return getattr(self.tokenizer, "pad_token_id", None)

    @property
    def eos_token_id(self) -> Optional[int]:
        return getattr(self.tokenizer, "eos_token_id", None)

    def encode_source(
        self,
        texts: Sequence[str],
        *,
        max_length: Optional[int] = None,
        truncation: bool = True,
        padding: Any = False,
        return_tensors: Optional[str] = None,
    ) -> Dict[str, Any]:
        if isinstance(texts, str):
            raise TypeError("texts must be a sequence of strings, not a single string")
        encoded = self.tokenizer(
            list(texts),
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            return_tensors=return_tensors,
        )
        return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}

    def encode_target(
        self,
        texts: Sequence[str],
        *,
        max_length: Optional[int] = None,
        truncation: bool = True,
    ) -> List[List[int]]:
        if isinstance(texts, str):
            raise TypeError("texts must be a sequence of strings, not a single string")
        texts = list(texts)
        if self.target_tokenizer is not None:
            encoded = self.target_tokenizer(texts, max_length=max_length, truncation=truncation)
        else:
            try:
                encoded = self.tokenizer(text_target=texts, max_length=max_length, truncation=truncation)
            except TypeError as exc:
                raise TypeError(
                    "This tokenizer does not support text_target=; pass a separate target_tokenizer"
                ) from exc
        return [list(ids) for ids in encoded["input_ids"]]

    def encode_pairs(
        self,
        sources: Sequence[str],
        targets: Sequence[str],
        *,
        max_source_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
    ) -> List[Dict[str, List[int]]]:
        """Unpadded examples {"input_ids", "labels"} for data/collators/seq2seq_collator.py."""
        if len(sources) != len(targets):
            raise ValueError(f"got {len(sources)} sources but {len(targets)} targets")
        src = self.encode_source(sources, max_length=max_source_length)["input_ids"]
        tgt = self.encode_target(targets, max_length=max_target_length)
        return [{"input_ids": list(s), "labels": t} for s, t in zip(src, tgt)]

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        tok = self.target_tokenizer or self.tokenizer
        return tok.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def batch_decode(self, batch_ids: Any, *, skip_special_tokens: bool = True) -> List[str]:
        tok = self.target_tokenizer or self.tokenizer
        if hasattr(batch_ids, "tolist"):
            batch_ids = batch_ids.tolist()
        return tok.batch_decode(batch_ids, skip_special_tokens=skip_special_tokens)
