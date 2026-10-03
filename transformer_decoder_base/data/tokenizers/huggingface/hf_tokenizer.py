# Purpose: Lazily adapts a caller-supplied Hugging Face tokenizer to the decoder-LM interface (encode/decode, pad/eos ids, left padding).
"""Hugging Face tokenizer adapter for causal LMs.

Construction: wrap an existing tokenizer object (``HFTokenizerAdapter(tok)``) or call
``HFTokenizerAdapter.from_pretrained(name_or_path, **kwargs)`` which imports ``transformers``
inside the classmethod. Passing a hub name may download files at CALL time; pass a local
directory and ``local_files_only=True`` to forbid network access.

Interface:
    encode(text, add_special_tokens=False) -> list[int]
    decode(ids, skip_special_tokens=True)  -> str
    batch_encode(texts, padding_side="left") -> {"input_ids": LongTensor (B, T),
                                                 "attention_mask": LongTensor (B, T), 1 = real}
    pad_id / eos_id / bos_id                -> Optional[int]
    ensure_pad_token()                      -> explicitly sets pad = eos when no pad token
                                               exists (common for GPT-style vocabularies);
                                               never done implicitly.

Left padding is the right choice for batched generation (see inference/batched_generation.py);
right padding for training collators.

Requires: transformers (pip install transformers) for ``from_pretrained``; torch for batch_encode.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch


class HFTokenizerAdapter:
    """Adapter over a ``transformers`` tokenizer (``PreTrainedTokenizer[Fast]``)."""

    def __init__(self, tokenizer: Any) -> None:
        for attr in ("encode", "decode"):
            if not hasattr(tokenizer, attr):
                raise TypeError(f"tokenizer lacks '{attr}'; expected a transformers tokenizer")
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, name_or_path: str, **kwargs: Any) -> "HFTokenizerAdapter":
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install transformers to use this adapter: pip install transformers") from exc
        return cls(AutoTokenizer.from_pretrained(name_or_path, **kwargs))

    @property
    def pad_id(self) -> Optional[int]:
        return getattr(self.tokenizer, "pad_token_id", None)

    @property
    def eos_id(self) -> Optional[int]:
        return getattr(self.tokenizer, "eos_token_id", None)

    @property
    def bos_id(self) -> Optional[int]:
        return getattr(self.tokenizer, "bos_token_id", None)

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    def ensure_pad_token(self) -> int:
        """Use EOS as pad if no pad token is defined; returns the pad id.

        Labels must then be masked via attention_mask, not by pad id (pad == eos).
        """
        if self.pad_id is None:
            if self.tokenizer.eos_token is None:
                raise ValueError("Tokenizer has neither a pad nor an eos token; add one explicitly")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return int(self.pad_id)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=add_special_tokens))

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode([int(i) for i in ids], skip_special_tokens=skip_special_tokens)

    def batch_encode(
        self,
        texts: Sequence[str],
        padding_side: str = "left",
        add_special_tokens: bool = False,
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize and pad a batch locally (does not mutate the tokenizer's padding_side)."""
        if padding_side not in ("left", "right"):
            raise ValueError("padding_side must be 'left' or 'right'")
        if self.pad_id is None:
            raise ValueError("Tokenizer has no pad token; call ensure_pad_token() explicitly first")
        rows = [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]
        if max_length is not None:
            rows = [r[-max_length:] if padding_side == "left" else r[:max_length] for r in rows]
        longest = max((len(r) for r in rows), default=0)
        input_ids = torch.full((len(rows), longest), int(self.pad_id), dtype=torch.long)
        attention_mask = torch.zeros((len(rows), longest), dtype=torch.long)
        for i, r in enumerate(rows):
            if not r:
                continue
            sl = slice(longest - len(r), longest) if padding_side == "left" else slice(0, len(r))
            input_ids[i, sl] = torch.tensor(r, dtype=torch.long)
            attention_mask[i, sl] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def apply_chat_template(self, messages: Sequence[Dict[str, str]], **kwargs: Any) -> Any:
        """Pass-through to the tokenizer's own chat template (if it defines one)."""
        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise AttributeError("This tokenizer does not provide apply_chat_template")
        return self.tokenizer.apply_chat_template(list(messages), **kwargs)
