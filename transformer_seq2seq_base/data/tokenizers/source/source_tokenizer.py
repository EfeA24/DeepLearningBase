# Purpose: Wraps a caller-supplied encode callable or vocabulary as a seq2seq source tokenizer with an explicit EOS and max_length contract.
"""Source (encoder-side) tokenizer adapter.

Contract:
    encode(text) -> list[int] = truncate(encode_fn(text), max_length - 1) + [EOS]
        (EOS only if ``add_eos``; ``encode_fn`` must NOT add special tokens itself).
    batch_encode(texts) -> {"input_ids": (B, S) int64 right-padded with pad_token_id,
                            "attention_mask": (B, S) int64, 1 = real token, 0 = pad}
No BOS is added on the source side (encoder inputs need no start token).

Requires: torch only. ``from_vocab`` uses a whitespace split (not a linguistic tokenizer).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Dict, List, Optional

import torch

EncodeFn = Callable[[str], Sequence[int]]
DecodeFn = Callable[[Sequence[int]], str]


class SourceTokenizer:
    """Source tokenizer around an injected ``encode_fn``.

    Args:
        encode_fn: text -> token ids without special tokens.
        pad_token_id: padding id for ``batch_encode``.
        eos_token_id: EOS id appended when ``add_eos``.
        max_length: maximum length INCLUDING EOS (None = no limit).
        add_eos: append EOS.
        truncation_side: "right" keeps the beginning of the text, "left" keeps the end.
        decode_fn: optional ids -> text callable for debugging/inspection.
    """

    def __init__(
        self,
        encode_fn: EncodeFn,
        *,
        pad_token_id: int,
        eos_token_id: Optional[int],
        max_length: Optional[int] = None,
        add_eos: bool = True,
        truncation_side: str = "right",
        decode_fn: Optional[DecodeFn] = None,
    ) -> None:
        if not callable(encode_fn):
            raise TypeError("encode_fn must be callable")
        if add_eos and eos_token_id is None:
            raise ValueError("add_eos=True requires eos_token_id")
        if max_length is not None and max_length < (2 if add_eos else 1):
            raise ValueError(f"max_length={max_length} too small")
        if truncation_side not in ("right", "left"):
            raise ValueError(f"truncation_side must be 'right' or 'left', got {truncation_side!r}")
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn
        self.pad_token_id = int(pad_token_id)
        self.eos_token_id = None if eos_token_id is None else int(eos_token_id)
        self.max_length = max_length
        self.add_eos = add_eos
        self.truncation_side = truncation_side

    @classmethod
    def from_vocab(
        cls,
        token_to_id: Mapping[str, int],
        *,
        unk_token_id: int,
        pad_token_id: int,
        eos_token_id: Optional[int],
        lowercase: bool = False,
        **kwargs,
    ) -> "SourceTokenizer":
        """Build a whitespace-split tokenizer from an explicit token -> id mapping."""
        vocab = dict(token_to_id)
        id_to_token = {i: t for t, i in vocab.items()}

        def encode(text: str) -> List[int]:
            if lowercase:
                text = text.lower()
            return [vocab.get(tok, unk_token_id) for tok in text.split()]

        def decode(ids: Sequence[int]) -> str:
            return " ".join(id_to_token.get(int(i), "<unk>") for i in ids)

        return cls(encode, pad_token_id=pad_token_id, eos_token_id=eos_token_id, decode_fn=decode, **kwargs)

    def encode(self, text: str) -> List[int]:
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        ids = [int(t) for t in self.encode_fn(text)]
        if self.max_length is not None:
            budget = self.max_length - (1 if self.add_eos else 0)
            if len(ids) > budget:
                ids = ids[:budget] if self.truncation_side == "right" else ids[len(ids) - budget:]
        if self.add_eos:
            ids.append(self.eos_token_id)
        return ids

    __call__ = encode

    def batch_encode(self, texts: Sequence[str]) -> Dict[str, torch.Tensor]:
        if len(texts) == 0:
            raise ValueError("texts must not be empty")
        encoded = [self.encode(t) for t in texts]
        max_len = max(len(e) for e in encoded)
        input_ids = torch.full((len(encoded), max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(encoded), max_len), dtype=torch.long)
        for row, ids in enumerate(encoded):
            if ids:
                input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                attention_mask[row, : len(ids)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, ids: Sequence[int]) -> str:
        if self.decode_fn is None:
            raise RuntimeError("no decode_fn was supplied to this SourceTokenizer")
        return self.decode_fn([int(i) for i in ids])
