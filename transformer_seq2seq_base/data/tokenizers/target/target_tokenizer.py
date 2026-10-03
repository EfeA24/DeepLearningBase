# Purpose: Wraps caller-supplied encode/decode callables as a seq2seq target tokenizer with decoder-start, EOS and skip_special_tokens decoding contracts.
"""Target (decoder-side) tokenizer adapter.

Contract:
    encode(text) -> labels: list[int] = truncate(encode_fn(text), max_length - 1) + [EOS]
        (no decoder start token; ``encode_fn`` must NOT add special tokens).
    encode_with_decoder_inputs(text) -> (decoder_input_ids, labels) where
        decoder_input_ids = [decoder_start_token_id] + labels[:-1]  (equal lengths).
    decode(ids, skip_special_tokens=True):
        drops a leading decoder start token, stops at the first EOS (``stop_at_eos``),
        removes special ids (pad/eos/start/extra) when ``skip_special_tokens``.
``decoder_start_token_id`` may equal pad (T5/Marian style), BOS, or even EOS (mBART style);
the leading-start skip handles all three.

Requires: pure Python; encode/decode callables are injected.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import List, Optional, Tuple

EncodeFn = Callable[[str], Sequence[int]]
DecodeFn = Callable[[Sequence[int]], str]


class TargetTokenizer:
    """Target tokenizer around injected ``encode_fn``/``decode_fn``.

    Args:
        encode_fn: text -> ids without special tokens.
        decode_fn: ids -> text (receives ids AFTER special-token handling).
        pad_token_id / eos_token_id / decoder_start_token_id: special ids.
        max_length: maximum label length including EOS (None = no limit).
        extra_special_token_ids: more ids removed by ``skip_special_tokens``.
    """

    def __init__(
        self,
        encode_fn: EncodeFn,
        decode_fn: DecodeFn,
        *,
        pad_token_id: int,
        eos_token_id: int,
        decoder_start_token_id: int,
        max_length: Optional[int] = None,
        extra_special_token_ids: Iterable[int] = (),
    ) -> None:
        if not callable(encode_fn) or not callable(decode_fn):
            raise TypeError("encode_fn and decode_fn must be callable")
        if max_length is not None and max_length < 1:
            raise ValueError("max_length must be >= 1 (EOS must fit)")
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn
        self.pad_token_id = int(pad_token_id)
        self.eos_token_id = int(eos_token_id)
        self.decoder_start_token_id = int(decoder_start_token_id)
        self.max_length = max_length
        self.special_token_ids = {
            self.pad_token_id,
            self.eos_token_id,
            self.decoder_start_token_id,
            *(int(i) for i in extra_special_token_ids),
        }

    @classmethod
    def from_vocab(
        cls,
        token_to_id: Mapping[str, int],
        *,
        unk_token_id: int,
        pad_token_id: int,
        eos_token_id: int,
        decoder_start_token_id: int,
        lowercase: bool = False,
        **kwargs,
    ) -> "TargetTokenizer":
        """Whitespace-split tokenizer over an explicit vocabulary (simple reference)."""
        vocab = dict(token_to_id)
        id_to_token = {i: t for t, i in vocab.items()}

        def encode(text: str) -> List[int]:
            if lowercase:
                text = text.lower()
            return [vocab.get(tok, unk_token_id) for tok in text.split()]

        def decode(ids: Sequence[int]) -> str:
            return " ".join(id_to_token.get(int(i), "<unk>") for i in ids)

        return cls(
            encode,
            decode,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            **kwargs,
        )

    def encode(self, text: str) -> List[int]:
        """Text -> labels ending with EOS."""
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        ids = [int(t) for t in self.encode_fn(text)]
        if self.max_length is not None and len(ids) > self.max_length - 1:
            ids = ids[: self.max_length - 1]
        return ids + [self.eos_token_id]

    __call__ = encode

    def encode_with_decoder_inputs(self, text: str) -> Tuple[List[int], List[int]]:
        labels = self.encode(text)
        return [self.decoder_start_token_id] + labels[:-1], labels

    def clean_ids(
        self,
        ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
        stop_at_eos: bool = True,
        strip_leading_start: bool = True,
    ) -> List[int]:
        """Apply the decode-time special-token policy and return the remaining ids."""
        out = [int(i) for i in ids]
        if strip_leading_start and out and out[0] == self.decoder_start_token_id:
            out = out[1:]
        if stop_at_eos and self.eos_token_id in out:
            out = out[: out.index(self.eos_token_id)]
        if skip_special_tokens:
            out = [i for i in out if i not in self.special_token_ids]
        return out

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = True, stop_at_eos: bool = True) -> str:
        return self.decode_fn(
            self.clean_ids(ids, skip_special_tokens=skip_special_tokens, stop_at_eos=stop_at_eos)
        )

    def batch_decode(self, batch_ids: Iterable[Sequence[int]], **kwargs) -> List[str]:
        """Decode several sequences; accepts lists or rows of a (B, T) tensor (via ``.tolist()``)."""
        if hasattr(batch_ids, "tolist"):
            batch_ids = batch_ids.tolist()
        return [self.decode(ids, **kwargs) for ids in batch_ids]
