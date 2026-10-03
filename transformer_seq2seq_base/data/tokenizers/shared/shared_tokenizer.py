# Purpose: Provides a shared-vocabulary seq2seq tokenizer adapter (one id space for source and target) and documents when embedding sharing/tying is valid.
"""Shared source/target tokenizer adapter.

One ``encode_fn``/``decode_fn`` and one id space serve both sides:
    encode_source(text) -> ids + [EOS]                (no start token)
    encode_target(text) -> labels = ids + [EOS]       (no start token)
    decode(ids)         -> text, skipping a leading decoder start, stopping at EOS and
                           removing special ids when ``skip_special_tokens``.

When sharing is valid: source and target use the SAME vocabulary and ids (e.g. a joint
BPE/SentencePiece model trained on both languages). Only then may the model share
source/target embeddings and tie the output projection to them
(``Seq2SeqTransformerConfig(share_embeddings=True, tie_output_projection=True)``).
With separate vocabularies use SourceTokenizer + TargetTokenizer and separate embeddings.

Requires: torch only (for ``batch_encode_source``).
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Dict, List, Optional

import torch

EncodeFn = Callable[[str], Sequence[int]]
DecodeFn = Callable[[Sequence[int]], str]


class SharedTokenizer:
    """Single-vocabulary tokenizer for both encoder and decoder.

    Args:
        encode_fn / decode_fn: text <-> ids without special tokens.
        vocab_size: size of the shared id space (validated against models).
        pad_token_id / eos_token_id / decoder_start_token_id: special ids.
        max_source_length / max_target_length: limits including EOS (None = no limit).
        extra_special_token_ids: more ids removed when skipping special tokens.
    """

    def __init__(
        self,
        encode_fn: EncodeFn,
        decode_fn: DecodeFn,
        *,
        vocab_size: int,
        pad_token_id: int,
        eos_token_id: int,
        decoder_start_token_id: int,
        max_source_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        extra_special_token_ids: Iterable[int] = (),
    ) -> None:
        if not callable(encode_fn) or not callable(decode_fn):
            raise TypeError("encode_fn and decode_fn must be callable")
        if vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        for name, value in (("pad", pad_token_id), ("eos", eos_token_id), ("start", decoder_start_token_id)):
            if not 0 <= int(value) < vocab_size:
                raise ValueError(f"{name} token id {value} outside [0, {vocab_size})")
        for name, value in (("max_source_length", max_source_length), ("max_target_length", max_target_length)):
            if value is not None and value < 1:
                raise ValueError(f"{name} must be >= 1")
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn
        self.vocab_size = int(vocab_size)
        self.pad_token_id = int(pad_token_id)
        self.eos_token_id = int(eos_token_id)
        self.decoder_start_token_id = int(decoder_start_token_id)
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
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
        **kwargs,
    ) -> "SharedTokenizer":
        """Whitespace-split shared tokenizer over an explicit vocabulary (simple reference)."""
        vocab = dict(token_to_id)
        id_to_token = {i: t for t, i in vocab.items()}

        def encode(text: str) -> List[int]:
            return [vocab.get(tok, unk_token_id) for tok in text.split()]

        def decode(ids: Sequence[int]) -> str:
            return " ".join(id_to_token.get(int(i), "<unk>") for i in ids)

        return cls(
            encode,
            decode,
            vocab_size=max(vocab.values()) + 1,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            **kwargs,
        )

    def _encode(self, text: str, max_length: Optional[int]) -> List[int]:
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        ids = [int(t) for t in self.encode_fn(text)]
        if max_length is not None and len(ids) > max_length - 1:
            ids = ids[: max_length - 1]
        return ids + [self.eos_token_id]

    def encode_source(self, text: str) -> List[int]:
        return self._encode(text, self.max_source_length)

    def encode_target(self, text: str) -> List[int]:
        """Labels ending with EOS (decoder inputs are built by the collator)."""
        return self._encode(text, self.max_target_length)

    def batch_encode_source(self, texts: Sequence[str]) -> Dict[str, torch.Tensor]:
        """(B, S) right-padded input_ids and attention_mask (1 = real token)."""
        if len(texts) == 0:
            raise ValueError("texts must not be empty")
        encoded = [self.encode_source(t) for t in texts]
        max_len = max(len(e) for e in encoded)
        input_ids = torch.full((len(encoded), max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(encoded), max_len), dtype=torch.long)
        for row, ids in enumerate(encoded):
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row, : len(ids)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = True, stop_at_eos: bool = True) -> str:
        out = [int(i) for i in ids]
        if out and out[0] == self.decoder_start_token_id:
            out = out[1:]
        if stop_at_eos and self.eos_token_id in out:
            out = out[: out.index(self.eos_token_id)]
        if skip_special_tokens:
            out = [i for i in out if i not in self.special_token_ids]
        return self.decode_fn(out)

    def validate_model_vocab(self, src_vocab_size: int, tgt_vocab_size: int) -> None:
        """Raise if a model's vocab sizes do not match this shared id space."""
        if src_vocab_size != self.vocab_size or tgt_vocab_size != self.vocab_size:
            raise ValueError(
                f"shared tokenizer vocab_size={self.vocab_size} but model has "
                f"src={src_vocab_size}, tgt={tgt_vocab_size}"
            )
