# Purpose: Provides the generic text-to-text inference entry point that runs a supplied decoding strategy and returns a structured result with token ids and optional text.
"""Generic seq2seq generation entry point.

Pipeline (each stage supplied by the caller, kept separate):
    1. preprocessing: ``encode_batch(texts) -> {"input_ids" (B, S), "attention_mask" (B, S)}``
       or already prepared ``input_ids``/``attention_mask`` tensors (1 = real token).
    2. decoding:      ``decoding_fn(model, input_ids, attention_mask, max_new_tokens=n)``
       returning (N, L) int64 sequences that start with ``num_prefix_tokens`` decoder start
       token(s), or an object with ``.sequences`` (and optional ``.sequence_scores``), e.g.
       ``functools.partial(greedy_decode, decoder_start_token_id=..., eos_token_id=..., pad_token_id=...)``.
       N = B * num_return_sequences for beam search.
    3. postprocessing: strip prefix, cut at the first EOS (EOS dropped unless ``keep_eos``),
       remove pad ids, then optional ``decode_fn(ids) -> str``.
Inputs are moved to the explicit ``device``; the model is NOT moved and its mode is not
changed (call ``model.eval()`` yourself).

Requires: torch only.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, List, Optional, Union

import torch


@dataclass
class Seq2SeqGenerationResult:
    """Structured generation output (row i of ``token_ids``/``texts`` matches row i of ``sequences``)."""

    token_ids: List[List[int]]
    texts: Optional[List[str]] = None
    sequences: Optional[torch.Tensor] = None
    scores: Optional[torch.Tensor] = None


def clean_generated_ids(
    sequences: torch.Tensor,
    *,
    eos_token_id: int,
    pad_token_id: int,
    num_prefix_tokens: int = 1,
    keep_eos: bool = False,
) -> List[List[int]]:
    """(N, L) decoder sequences -> list of generated ids without prefix/EOS-tail/pads."""
    if sequences.dim() != 2:
        raise ValueError(f"sequences must be (N, L), got {tuple(sequences.shape)}")
    if num_prefix_tokens < 0:
        raise ValueError("num_prefix_tokens must be >= 0")
    cleaned: List[List[int]] = []
    for row in sequences.tolist():
        ids = row[num_prefix_tokens:]
        if eos_token_id in ids:
            cut = ids.index(eos_token_id)
            ids = ids[: cut + 1] if keep_eos else ids[:cut]
        cleaned.append([i for i in ids if i != pad_token_id or i == eos_token_id])
    return cleaned


def generate(
    model: Any,
    decoding_fn: Callable[..., Any],
    *,
    device: Union[str, torch.device],
    max_new_tokens: int,
    eos_token_id: int,
    pad_token_id: int,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    texts: Optional[Sequence[str]] = None,
    encode_batch: Optional[Callable[[List[str]], Mapping[str, torch.Tensor]]] = None,
    decode_fn: Optional[Callable[[List[int]], str]] = None,
    num_prefix_tokens: int = 1,
    keep_eos: bool = False,
) -> Seq2SeqGenerationResult:
    """Run preprocessing (optional), decoding and postprocessing; see module docstring."""
    if (input_ids is None) == (texts is None):
        raise ValueError("pass exactly one of input_ids or texts")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")
    if texts is not None:
        if isinstance(texts, str):
            raise TypeError("texts must be a sequence of strings, not a single string")
        if encode_batch is None:
            raise ValueError("texts require encode_batch")
        encoded = encode_batch(list(texts))
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must be (B, S), got {tuple(input_ids.shape)}")
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    output = decoding_fn(model, input_ids, attention_mask, max_new_tokens=max_new_tokens)
    sequences = getattr(output, "sequences", output)
    if not isinstance(sequences, torch.Tensor):
        raise TypeError("decoding_fn must return a tensor or an object with a .sequences tensor")
    scores = getattr(output, "sequence_scores", None)
    token_ids = clean_generated_ids(
        sequences, eos_token_id=eos_token_id, pad_token_id=pad_token_id,
        num_prefix_tokens=num_prefix_tokens, keep_eos=keep_eos,
    )
    decoded = [decode_fn(ids) for ids in token_ids] if decode_fn is not None else None
    return Seq2SeqGenerationResult(token_ids=token_ids, texts=decoded, sequences=sequences, scores=scores)
