# Purpose: Implements greedy autoregressive seq2seq decoding (encode once, step-wise decode with optional KV cache) with decoder-start/EOS/max_new_tokens handling.
"""Greedy decoding for encoder-decoder models.

Model contract (e.g. models/seq2seq/transformer.py):
    model.encode(input_ids (B, S), attention_mask (B, S)) -> memory (B, S, D)
    model.decode(decoder_input_ids, memory, memory_attention_mask=..., past_key_values=...,
                 use_cache=...) -> (logits (B, T, V), present_key_values or None)
With ``use_cache=True`` only the newest token is fed after the first step.

Output: (B, 1 + n) int64 sequences starting with ``decoder_start_token_id``; after a row
emits EOS (or a stopping criterion fires) it is filled with ``pad_token_id``. n <=
max_new_tokens (loop exits early when every row is finished).
The caller is responsible for ``model.eval()`` and device placement; no grad is tracked.

Requires: torch only. Stopping criteria (optional) follow decoding/stopping/stopping_criteria.py.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

import torch


@torch.no_grad()
def greedy_decode(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    *,
    decoder_start_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    max_new_tokens: int,
    use_cache: bool = True,
    stopping_criteria: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    """Greedy (argmax) decoding.

    Args:
        model: object satisfying the encode/decode contract above.
        input_ids: (B, S) source ids.
        attention_mask: (B, S) 1 = real source token; None = no source padding.
        decoder_start_token_id / eos_token_id / pad_token_id: special ids.
        max_new_tokens: maximum generated tokens (excluding the start token).
        use_cache: reuse decoder K/V between steps.
        stopping_criteria: optional ``fn(sequences) -> (B,) bool`` (True = stop).
    """
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must be (B, S), got {tuple(input_ids.shape)}")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    memory = model.encode(input_ids, attention_mask)
    batch_size = input_ids.size(0)
    sequences = torch.full((batch_size, 1), decoder_start_token_id, dtype=torch.long, device=input_ids.device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
    past = None
    for _ in range(max_new_tokens):
        step_input = sequences[:, -1:] if (use_cache and past is not None) else sequences
        logits, present = model.decode(
            step_input, memory, memory_attention_mask=attention_mask,
            past_key_values=past, use_cache=use_cache,
        )
        past = present if use_cache else None
        next_tokens = logits[:, -1, :].argmax(dim=-1)
        next_tokens = torch.where(finished, torch.full_like(next_tokens, pad_token_id), next_tokens)
        sequences = torch.cat([sequences, next_tokens.unsqueeze(1)], dim=1)
        finished = finished | (next_tokens == eos_token_id)
        if stopping_criteria is not None:
            finished = finished | stopping_criteria(sequences).to(torch.bool)
        if bool(finished.all()):
            break
    return sequences
