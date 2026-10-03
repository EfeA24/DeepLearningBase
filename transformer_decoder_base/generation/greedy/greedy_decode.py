# Purpose: Runs greedy (argmax) autoregressive decoding for decoder LMs with EOS handling, max_new_tokens and optional KV cache.
"""Greedy decoding loop.

Model callable contract (shared by generation/ and inference/ in this template):
    model_fn(input_ids, past, attention_mask) -> (logits, new_past)
        input_ids:      (B, T_new) LongTensor — full prompt on the first call, then only the
                        newest token when ``use_cache=True``; always the full sequence when
                        ``use_cache=False`` (past is then always None).
        past:           None on the first call, otherwise whatever the previous call returned.
        attention_mask: None, or (B, T_total) 1 = real token covering past + new tokens.
        logits:         (B, T_new, V) raw logits; only ``logits[:, -1]`` is used.
    ``TransformerDecoderLM.generation_step`` satisfies this contract.

Batching: prompts in one batch must have equal length unless you pass a LEFT-padded batch
with ``attention_mask`` (the model must derive positions from the mask). Right-padded prompts
are not supported (the next token would be predicted from a pad position).

EOS: once a row emits ``eos_token_id`` it is finished; later positions are filled with
``pad_token_id`` (defaults to ``eos_token_id``). The loop stops early when all rows finished.
Returns (B, T_prompt + n_generated) LongTensor, n_generated <= max_new_tokens.

Requires: torch.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import torch

ModelFn = Callable[[torch.Tensor, Any, Optional[torch.Tensor]], Tuple[torch.Tensor, Any]]


@torch.no_grad()
def greedy_decode(
    model_fn: ModelFn,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    attention_mask: Optional[torch.Tensor] = None,
    use_cache: bool = True,
) -> torch.Tensor:
    """Greedy-decode up to ``max_new_tokens`` tokens after ``input_ids`` (B, T)."""
    if input_ids.dim() != 2 or input_ids.shape[1] == 0:
        raise ValueError(f"input_ids must be a non-empty (B, T) tensor, got {tuple(input_ids.shape)}")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")
    if attention_mask is not None and attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must match input_ids shape")
    fill_id = pad_token_id if pad_token_id is not None else eos_token_id

    sequences = input_ids
    mask = attention_mask
    finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
    past: Any = None
    step_input = input_ids
    for _ in range(max_new_tokens):
        logits, new_past = model_fn(step_input, past if use_cache else None, mask)
        next_token = logits[:, -1, :].argmax(dim=-1)
        if eos_token_id is not None:
            next_token = torch.where(finished, torch.full_like(next_token, fill_id), next_token)
            finished = finished | (next_token == eos_token_id)
        sequences = torch.cat((sequences, next_token[:, None]), dim=1)
        if mask is not None:
            mask = torch.cat((mask, torch.ones_like(next_token[:, None], dtype=mask.dtype)), dim=1)
        if use_cache:
            past = new_past
            step_input = next_token[:, None]
        else:
            step_input = sequences
        if eos_token_id is not None and bool(finished.all()):
            break
    return sequences
