# Purpose: Performs greedy next-token decoding from a supplied recurrent step function and hidden state.
"""Greedy decoding for recurrent language models.

Step contract: ``step_fn(tokens (B,) int64, state) -> (logits (B, V), new_state)`` where
``state`` may be None on the first call. ``make_step_fn(model)`` adapts a batch-first model
``model(inputs (B, 1), state) -> (logits (B, 1, V), new_state)``.

Procedure: prompt tokens (B, P), P >= 1, are fed one by one to build the state; then up to
``max_new_tokens`` tokens are chosen with argmax. With ``eos_id``, a row is finished after it
emits EOS; later positions in that row are filled with ``pad_id`` (or ``eos_id`` if pad_id is
None), and decoding stops early once every row is finished.

Returned state: it has consumed the prompt and every generated token EXCEPT the last one
(no wasted step). To continue generation, feed ``tokens[:, -1]`` first. With
``max_new_tokens=0`` nothing is fed and the input state is returned unchanged.
Runs under ``torch.no_grad()``; the caller sets ``model.eval()`` and device placement.

Requires: torch only. No sibling files.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import torch
from torch import nn

StepFn = Callable[[torch.Tensor, Any], Tuple[torch.Tensor, Any]]


@dataclass
class GreedyOutput:
    """``tokens`` (B, P + N) prompt + generated; ``new_tokens`` (B, N); ``finished`` (B,) bool."""

    tokens: torch.Tensor
    new_tokens: torch.Tensor
    finished: torch.Tensor
    state: Any


def make_step_fn(model: nn.Module) -> StepFn:
    """Adapt ``model(inputs (B, 1), state) -> (logits (B, 1, V), state)`` to the step contract."""

    def step_fn(tokens: torch.Tensor, state: Any) -> Tuple[torch.Tensor, Any]:
        logits, new_state = model(tokens.unsqueeze(1), state)
        return logits[:, -1], new_state

    return step_fn


def greedy_decode(
    step_fn: StepFn,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    state: Any = None,
    eos_id: Optional[int] = None,
    pad_id: Optional[int] = None,
) -> GreedyOutput:
    """Greedy-decode up to ``max_new_tokens`` tokens after ``prompt_ids`` (B, P)."""
    if prompt_ids.dim() != 2 or prompt_ids.shape[1] < 1:
        raise ValueError(f"prompt_ids must have shape (B, P) with P >= 1, got {tuple(prompt_ids.shape)}.")
    if prompt_ids.dtype != torch.long:
        raise TypeError(f"prompt_ids must be int64, got {prompt_ids.dtype}.")
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}.")

    batch_size = prompt_ids.shape[0]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=prompt_ids.device)
    if max_new_tokens == 0:
        return GreedyOutput(prompt_ids, prompt_ids[:, :0], finished, state)

    fill_id = pad_id if pad_id is not None else eos_id
    generated = []
    with torch.no_grad():
        logits = None
        for position in range(prompt_ids.shape[1]):
            logits, state = step_fn(prompt_ids[:, position], state)
        for step in range(max_new_tokens):
            next_tokens = logits.argmax(dim=-1)
            if eos_id is not None:
                next_tokens = torch.where(finished, torch.full_like(next_tokens, fill_id), next_tokens)
                finished = finished | (next_tokens == eos_id)
            generated.append(next_tokens)
            if step == max_new_tokens - 1 or (eos_id is not None and bool(finished.all())):
                break
            logits, state = step_fn(next_tokens, state)

    new_tokens = torch.stack(generated, dim=1)
    return GreedyOutput(torch.cat([prompt_ids, new_tokens], dim=1), new_tokens, finished, state)
