# Purpose: Generates token or value sequences autoregressively from a recurrent step function with explicit initial state, decoding callback and stop rule for LSTM/GRU models.
"""Autoregressive generation for recurrent models.

Step contract:
    ``step_fn(current_input, state) -> (output, new_state)``
        - token mode: ``current_input`` ``(B,)`` int64 ids, ``output`` ``(B, V)`` logits;
        - value mode: ``current_input`` ``(B, F)`` floats, ``output`` ``(B, F)`` predictions.
    ``select_fn(output) -> next_input`` chooses the next input (e.g. ``greedy_select``, ``make_sampling_select``,
    or identity for value forecasting).
    ``stop_fn(next_input, step) -> (B,) bool`` optional extra per-row stop rule.

Token mode with ``eos_id``: rows that emitted EOS are marked finished; later positions are filled with
``pad_id`` and their state is no longer meaningful. Generation stops early when all rows are finished.
The model must already be on the right device and in eval mode; runs under ``torch.no_grad``.

``make_recurrent_step_fn`` adapts a model ``model(x (B, 1[, F]), state) -> (out (B, 1, V|F), state)``.

Requires: torch only.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import nn

StepFn = Callable[[torch.Tensor, Any], Tuple[torch.Tensor, Any]]


def greedy_select(logits: torch.Tensor) -> torch.Tensor:
    """Argmax over the last dim: ``(B, V)`` -> ``(B,)``."""
    return logits.argmax(dim=-1)


def make_sampling_select(
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a sampler ``(B, V) logits -> (B,) ids`` with temperature and optional top-k filtering."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0; use greedy_select for deterministic decoding.")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be >= 1.")

    def select(logits: torch.Tensor) -> torch.Tensor:
        scaled = logits / temperature
        if top_k is not None and top_k < scaled.shape[-1]:
            kth = torch.topk(scaled, top_k, dim=-1).values[..., -1:]
            scaled = scaled.masked_fill(scaled < kth, float("-inf"))
        probs = torch.softmax(scaled.float(), dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)

    return select


def make_recurrent_step_fn(model: nn.Module) -> StepFn:
    """Adapt a batch-first sequence model to the single-step contract."""

    def step(current_input: torch.Tensor, state: Any) -> Tuple[torch.Tensor, Any]:
        out, new_state = model(current_input.unsqueeze(1), state)
        return out[:, -1], new_state

    return step


@torch.no_grad()
def generate(
    step_fn: StepFn,
    initial_input: torch.Tensor,
    max_steps: int,
    select_fn: Callable[[torch.Tensor], torch.Tensor] = greedy_select,
    initial_state: Any = None,
    eos_id: Optional[int] = None,
    pad_id: int = 0,
    stop_fn: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
) -> Dict[str, Any]:
    """Run up to ``max_steps`` decoding steps.

    Args:
        initial_input: First input, e.g. BOS ids ``(B,)`` or last observed values ``(B, F)``. It is
            NOT included in the returned sequences.
        initial_state: Encoder/warm-up state (e.g. from running the model over a prompt) or None.

    Returns:
        ``{"sequences": (B, S[, F]), "finished": (B,) bool, "final_state": state}`` with ``S <= max_steps``.
    """
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1.")
    bsz = initial_input.shape[0]
    finished = torch.zeros(bsz, dtype=torch.bool, device=initial_input.device)
    current, state = initial_input, initial_state
    generated = []
    for step in range(max_steps):
        output, state = step_fn(current, state)
        next_input = select_fn(output)
        if next_input.shape[0] != bsz:
            raise ValueError("select_fn must preserve the batch dimension.")
        if eos_id is not None and next_input.dim() == 1:
            next_input = torch.where(finished, torch.full_like(next_input, pad_id), next_input)
            finished = finished | (next_input == eos_id)
        if stop_fn is not None:
            finished = finished | stop_fn(next_input, step).to(torch.bool)
        generated.append(next_input)
        current = next_input
        if bool(finished.all()):
            break
    return {"sequences": torch.stack(generated, dim=1), "finished": finished, "final_state": state}
