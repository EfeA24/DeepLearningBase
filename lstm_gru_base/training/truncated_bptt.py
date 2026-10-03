# Purpose: Implements truncated backpropagation through time over long (B, N) streams with explicit state carry and detach policy for LSTM/GRU language and sequence models.
"""Truncated BPTT (k1 = k2 = ``chunk_len``).

Policy:
    - A long batch of parallel streams ``inputs`` ``(B, N[, F])`` / ``targets`` ``(B, N[, ...])`` is split along
      time into consecutive chunks of ``chunk_len`` steps (last chunk may be shorter).
    - For each chunk: forward with the carried state, loss, backward, optional clipping, optimizer step.
    - The returned state is DETACHED before the next chunk: values carry over (the model "remembers"),
      but gradients stop at the chunk boundary, bounding memory to ``chunk_len`` steps.
    - Row ``b`` must be a contiguous continuation across chunks (see data/datasets/language_dataset.py
      ``batchify_stream``); shuffling rows breaks the carried state's meaning.

Model contract: ``model(x_chunk, state) -> (predictions, new_state)`` where state is None, a tensor, or a
tuple of tensors (LSTM). ``loss_fn(predictions, targets_chunk) -> scalar mean``.

Requires: torch only.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import torch
from torch import nn


def detach_state(state: Any) -> Any:
    """Detach a nested recurrent state (None / tensor / tuple / list)."""
    if state is None:
        return None
    if isinstance(state, torch.Tensor):
        return state.detach()
    if isinstance(state, (tuple, list)):
        return type(state)(detach_state(s) for s in state)
    raise TypeError(f"Unsupported state type {type(state).__name__}.")


def iter_chunks(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    chunk_len: int,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Yield ``(inputs[:, s:s+k], targets[:, s:s+k])`` along the time axis (dim 1)."""
    if chunk_len < 1:
        raise ValueError(f"chunk_len must be >= 1, got {chunk_len}.")
    if inputs.shape[:2] != targets.shape[:2]:
        raise ValueError(f"inputs {tuple(inputs.shape[:2])} and targets {tuple(targets.shape[:2])} must share (B, N).")
    for start in range(0, inputs.shape[1], chunk_len):
        yield inputs[:, start : start + chunk_len], targets[:, start : start + chunk_len]


def train_tbptt(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    chunk_len: int,
    initial_state: Any = None,
    clip_grad_norm: Optional[float] = None,
) -> Dict[str, Any]:
    """Run TBPTT over one batch of streams already on the model's device.

    Returns ``{"loss": time-step-weighted mean loss, "num_chunks": int, "final_state": detached state}``.
    The loss is weighted by chunk length so a short trailing chunk does not count as a full one.
    """
    model.train()
    state = detach_state(initial_state)
    total_loss, total_steps, num_chunks = 0.0, 0, 0
    for x_chunk, y_chunk in iter_chunks(inputs, targets, chunk_len):
        optimizer.zero_grad(set_to_none=True)
        predictions, state = model(x_chunk, state)
        loss = loss_fn(predictions, y_chunk)
        loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        state = detach_state(state)
        steps = x_chunk.shape[1]
        total_loss += float(loss.detach()) * steps
        total_steps += steps
        num_chunks += 1
    return {"loss": total_loss / max(total_steps, 1), "num_chunks": num_chunks, "final_state": state}


@torch.no_grad()
def evaluate_tbptt(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    chunk_len: int,
    initial_state: Any = None,
) -> Dict[str, Any]:
    """Evaluate a stream chunk-by-chunk with carried state (no parameter updates)."""
    model.eval()
    state = initial_state
    total_loss, total_steps = 0.0, 0
    for x_chunk, y_chunk in iter_chunks(inputs, targets, chunk_len):
        predictions, state = model(x_chunk, state)
        steps = x_chunk.shape[1]
        total_loss += float(loss_fn(predictions, y_chunk)) * steps
        total_steps += steps
    return {"loss": total_loss / max(total_steps, 1), "final_state": state}
