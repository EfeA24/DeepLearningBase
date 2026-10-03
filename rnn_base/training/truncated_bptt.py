# Purpose: Implements truncated backpropagation through time: batchifies long streams, splits them into k-step chunks and trains chunk by chunk.
"""Truncated BPTT utilities (standalone, especially reusable).

Data layout:
- ``batchify_stream(stream (N, *F), B)`` -> (B, N // B, *F): row b is the b-th contiguous
  segment of the stream (remainder dropped), so the hidden state of row b can be carried
  from one chunk to the next.
- ``iter_chunks(inputs (B, T, ...), targets (B, T, ...), k)`` yields consecutive time chunks
  of k steps (the last may be shorter).

Detach boundary: after each chunk's backward/step the returned state is DETACHED, so
gradients flow through at most k steps (k1 = k2 = k), while the forward state still carries
information across the whole stream.

Model contract: ``model(inputs_chunk, state) -> (outputs, new_state)``; ``state`` may be None
(model initializes zeros) or any tensor/tuple/list tree. ``loss_fn(outputs, targets_chunk)``
returns a mean scalar loss. The caller sets ``model.train()`` and owns the optimizer.

Requires: torch only. No sibling files.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional, Tuple

import torch
from torch import nn


def detach_state(state: Any) -> Any:
    """Detach every tensor of a nested recurrent state (None/tensor/tuple/list)."""
    if state is None:
        return None
    if isinstance(state, torch.Tensor):
        return state.detach()
    if isinstance(state, (tuple, list)):
        return type(state)(detach_state(item) for item in state)
    raise TypeError(f"Unsupported state element type: {type(state).__name__}.")


def batchify_stream(stream: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Reshape a 1-D (N,) or (N, *F) stream into (B, N // B, *F) contiguous per-row segments."""
    if stream.dim() == 0:
        raise ValueError("stream must have at least one dimension (time first).")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    steps = stream.shape[0] // batch_size
    if steps == 0:
        raise ValueError(f"Stream of length {stream.shape[0]} is shorter than batch_size={batch_size}.")
    return stream[: steps * batch_size].reshape(batch_size, steps, *stream.shape[1:])


def num_chunks(total_steps: int, chunk_len: int) -> int:
    if chunk_len <= 0:
        raise ValueError(f"chunk_len must be positive, got {chunk_len}.")
    return math.ceil(total_steps / chunk_len)


def iter_chunks(
    inputs: torch.Tensor, targets: torch.Tensor, chunk_len: int, time_dim: int = 1
) -> Iterator[Tuple[int, torch.Tensor, torch.Tensor]]:
    """Yield ``(start, inputs_chunk, targets_chunk)`` views of up to ``chunk_len`` steps along ``time_dim``."""
    total = inputs.shape[time_dim]
    if targets.shape[time_dim] != total:
        raise ValueError("inputs and targets must have the same length along time_dim.")
    for index in range(num_chunks(total, chunk_len)):
        start = index * chunk_len
        length = min(chunk_len, total - start)
        yield start, inputs.narrow(time_dim, start, length), targets.narrow(time_dim, start, length)


def tbptt_chunk_step(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    state: Any,
    loss_fn: Callable[[Any, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    max_grad_norm: Optional[float] = None,
) -> Tuple[torch.Tensor, Any]:
    """Forward/backward/step on one chunk; returns (detached loss, detached new state)."""
    optimizer.zero_grad(set_to_none=True)
    outputs, new_state = model(inputs, state)
    loss = loss_fn(outputs, targets)
    loss.backward()
    if max_grad_norm is not None:
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return loss.detach(), detach_state(new_state)


@dataclass
class TBPTTResult:
    """``mean_loss``: per-chunk mean losses weighted by chunk length (time steps)."""

    mean_loss: float
    num_chunks: int
    final_state: Any


def run_truncated_bptt(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    chunk_len: int,
    loss_fn: Callable[[Any, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    initial_state: Any = None,
    max_grad_norm: Optional[float] = None,
    time_dim: int = 1,
) -> TBPTTResult:
    """Train over a batchified stream chunk by chunk, carrying detached state between chunks.

    Args:
        inputs / targets: (B, T, ...) tensors already on the model's device.
        chunk_len: Truncation length k.
        initial_state: Starting state (None lets the model initialize zeros).
    """
    state = detach_state(initial_state)
    weighted_loss = 0.0
    total_steps = 0
    count = 0
    for _, x_chunk, y_chunk in iter_chunks(inputs, targets, chunk_len, time_dim):
        loss, state = tbptt_chunk_step(model, x_chunk, y_chunk, state, loss_fn, optimizer, max_grad_norm)
        steps = x_chunk.shape[time_dim]
        weighted_loss += loss.item() * steps
        total_steps += steps
        count += 1
    mean_loss = weighted_loss / total_steps if total_steps else math.nan
    return TBPTTResult(mean_loss=mean_loss, num_chunks=count, final_state=state)
