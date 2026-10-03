# Purpose: Offers small shape, shift and batch-first/time-major helpers for recurrent sequence tensors (no task policy).
"""Sequence tensor helpers.

Layouts: batch-first (B, T, *F) is the template default; time-major is (T, B, *F).
- ``to_time_major`` / ``to_batch_first``: swap dims 0 and 1 (views).
- ``add_batch_dim``: (T, *F) -> (1, T, *F) for single-sequence inference.
- ``shift_sequence``: shift along ``dim`` by ``shift`` steps (positive = towards later
  positions), filling vacated positions with ``fill_value``; length is preserved.
- ``reverse_padded``: reverse only the valid (right-padded) prefix of each row, padding stays
  at the end (useful for hand-written backward-direction RNNs).
- ``flatten_time`` / ``unflatten_time``: (B, T, *F) <-> (B*T, *F) for per-step layers/losses.

Requires: torch only. No sibling files.
"""
from __future__ import annotations

from typing import Union

import torch


def to_time_major(x: torch.Tensor) -> torch.Tensor:
    """(B, T, ...) -> (T, B, ...)."""
    if x.dim() < 2:
        raise ValueError("Expected at least 2 dims (B, T, ...).")
    return x.transpose(0, 1)


def to_batch_first(x: torch.Tensor) -> torch.Tensor:
    """(T, B, ...) -> (B, T, ...)."""
    if x.dim() < 2:
        raise ValueError("Expected at least 2 dims (T, B, ...).")
    return x.transpose(0, 1)


def add_batch_dim(x: torch.Tensor) -> torch.Tensor:
    """(T, *F) -> (1, T, *F)."""
    return x.unsqueeze(0)


def shift_sequence(x: torch.Tensor, shift: int, dim: int = 1, fill_value: Union[int, float] = 0) -> torch.Tensor:
    """Shift ``x`` along ``dim`` by ``shift`` steps, preserving length."""
    length = x.shape[dim]
    if shift == 0:
        return x.clone()
    if abs(shift) >= length:
        return torch.full_like(x, fill_value)
    fill_shape = list(x.shape)
    fill_shape[dim] = abs(shift)
    fill = torch.full(fill_shape, fill_value, dtype=x.dtype, device=x.device)
    if shift > 0:
        return torch.cat([fill, x.narrow(dim, 0, length - shift)], dim=dim)
    return torch.cat([x.narrow(dim, -shift, length + shift), fill], dim=dim)


def reverse_padded(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Reverse the first ``lengths[b]`` steps of each row of right-padded (B, T, ...) ``x``."""
    if x.dim() < 2 or lengths.shape != (x.shape[0],):
        raise ValueError("Expected x (B, T, ...) and lengths (B,).")
    steps = x.shape[1]
    positions = torch.arange(steps, device=x.device).unsqueeze(0)
    lengths = lengths.to(x.device).unsqueeze(1)
    source = torch.where(positions < lengths, lengths - 1 - positions, positions)
    index = source.view(*source.shape, *([1] * (x.dim() - 2))).expand_as(x)
    return x.gather(1, index)


def flatten_time(x: torch.Tensor) -> torch.Tensor:
    """(B, T, *F) -> (B*T, *F)."""
    if x.dim() < 2:
        raise ValueError("Expected at least 2 dims (B, T, ...).")
    return x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])


def unflatten_time(x: torch.Tensor, batch_size: int) -> torch.Tensor:
    """(B*T, *F) -> (B, T, *F) given ``batch_size``."""
    if x.shape[0] % batch_size != 0:
        raise ValueError(f"First dim {x.shape[0]} is not divisible by batch_size {batch_size}.")
    return x.reshape(batch_size, x.shape[0] // batch_size, *x.shape[1:])
