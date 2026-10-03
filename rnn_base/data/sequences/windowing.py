# Purpose: Splits lists or tensors into fixed-size windows with explicit size/stride/drop_last semantics for sequence models.
"""Windowing utilities for sequences (standalone, especially reusable).

Semantics (shared by every function here):
- Full windows start at 0, stride, 2*stride, ... while ``start + size <= length``.
- ``drop_last=True``: only full windows are produced (a sequence shorter than ``size``
  yields zero windows).
- ``drop_last=False``: if the full windows do not reach the end of the sequence, one extra
  partial window starting at the next stride position is added (only if that start is
  < length). With ``stride > size`` gaps between windows are skipped by design.
- Tensor windows use ``Tensor.unfold`` and are views (no copy) unless a padded partial
  window must be materialized.

Shapes for :func:`window_tensor` with ``dim=d``: input (..., L, ...) -> output where dim d
becomes the number of windows N and a new dim d+1 of length ``size`` is inserted.
Example: (T, F) with dim=0 -> (N, size, F); (B, T) with dim=1 -> (B, N, size).

Requires: torch only. No sibling files.
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Union

import torch


def _validate(size: int, stride: int, length: int) -> None:
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}.")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}.")
    if length < 0:
        raise ValueError(f"length must be non-negative, got {length}.")


def num_full_windows(length: int, size: int, stride: int) -> int:
    """Number of complete windows of ``size`` that fit in ``length`` with ``stride``."""
    _validate(size, stride, length)
    return 0 if length < size else (length - size) // stride + 1


def window_starts(length: int, size: int, stride: int, drop_last: bool = True) -> List[int]:
    """Start indices of every window (see module docstring for partial-window rules)."""
    full = num_full_windows(length, size, stride)
    starts = [index * stride for index in range(full)]
    if not drop_last:
        covered_end = starts[-1] + size if starts else 0
        next_start = starts[-1] + stride if starts else 0
        if covered_end < length and next_start < length:
            starts.append(next_start)
    return starts


def num_windows(length: int, size: int, stride: int, drop_last: bool = True) -> int:
    """Total number of windows (full plus optional partial)."""
    return len(window_starts(length, size, stride, drop_last))


def window_list(
    sequence: Sequence[Any], size: int, stride: int, drop_last: bool = True
) -> List[List[Any]]:
    """Window a Python sequence into lists. A partial last window (drop_last=False) is shorter than ``size``."""
    return [list(sequence[s : s + size]) for s in window_starts(len(sequence), size, stride, drop_last)]


def window_tensor(
    tensor: torch.Tensor,
    size: int,
    stride: int,
    dim: int = 0,
    drop_last: bool = True,
    pad_value: Optional[Union[int, float]] = None,
) -> torch.Tensor:
    """Window ``tensor`` along ``dim``.

    A partial last window (``drop_last=False``) is right-padded to ``size`` with ``pad_value``,
    which must then be provided; combine with a mask or ignore_index if padding must not count.
    """
    if tensor.dim() == 0:
        raise ValueError("window_tensor requires a tensor with at least one dimension.")
    dim = dim % tensor.dim()
    length = tensor.shape[dim]
    starts = window_starts(length, size, stride, drop_last)
    full = num_full_windows(length, size, stride)

    pieces = []
    if full > 0:
        pieces.append(tensor.unfold(dim, size, stride).movedim(-1, dim + 1))
    if len(starts) > full:
        if pad_value is None:
            raise ValueError("drop_last=False produced a partial window; pass pad_value to pad it.")
        start = starts[-1]
        tail = tensor.narrow(dim, start, length - start)
        pad_shape = list(tail.shape)
        pad_shape[dim] = size - tail.shape[dim]
        padding = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
        pieces.append(torch.cat([tail, padding], dim=dim).unsqueeze(dim))

    if not pieces:
        empty_shape = list(tensor.shape)
        empty_shape[dim] = 0
        empty_shape.insert(dim + 1, size)
        return tensor.new_empty(empty_shape)
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=dim)


def window_pairs(
    sequence: Sequence[Any], input_size: int, target_size: int, stride: int
) -> List[tuple]:
    """Non-padded (input_window, target_window) list pairs where the target immediately follows the input.

    Only complete pairs are returned (length ``input_size + target_size`` each).
    """
    if target_size <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}.")
    total = input_size + target_size
    return [
        (list(sequence[s : s + input_size]), list(sequence[s + input_size : s + total]))
        for s in window_starts(len(sequence), total, stride, drop_last=True)
    ]
