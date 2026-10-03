# Purpose: Builds shifted input/target sequences for next-step (language-model or forecasting) prediction with recurrent models.
"""Shifting utilities for next-step prediction (standalone, especially reusable).

Next-step contract (time along ``dim``, default last dim for token ids (B, T) or (T,)):
- ``inputs  = x[..., :-1]`` and ``targets = x[..., 1:]`` -> both have length T - 1, so
  ``targets[t]`` is the item that follows ``inputs[t]``.
- For numeric series (B, T, F) pass ``dim=1``.
- LM helper: targets whose position is padding become ``ignore_index`` (default -100);
  inputs keep the pad id (pair it with ``nn.Embedding(padding_idx=pad_id)``). Padding is
  identified either from a validity mask (True = valid, shape like ``token_ids``) or a pad id.
  The matching input mask is ``mask[..., :-1]``.

Requires: torch only. No sibling files.
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

import torch


def shift_list(sequence: Sequence[Any]) -> Tuple[List[Any], List[Any]]:
    """Return ``(sequence[:-1], sequence[1:])`` as lists; requires length >= 2."""
    if len(sequence) < 2:
        raise ValueError("Next-step shifting needs a sequence of length >= 2.")
    return list(sequence[:-1]), list(sequence[1:])


def shift_inputs_targets(x: torch.Tensor, dim: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return views ``(x[:-1], x[1:])`` along ``dim``; requires length >= 2 along ``dim``."""
    if x.dim() == 0:
        raise ValueError("shift_inputs_targets requires a tensor with a time dimension.")
    length = x.shape[dim]
    if length < 2:
        raise ValueError(f"Need length >= 2 along dim {dim}, got {length}.")
    return x.narrow(dim, 0, length - 1), x.narrow(dim, 1, length - 1)


def make_lm_inputs_targets(
    token_ids: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    pad_id: Optional[int] = None,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build LM ``(inputs, targets)`` from integer ids (B, T) or (T,) along the last dim.

    Args:
        token_ids: Integer token ids.
        mask: Optional bool validity mask (True = real token), same shape as ``token_ids``.
            Takes precedence over ``pad_id``.
        pad_id: Optional pad id used to detect padded targets when ``mask`` is None.
        ignore_index: Label written at padded target positions.
    """
    if token_ids.dtype.is_floating_point or token_ids.dtype == torch.bool:
        raise TypeError(f"token_ids must be an integer tensor, got {token_ids.dtype}.")
    inputs, targets = shift_inputs_targets(token_ids, dim=-1)
    if mask is not None:
        if mask.shape != token_ids.shape or mask.dtype != torch.bool:
            raise ValueError("mask must be a bool tensor with the same shape as token_ids.")
        pad_positions = ~mask[..., 1:]
    elif pad_id is not None:
        pad_positions = targets == pad_id
    else:
        return inputs, targets.clone()
    return inputs, targets.masked_fill(pad_positions, ignore_index)


def shift_right(x: torch.Tensor, fill_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    """Shift one step towards later positions: out[0] = fill_value, out[t] = x[t-1] (same length)."""
    length = x.shape[dim]
    fill_shape = list(x.shape)
    fill_shape[dim] = 1
    fill = torch.full(fill_shape, fill_value, dtype=x.dtype, device=x.device)
    return torch.cat([fill, x.narrow(dim, 0, length - 1)], dim=dim)


def shift_left(x: torch.Tensor, fill_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    """Shift one step towards earlier positions: out[t] = x[t+1], out[-1] = fill_value (same length)."""
    length = x.shape[dim]
    fill_shape = list(x.shape)
    fill_shape[dim] = 1
    fill = torch.full(fill_shape, fill_value, dtype=x.dtype, device=x.device)
    return torch.cat([x.narrow(dim, 1, length - 1), fill], dim=dim)
