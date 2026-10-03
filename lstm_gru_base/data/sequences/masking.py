# Purpose: Builds valid-position boolean masks from lengths or pad ids for padded LSTM/GRU batches.
"""Masking helpers for gated-RNN batches.

Contract:
    - Batch-first tensors: token ids ``(B, T)``, lengths ``(B,)`` int64.
    - Mask polarity in this template: ``True`` = valid (real token), ``False`` = padding.
      Use ``invert_mask`` to get the ``True`` = pad convention (e.g. ``key_padding_mask`` of nn.MultiheadAttention).
    - ``side`` describes where padding was added: ``"right"`` (valid prefix) or ``"left"`` (valid suffix).

Requires: torch only.
"""
from __future__ import annotations

from typing import Optional

import torch


def lengths_to_mask(
    lengths: torch.Tensor,
    max_length: Optional[int] = None,
    side: str = "right",
) -> torch.Tensor:
    """Convert lengths ``(B,)`` to a bool mask ``(B, T)`` with True at valid positions."""
    if side not in ("right", "left"):
        raise ValueError(f"side must be 'right' or 'left', got {side!r}.")
    if lengths.dim() != 1:
        raise ValueError(f"lengths must be 1-D (B,), got shape {tuple(lengths.shape)}.")
    if lengths.numel() and int(lengths.min()) < 0:
        raise ValueError("lengths must be non-negative.")
    lengths = lengths.long()
    longest = int(lengths.max().item()) if lengths.numel() else 0
    t = longest if max_length is None else int(max_length)
    if t < longest:
        raise ValueError(f"max_length={t} is smaller than the largest length {longest}.")
    positions = torch.arange(t, device=lengths.device).unsqueeze(0)
    if side == "right":
        return positions < lengths.unsqueeze(1)
    return positions >= (t - lengths).unsqueeze(1)


def mask_from_pad_id(token_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Return a bool mask ``(B, T)`` with True where ``token_ids != pad_id``.

    Only correct if ``pad_id`` never appears as a real token inside a sequence.
    """
    if token_ids.dim() != 2:
        raise ValueError(f"token_ids must be (B, T), got shape {tuple(token_ids.shape)}.")
    return token_ids != pad_id


def mask_to_lengths(mask: torch.Tensor) -> torch.Tensor:
    """Count valid positions per row of a ``(B, T)`` bool mask (True = valid)."""
    if mask.dim() != 2:
        raise ValueError(f"mask must be (B, T), got shape {tuple(mask.shape)}.")
    return mask.long().sum(dim=1)


def invert_mask(mask: torch.Tensor) -> torch.Tensor:
    """Switch between True = valid and True = pad conventions."""
    return ~mask.bool()


def masked_fill_padding(values: torch.Tensor, mask: torch.Tensor, fill_value: float) -> torch.Tensor:
    """Fill padded positions of ``values`` ``(B, T, ...)`` with ``fill_value`` given mask ``(B, T)`` (True = valid)."""
    if values.shape[:2] != mask.shape:
        raise ValueError(f"mask shape {tuple(mask.shape)} must match values[:2] {tuple(values.shape[:2])}.")
    expanded = mask.bool().reshape(*mask.shape, *([1] * (values.dim() - 2)))
    return values.masked_fill(~expanded, fill_value)
