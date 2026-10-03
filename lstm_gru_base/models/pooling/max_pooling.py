# Purpose: Provides mask-aware max pooling over padded recurrent outputs with safe handling of fully masked rows for LSTM/GRU sequence representations.
"""Masked max pooling.

Contract:
    - ``outputs``: ``(B, T, H)``; ``mask``: ``(B, T)`` bool, True = valid, or ``lengths`` ``(B,)`` (right padding).
    - Returns ``(B, H)``: per-feature max over valid steps only (padding never wins, even if it is 0 and
      all real values are negative).
    - Rows with no valid steps return ``empty_fill`` (default 0.0) instead of ``-inf``/dtype-min.

Requires: torch only.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn


def masked_max(
    outputs: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    lengths: Optional[torch.Tensor] = None,
    empty_fill: float = 0.0,
) -> torch.Tensor:
    """Max over valid time steps of ``(B, T, H)`` outputs -> ``(B, H)``."""
    if outputs.dim() != 3:
        raise ValueError(f"outputs must be (B, T, H), got {tuple(outputs.shape)}.")
    if (mask is None) == (lengths is None):
        raise ValueError("Pass exactly one of mask or lengths.")
    if mask is None:
        positions = torch.arange(outputs.shape[1], device=outputs.device).unsqueeze(0)
        mask = positions < lengths.to(outputs.device).unsqueeze(1)
    if mask.shape != outputs.shape[:2]:
        raise ValueError(f"mask must be {tuple(outputs.shape[:2])}, got {tuple(mask.shape)}.")
    valid = mask.bool()
    filled = outputs.masked_fill(~valid.unsqueeze(-1), torch.finfo(outputs.dtype).min)
    pooled = filled.max(dim=1).values
    has_valid = valid.any(dim=1, keepdim=True)
    return torch.where(has_valid, pooled, torch.full_like(pooled, empty_fill))


class MaxPooling(nn.Module):
    """Module wrapper around ``masked_max``."""

    def __init__(self, empty_fill: float = 0.0) -> None:
        super().__init__()
        self.empty_fill = empty_fill

    def forward(
        self,
        outputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return masked_max(outputs, mask=mask, lengths=lengths, empty_fill=self.empty_fill)
