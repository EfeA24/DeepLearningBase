# Purpose: Provides mask-aware mean pooling over padded recurrent outputs for LSTM/GRU sequence representations.
"""Masked mean pooling.

Contract:
    - ``outputs``: ``(B, T, H)``; ``mask``: ``(B, T)`` bool, True = valid. Alternatively pass ``lengths``
      ``(B,)`` for right-padded batches.
    - Returns ``(B, H)`` = sum of valid steps / number of valid steps. Rows with zero valid steps return zeros.

Requires: torch only.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn


def _resolve_mask(outputs: torch.Tensor, mask: Optional[torch.Tensor], lengths: Optional[torch.Tensor]) -> torch.Tensor:
    if (mask is None) == (lengths is None):
        raise ValueError("Pass exactly one of mask or lengths.")
    if mask is None:
        positions = torch.arange(outputs.shape[1], device=outputs.device).unsqueeze(0)
        mask = positions < lengths.to(outputs.device).unsqueeze(1)
    if mask.shape != outputs.shape[:2]:
        raise ValueError(f"mask must be {tuple(outputs.shape[:2])}, got {tuple(mask.shape)}.")
    return mask.bool()


def masked_mean(
    outputs: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Average valid time steps of ``(B, T, H)`` outputs -> ``(B, H)``."""
    if outputs.dim() != 3:
        raise ValueError(f"outputs must be (B, T, H), got {tuple(outputs.shape)}.")
    valid = _resolve_mask(outputs, mask, lengths)
    weights = valid.unsqueeze(-1).to(outputs.dtype)
    total = (outputs * weights).sum(dim=1)
    counts = weights.sum(dim=1).clamp_min(1.0)
    return total / counts


class MeanPooling(nn.Module):
    """Module wrapper around ``masked_mean``."""

    def forward(
        self,
        outputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return masked_mean(outputs, mask=mask, lengths=lengths)
