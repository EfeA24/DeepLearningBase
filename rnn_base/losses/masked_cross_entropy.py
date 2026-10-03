# Purpose: Computes cross entropy only at valid (mask=True) sequence positions, normalized by the valid-token count.
"""Masked cross entropy for padded sequence batches.

- logits: (B, T, V) raw scores; targets: (B, T) int64; mask: (B, T) bool, True = valid.
- Targets at invalid positions may hold any value (pad id, -100, ...); they are never indexed.
- "mean" = sum of valid-token losses / number of valid tokens (token-weighted). If no token
  is valid it returns 0.0 connected to the graph (backward works, gradients are zero).
- "sum" = sum of valid-token losses; "none" = (B, T) losses with zeros at invalid positions.
- ``masked_cross_entropy_sum_and_count`` returns (sum, count) for exact accumulation across
  micro-batches / epochs (divide total sum by total count once).

Requires: torch only. No sibling files.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


def _masked_token_losses(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if logits.dim() != 3 or targets.shape != logits.shape[:2]:
        raise ValueError(f"Expected logits (B, T, V) and targets (B, T); got {tuple(logits.shape)} and {tuple(targets.shape)}.")
    if mask.shape != targets.shape or mask.dtype != torch.bool:
        raise ValueError("mask must be a bool tensor with the same shape as targets (True = valid).")
    if targets.dtype != torch.long:
        raise TypeError(f"targets must be int64 class indices, got {targets.dtype}.")
    safe_targets = targets.masked_fill(~mask, 0)
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), safe_targets.reshape(-1), reduction="none"
    ).view(targets.shape)
    return losses.masked_fill(~mask, 0.0)


def masked_cross_entropy_sum_and_count(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (sum of valid-token losses, number of valid tokens as int64 tensor)."""
    losses = _masked_token_losses(logits, targets, mask)
    return losses.sum(), mask.sum()


def masked_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """Cross entropy restricted to ``mask=True`` positions; see module docstring."""
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"reduction must be 'mean', 'sum' or 'none', got {reduction!r}.")
    losses = _masked_token_losses(logits, targets, mask)
    if reduction == "none":
        return losses
    total = losses.sum()
    if reduction == "sum":
        return total
    return total / mask.sum().clamp(min=1).to(total.dtype)


class MaskedCrossEntropyLoss(nn.Module):
    """Module form of :func:`masked_cross_entropy`; forward(logits, targets, mask)."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum' or 'none', got {reduction!r}.")
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return masked_cross_entropy(logits, targets, mask, self.reduction)
