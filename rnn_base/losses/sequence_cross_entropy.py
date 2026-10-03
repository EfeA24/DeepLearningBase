# Purpose: Computes token-level or sequence-level cross entropy for recurrent models with explicit flattening and ignore_index.
"""Sequence cross entropy.

Accepted inputs (logits are raw, never softmaxed):
- token level: logits (B, T, V), targets (B, T) int64  -> flattened to (B*T, V) / (B*T,)
- sequence level: logits (B, C), targets (B,) int64
Targets equal to ``ignore_index`` (default -100, e.g. padded LM targets) are excluded.
Reductions:
- "mean": sum of losses over non-ignored targets / number of non-ignored targets
  (token-weighted, not a mean of per-sequence means); 0.0 (with grad) if nothing is valid;
- "sum": sum over non-ignored targets;
- "none": unreduced loss with the targets' shape (ignored positions are 0).

Requires: torch only. No sibling files.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

_REDUCTIONS = ("mean", "sum", "none")


def sequence_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Cross entropy over (B, T, V)/(B, T) or (B, C)/(B,) inputs; see module docstring."""
    if reduction not in _REDUCTIONS:
        raise ValueError(f"reduction must be one of {_REDUCTIONS}, got {reduction!r}.")
    if logits.dim() != targets.dim() + 1 or logits.shape[:-1] != targets.shape:
        raise ValueError(
            f"Shape mismatch: logits {tuple(logits.shape)} must be targets {tuple(targets.shape)} + (V,)."
        )
    if targets.dtype != torch.long:
        raise TypeError(f"targets must be int64 class indices, got {targets.dtype}.")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}.")

    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    losses = F.cross_entropy(
        flat_logits, flat_targets, ignore_index=ignore_index, reduction="none", label_smoothing=label_smoothing
    )
    if reduction == "none":
        return losses.view(targets.shape)
    total = losses.sum()
    if reduction == "sum":
        return total
    count = (flat_targets != ignore_index).sum()
    return total / count.clamp(min=1).to(total.dtype)


class SequenceCrossEntropyLoss(nn.Module):
    """Module form of :func:`sequence_cross_entropy`."""

    def __init__(self, ignore_index: int = -100, reduction: str = "mean", label_smoothing: float = 0.0) -> None:
        super().__init__()
        if reduction not in _REDUCTIONS:
            raise ValueError(f"reduction must be one of {_REDUCTIONS}, got {reduction!r}.")
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return sequence_cross_entropy(logits, targets, self.ignore_index, self.reduction, self.label_smoothing)
