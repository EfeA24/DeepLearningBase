# Purpose: Masked-language-modeling cross-entropy over selected targets (labels -100 ignored) normalized by the number of predicted tokens.
"""MLM loss.

Inputs:
    logits: (B, T, V) or already-selected (N, V) raw scores (no softmax).
    labels: (B, T) or (N,) long; original token id at masked positions, ``ignore_index`` (-100) elsewhere.
Normalization: "mean" = sum of token losses / number of non-ignored tokens (token-count
normalization, not a mean of per-sequence means). If no token is selected, "mean" returns a zero
that is still connected to the graph (so backward() works).
For gradient accumulation across micro-batches, use ``mlm_loss_sum_and_count`` and divide the
summed loss by the total count yourself.

Requires: torch only.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["mlm_loss", "mlm_loss_sum_and_count", "MLMLoss"]


def _flatten(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if logits.shape[:-1] != labels.shape:
        raise ValueError(f"logits {tuple(logits.shape)} and labels {tuple(labels.shape)} are incompatible")
    if labels.dtype.is_floating_point or labels.dtype == torch.bool:
        raise TypeError(f"labels must be integer token ids, got {labels.dtype}")
    return logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)


def mlm_loss_sum_and_count(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (sum of token CE losses, number of non-ignored tokens as long tensor)."""
    flat_logits, flat_labels = _flatten(logits, labels)
    loss_sum = F.cross_entropy(flat_logits.float(), flat_labels, ignore_index=ignore_index, reduction="sum")
    count = (flat_labels != ignore_index).sum()
    return loss_sum, count


def mlm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    """MLM cross-entropy; ``reduction`` in {"mean", "sum", "none"} ("none" returns per-token (N_total,))."""
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"reduction must be 'mean', 'sum' or 'none', got {reduction!r}")
    if reduction == "none":
        flat_logits, flat_labels = _flatten(logits, labels)
        return F.cross_entropy(flat_logits.float(), flat_labels, ignore_index=ignore_index, reduction="none")
    loss_sum, count = mlm_loss_sum_and_count(logits, labels, ignore_index)
    if reduction == "sum":
        return loss_sum
    return loss_sum / count.clamp(min=1).to(loss_sum.dtype)


class MLMLoss(nn.Module):
    """Module wrapper around ``mlm_loss``."""

    def __init__(self, ignore_index: int = -100, reduction: str = "mean") -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return mlm_loss(logits, labels, ignore_index=self.ignore_index, reduction=self.reduction)
