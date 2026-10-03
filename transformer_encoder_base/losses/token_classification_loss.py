# Purpose: Token classification loss (NER/POS) that excludes padding and -100 labels and normalizes by the number of valid tokens.
"""Token classification loss.

Inputs:
    logits: (B, T, C) raw scores.
    labels: (B, T) long; ``ignore_index`` (-100) marks special tokens, non-first subwords, padding.
    attention_mask (optional): (B, T) bool or 0/1, True/1 = valid; masked positions are ignored even
        if their label is not -100.
    class_weights (optional): (C,) per-class weights.
Normalization ("mean"): sum of (weighted) token losses / number of valid tokens. Note this differs
from F.cross_entropy(weight=...), which divides by the sum of weights. With no valid token the result
is a graph-connected zero.

Requires: torch only.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["token_classification_loss", "token_loss_sum_and_count", "TokenClassificationLoss"]


def token_loss_sum_and_count(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    class_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (summed token loss, valid token count as long tensor)."""
    if logits.dim() != 3 or labels.shape != logits.shape[:2]:
        raise ValueError(f"expected logits (B, T, C) and labels (B, T); got {tuple(logits.shape)}, {tuple(labels.shape)}")
    if labels.dtype.is_floating_point:
        raise TypeError("labels must be integer class ids")
    labels = labels.long()
    if attention_mask is not None:
        if attention_mask.shape != labels.shape:
            raise ValueError("attention_mask must be (B, T) like labels")
        if attention_mask.dtype.is_floating_point:
            raise TypeError("attention_mask must be bool or integer 0/1")
        labels = labels.masked_fill(attention_mask == 0, ignore_index)
    flat_labels = labels.reshape(-1)
    weight = None if class_weights is None else class_weights.to(device=logits.device, dtype=torch.float32)
    loss_sum = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        flat_labels,
        weight=weight,
        ignore_index=ignore_index,
        reduction="sum",
    )
    return loss_sum, (flat_labels != ignore_index).sum()


def token_classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    class_weights: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Token CE with ``reduction`` in {"mean" (per valid token), "sum"}."""
    if reduction not in ("mean", "sum"):
        raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")
    loss_sum, count = token_loss_sum_and_count(logits, labels, attention_mask, ignore_index, class_weights)
    if reduction == "sum":
        return loss_sum
    return loss_sum / count.clamp(min=1).to(loss_sum.dtype)


class TokenClassificationLoss(nn.Module):
    """Module wrapper around ``token_classification_loss``."""

    def __init__(
        self,
        ignore_index: int = -100,
        class_weights: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.register_buffer("class_weights", None if class_weights is None else class_weights.float())

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return token_classification_loss(
            logits, labels, attention_mask, self.ignore_index, self.class_weights, self.reduction
        )
