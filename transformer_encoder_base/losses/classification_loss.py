# Purpose: Text classification loss over raw logits: single-label cross-entropy (class weights, label smoothing, ignore_index) or multilabel BCE-with-logits.
"""Classification losses for encoder classifiers.

Single-label (``multilabel=False``):
    logits (B, C) raw scores; labels (B,) long in [0, C) or ``ignore_index`` (-100, skipped).
    F.cross_entropy semantics: with ``class_weights`` the "mean" reduction divides by the sum of the
    weights of the non-ignored targets. ``label_smoothing`` in [0, 1).
Multilabel (``multilabel=True``):
    logits (B, C) raw scores; labels (B, C) float multi-hot in {0, 1} (soft targets in [0, 1] allowed).
    Uses BCE-with-logits; ``pos_weight`` (C,) rescales positives; no ignore_index / smoothing.
Never pass probabilities: sigmoid/softmax are applied inside the loss.

Requires: torch only.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["ClassificationLoss"]


class ClassificationLoss(nn.Module):
    """Configurable CE (single-label) or BCE-with-logits (multilabel) loss."""

    def __init__(
        self,
        multilabel: bool = False,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum' or 'none', got {reduction!r}")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")
        if multilabel and (class_weights is not None or label_smoothing > 0):
            raise ValueError("class_weights/label_smoothing apply to single-label mode; use pos_weight for multilabel")
        if not multilabel and pos_weight is not None:
            raise ValueError("pos_weight applies to multilabel mode only")
        self.multilabel = multilabel
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.register_buffer("class_weights", None if class_weights is None else class_weights.float())
        self.register_buffer("pos_weight", None if pos_weight is None else pos_weight.float())

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.dim() != 2:
            raise ValueError(f"logits must be (B, C), got shape {tuple(logits.shape)}")
        if self.multilabel:
            if labels.shape != logits.shape:
                raise ValueError(f"multilabel targets must be (B, C) = {tuple(logits.shape)}, got {tuple(labels.shape)}")
            return F.binary_cross_entropy_with_logits(
                logits.float(), labels.float(), pos_weight=self.pos_weight, reduction=self.reduction
            )
        if labels.shape != logits.shape[:1]:
            raise ValueError(f"labels must be (B,) = ({logits.shape[0]},), got {tuple(labels.shape)}")
        if labels.dtype.is_floating_point:
            raise TypeError("single-label targets must be integer class ids")
        return F.cross_entropy(
            logits.float(),
            labels.long(),
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction,
        )
