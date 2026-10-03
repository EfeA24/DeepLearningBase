# Purpose: Computes per-box classification loss (softmax cross-entropy or one-hot BCE-with-logits) for matched detection predictions.
"""Per-box classification loss for matched predictions.

Input contract:
    logits: FloatTensor (N, C) raw class logits of predictions assigned to objects.
    labels: LongTensor (N,) class ids in [0, C) (foreground only; background is
            handled by the objectness loss in YOLO-style heads).
    weights: optional (N,) per-box weights.

Modes:
    "ce"  - softmax cross-entropy (mutually exclusive classes); supports
            ``label_smoothing`` and ``class_weights``.
    "bce" - sigmoid BCE against one-hot targets (independent classes, typical
            for YOLO-style heads); ``label_smoothing`` s maps one-hot targets to
            1 - s + s / C (true class) and s / C (other classes). Per-box loss
            sums over classes.

Reduction "mean" divides by ``normalizer`` if given, else N (or sum of weights);
empty input returns a graph-connected zero.

Requires: torch only.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ClassificationLoss(nn.Module):
    """Classification loss over matched predictions.

    Args:
        num_classes: C.
        mode: "ce" or "bce".
        label_smoothing: In [0, 1).
        class_weights: Optional per-class weights (C,).
        reduction: "mean", "sum" or "none".
    """

    def __init__(
        self,
        num_classes: int,
        mode: str = "bce",
        label_smoothing: float = 0.0,
        class_weights: Optional[Sequence[float]] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be > 0")
        if mode not in ("ce", "bce"):
            raise ValueError("mode must be 'ce' or 'bce'")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be 'mean', 'sum' or 'none'")
        if class_weights is not None and len(class_weights) != num_classes:
            raise ValueError("class_weights must have num_classes entries")
        self.num_classes = num_classes
        self.mode = mode
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        if class_weights is not None:
            self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

    def per_box(self, logits: Tensor, labels: Tensor) -> Tensor:
        if logits.ndim != 2 or logits.shape[1] != self.num_classes:
            raise ValueError(f"logits must be (N, {self.num_classes}), got {tuple(logits.shape)}")
        if labels.shape != (logits.shape[0],) or labels.dtype != torch.long:
            raise ValueError("labels must be int64 of shape (N,)")
        if labels.numel() and (labels.min() < 0 or labels.max() >= self.num_classes):
            raise ValueError(f"labels must be in [0, {self.num_classes})")
        logits_f = logits.float()
        if self.mode == "ce":
            return F.cross_entropy(
                logits_f, labels, weight=self.class_weights, reduction="none", label_smoothing=self.label_smoothing
            )
        s = self.label_smoothing
        targets = F.one_hot(labels, self.num_classes).to(logits_f.dtype)
        targets = targets * (1.0 - s) + s / self.num_classes
        loss = F.binary_cross_entropy_with_logits(logits_f, targets, reduction="none")
        if self.class_weights is not None:
            loss = loss * self.class_weights
        return loss.sum(dim=1)

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        weights: Optional[Tensor] = None,
        normalizer: Optional[float] = None,
    ) -> Tensor:
        losses = self.per_box(logits, labels)
        if weights is not None:
            if weights.shape != losses.shape:
                raise ValueError("weights must have shape (N,)")
            losses = losses * weights.to(losses.dtype)
        if self.reduction == "none":
            return losses
        if losses.numel() == 0:
            return logits.sum() * 0.0
        total = losses.sum()
        if self.reduction == "sum":
            return total
        if normalizer is not None:
            denom = float(normalizer)
        elif weights is not None:
            denom = float(weights.sum())
        else:
            denom = float(losses.shape[0])
        return total / max(denom, 1e-12)
