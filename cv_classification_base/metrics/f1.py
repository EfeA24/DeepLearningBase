# Purpose: Computes F1 scores with explicit averaging and zero-division policy from accumulated confusion counts for single-label image classification.
"""F1 from a confusion matrix (rows = true class, cols = predicted class).

Per class c: ``F1_c = 2*TP_c / (2*TP_c + FP_c + FN_c)`` (harmonic mean of precision and recall).
Undefined when the class is neither present nor predicted; the value is then ``zero_division``
(NaN -> excluded from "macro"/"weighted").

Averaging:
    "none"     -> per-class tensor ``(C,)``
    "macro"    -> mean of per-class F1 (NOT the harmonic mean of macro precision and macro recall)
    "weighted" -> per-class F1 weighted by true-class support
    "micro"    -> ``2*sum TP / (2*sum TP + sum FP + sum FN)`` (equals accuracy for single-label)

``F1Meter`` accumulates counts batch-wise so the score is computed once over the full epoch
(averaging per-batch F1 values would be wrong).

Requires: torch.
"""
from __future__ import annotations

import math
from typing import Optional, Union

import torch

AVERAGES = ("none", "macro", "micro", "weighted")


def f1_from_confusion(
    matrix: torch.Tensor, average: str = "macro", zero_division: float = 0.0
) -> Union[torch.Tensor, float]:
    """Return per-class F1 ``(C,)`` for ``average="none"``, else a Python float."""
    if matrix.dim() != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"confusion matrix must be square (C, C), got {tuple(matrix.shape)}")
    if average not in AVERAGES:
        raise ValueError(f"average must be one of {AVERAGES}, got {average!r}")
    m = matrix.to(torch.float64)
    tp = m.diagonal()
    fp = m.sum(dim=0) - tp
    fn = m.sum(dim=1) - tp
    if average == "micro":
        denom = 2 * tp.sum() + fp.sum() + fn.sum()
        return float(2 * tp.sum() / denom) if denom > 0 else float(zero_division)
    denom = 2 * tp + fp + fn
    per_class = torch.where(
        denom > 0, 2 * tp / denom.clamp_min(1), torch.full_like(tp, float(zero_division))
    )
    if average == "none":
        return per_class.float()
    defined = ~torch.isnan(per_class)
    if not bool(defined.any()):
        return math.nan
    if average == "macro":
        return float(per_class[defined].mean())
    support = (tp + fn)[defined]
    if support.sum() == 0:
        return math.nan
    return float((per_class[defined] * support).sum() / support.sum())


class F1Meter:
    """Accumulates a confusion matrix from logits or predicted ids and reports F1."""

    def __init__(
        self,
        num_classes: int,
        average: str = "macro",
        zero_division: float = 0.0,
        input_mode: str = "logits",
        ignore_index: Optional[int] = None,
    ) -> None:
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        if average not in AVERAGES:
            raise ValueError(f"average must be one of {AVERAGES}, got {average!r}")
        if input_mode not in ("logits", "preds"):
            raise ValueError("input_mode must be 'logits' or 'preds'")
        self.num_classes = num_classes
        self.average = average
        self.zero_division = zero_division
        self.input_mode = input_mode
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        self.matrix = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long)

    @torch.no_grad()
    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        preds = outputs.argmax(dim=1) if self.input_mode == "logits" else outputs
        preds = preds.detach().reshape(-1).long().cpu()
        targets = targets.detach().reshape(-1).long().cpu()
        if preds.shape != targets.shape:
            raise ValueError("predictions and targets must contain the same number of samples")
        if self.ignore_index is not None:
            keep = targets != self.ignore_index
            preds, targets = preds[keep], targets[keep]
        if preds.numel() == 0:
            return
        if min(preds.min(), targets.min()) < 0 or max(preds.max(), targets.max()) >= self.num_classes:
            raise ValueError(f"class ids must lie in [0, {self.num_classes})")
        counts = torch.bincount(targets * self.num_classes + preds, minlength=self.num_classes**2)
        self.matrix += counts.reshape(self.num_classes, self.num_classes)

    def compute(self) -> Union[torch.Tensor, float]:
        return f1_from_confusion(self.matrix, self.average, self.zero_division)
