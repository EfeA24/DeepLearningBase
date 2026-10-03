# Purpose: Accumulates a fixed-size confusion matrix and derives per-class, macro, micro and weighted precision/recall/F1 with an explicit zero-division policy.
"""Confusion-matrix based classification metrics.

Layout: ``matrix[true_class, predicted_class]`` (rows = targets, columns = predictions), int64,
shape (num_classes, num_classes). Class order is index order 0..num_classes-1.

Inputs per ``update``: ``preds`` and ``targets`` int tensors of identical shape (any shape,
flattened). To use logits (B, C) call ``update_from_logits`` (argmax over dim 1).

Averaging:
    - "none": per-class tensor (num_classes,).
    - "macro": unweighted mean over classes.
    - "weighted": mean weighted by class support (true count).
    - "micro": global TP/FP/FN; for single-label multi-class, micro P = micro R = F1 = accuracy.
Absent classes (support 0 and never predicted) produce 0/0; ``zero_division`` decides the value
(0.0, 1.0, or float('nan')). With "macro", NaN entries are excluded from the mean when
``zero_division`` is NaN (nan-mean); otherwise all classes count.

Requires: torch.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch

_AVERAGES = ("none", "macro", "micro", "weighted")


class ConfusionCounts:
    """Fixed-size confusion matrix accumulator.

    Args:
        num_classes: Number of classes (>= 2 recommended; >= 1 allowed).
        ignore_index: Optional target value to skip (e.g. -100).
        zero_division: Value used for 0/0 ratios (0.0, 1.0 or nan).
    """

    def __init__(self, num_classes: int, ignore_index: Optional[int] = None, zero_division: float = 0.0) -> None:
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        if not (zero_division in (0.0, 1.0) or math.isnan(zero_division)):
            raise ValueError("zero_division must be 0.0, 1.0 or float('nan')")
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.zero_division = zero_division
        self.reset()

    def reset(self) -> None:
        self.matrix = torch.zeros(self.num_classes, self.num_classes, dtype=torch.int64)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        if preds.shape != targets.shape:
            raise ValueError(f"preds {tuple(preds.shape)} and targets {tuple(targets.shape)} must match")
        if preds.dtype.is_floating_point or targets.dtype.is_floating_point:
            raise TypeError("preds and targets must be integer tensors (use update_from_logits for logits)")
        preds = preds.reshape(-1).cpu()
        targets = targets.reshape(-1).cpu()
        if self.ignore_index is not None:
            keep = targets != self.ignore_index
            preds, targets = preds[keep], targets[keep]
        if targets.numel() == 0:
            return
        n = self.num_classes
        if targets.min() < 0 or targets.max() >= n or preds.min() < 0 or preds.max() >= n:
            raise ValueError(f"preds and targets must be in [0, {n})")
        self.matrix += torch.bincount(targets * n + preds, minlength=n * n).reshape(n, n)

    def update_from_logits(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """Logits (B, C) -> argmax predictions (B,)."""
        if logits.ndim != 2 or logits.shape[1] != self.num_classes:
            raise ValueError(f"logits must have shape (B, {self.num_classes}), got {tuple(logits.shape)}")
        self.update(logits.argmax(dim=1), targets)

    def _counts(self):
        m = self.matrix.to(torch.float64)
        tp = m.diag()
        fp = m.sum(dim=0) - tp
        fn = m.sum(dim=1) - tp
        support = m.sum(dim=1)
        return tp, fp, fn, support

    def _ratio(self, num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
        out = torch.full_like(num, float(self.zero_division))
        mask = den > 0
        out[mask] = num[mask] / den[mask]
        return out

    def _reduce(self, per_class: torch.Tensor, support: torch.Tensor, average: str):
        if average == "none":
            return per_class
        if average == "macro":
            valid = ~torch.isnan(per_class)
            return per_class[valid].mean().item() if valid.any() else float("nan")
        if average == "weighted":
            total = support.sum()
            if total == 0:
                return float(self.zero_division)
            vals = torch.nan_to_num(per_class, nan=0.0)
            return (vals * support).sum().item() / total.item()
        raise ValueError(f"average must be one of {_AVERAGES}, got {average!r}")

    def precision(self, average: str = "macro"):
        tp, fp, fn, support = self._counts()
        if average == "micro":
            return self._ratio(tp.sum().reshape(1), (tp.sum() + fp.sum()).reshape(1)).item()
        return self._reduce(self._ratio(tp, tp + fp), support, average)

    def recall(self, average: str = "macro"):
        tp, fp, fn, support = self._counts()
        if average == "micro":
            return self._ratio(tp.sum().reshape(1), (tp.sum() + fn.sum()).reshape(1)).item()
        return self._reduce(self._ratio(tp, tp + fn), support, average)

    def f1(self, average: str = "macro"):
        """F1 = 2TP / (2TP + FP + FN) per class (equivalent to harmonic mean of P and R)."""
        tp, fp, fn, support = self._counts()
        if average == "micro":
            return self._ratio((2 * tp.sum()).reshape(1), (2 * tp.sum() + fp.sum() + fn.sum()).reshape(1)).item()
        return self._reduce(self._ratio(2 * tp, 2 * tp + fp + fn), support, average)

    def accuracy(self) -> float:
        total = self.matrix.sum().item()
        if total == 0:
            raise RuntimeError("No samples accumulated")
        return self.matrix.diag().sum().item() / total

    def compute(self, average: str = "macro") -> Dict[str, object]:
        """Return accuracy plus precision/recall/F1 under ``average``."""
        if average not in _AVERAGES:
            raise ValueError(f"average must be one of {_AVERAGES}, got {average!r}")
        return {
            "accuracy": self.accuracy(),
            "precision": self.precision(average),
            "recall": self.recall(average),
            "f1": self.f1(average),
        }
