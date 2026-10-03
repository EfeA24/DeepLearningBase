# Purpose: Computes per-class and macro/micro/weighted recall with an explicit zero-division policy for single-label image classification.
"""Recall from a confusion matrix (rows = true class, cols = predicted class).

Per class c: ``recall_c = TP_c / (TP_c + FN_c)`` where ``TP_c + FN_c = M[c, :].sum()`` (support).

Zero division (class absent from targets): the value is ``zero_division`` (default 0.0). If
``zero_division`` is NaN, absent classes are EXCLUDED from "macro"/"weighted" averages.

Averaging:
    "none"     -> per-class tensor ``(C,)``
    "macro"    -> unweighted mean over all C classes ("balanced accuracy" when no class is absent)
    "weighted" -> support-weighted mean (equals accuracy for single-label problems)
    "micro"    -> ``sum TP / sum support`` (equals accuracy for single-label problems)

Requires: torch.
"""
from __future__ import annotations

import math
from typing import Union

import torch

AVERAGES = ("none", "macro", "micro", "weighted")


def _validate(matrix: torch.Tensor, average: str) -> torch.Tensor:
    if matrix.dim() != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"confusion matrix must be square (C, C), got {tuple(matrix.shape)}")
    if average not in AVERAGES:
        raise ValueError(f"average must be one of {AVERAGES}, got {average!r}")
    return matrix.to(torch.float64)


def _average(values: torch.Tensor, support: torch.Tensor, average: str) -> torch.Tensor:
    defined = ~torch.isnan(values)
    if not bool(defined.any()):
        return torch.tensor(math.nan, dtype=torch.float64)
    if average == "macro":
        return values[defined].mean()
    weights = support[defined]
    if weights.sum() == 0:
        return torch.tensor(math.nan, dtype=torch.float64)
    return (values[defined] * weights).sum() / weights.sum()


def recall_from_confusion(
    matrix: torch.Tensor, average: str = "macro", zero_division: float = 0.0
) -> Union[torch.Tensor, float]:
    """Return per-class recall ``(C,)`` for ``average="none"``, else a Python float."""
    m = _validate(matrix, average)
    tp = m.diagonal()
    support = m.sum(dim=1)
    if average == "micro":
        total = support.sum()
        return float(tp.sum() / total) if total > 0 else float(zero_division)
    per_class = torch.where(
        support > 0, tp / support.clamp_min(1), torch.full_like(tp, float(zero_division))
    )
    if average == "none":
        return per_class.float()
    return float(_average(per_class, support, average))


def recall_score(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    average: str = "macro",
    zero_division: float = 0.0,
) -> Union[torch.Tensor, float]:
    """Convenience: recall from 1D predicted ids and targets in ``[0, num_classes)``."""
    if preds.shape != targets.shape or preds.dim() != 1:
        raise ValueError("preds and targets must be 1D tensors of equal length")
    preds, targets = preds.long().cpu(), targets.long().cpu()
    if preds.numel() and (min(preds.min(), targets.min()) < 0 or max(preds.max(), targets.max()) >= num_classes):
        raise ValueError(f"class ids must lie in [0, {num_classes})")
    matrix = torch.bincount(targets * num_classes + preds, minlength=num_classes**2)
    return recall_from_confusion(matrix.reshape(num_classes, num_classes), average, zero_division)
