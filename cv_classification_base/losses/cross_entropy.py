# Purpose: Computes single-label cross-entropy from class logits with optional class weights and ignore_index for image classification.
"""Single-label cross-entropy loss.

Contract:
    logits: float ``(B, C)`` raw scores (no softmax applied beforehand).
    targets: int64 ``(B,)`` class ids in ``[0, C)`` or ``ignore_index``.
    class_weights: optional float ``(C,)``; with ``reduction="mean"`` the result is the weighted
        mean ``sum(w[y_i] * l_i) / sum(w[y_i])`` over non-ignored samples (torch semantics).

If every target equals ``ignore_index`` the "mean" reduction returns NaN (torch behavior);
``ClassificationCrossEntropy`` returns 0 (with grad) instead so a fully-ignored batch is harmless.

Requires: torch.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch import nn

REDUCTIONS = ("mean", "sum", "none")


def _check_inputs(logits: torch.Tensor, targets: torch.Tensor) -> None:
    if logits.dim() != 2:
        raise ValueError(f"logits must be (B, C), got shape {tuple(logits.shape)}")
    if targets.dim() != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError(f"targets must be (B,) matching logits, got shape {tuple(targets.shape)}")
    if targets.dtype != torch.long:
        raise TypeError(f"targets must be int64 class ids, got dtype {targets.dtype}")


class ClassificationCrossEntropy(nn.Module):
    """``F.cross_entropy`` wrapper with validated inputs and a registered class-weight buffer."""

    def __init__(
        self,
        class_weights: Optional[Union[torch.Tensor, Sequence[float]]] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in REDUCTIONS:
            raise ValueError(f"reduction must be one of {REDUCTIONS}, got {reduction!r}")
        if class_weights is not None:
            w = torch.as_tensor(class_weights, dtype=torch.float32)
            if w.dim() != 1 or torch.any(w < 0):
                raise ValueError("class_weights must be a non-negative 1D tensor of length C")
            self.register_buffer("class_weights", w)
        else:
            self.class_weights = None
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        _check_inputs(logits, targets)
        weight = None
        if self.class_weights is not None:
            if self.class_weights.numel() != logits.shape[1]:
                raise ValueError(
                    f"class_weights has {self.class_weights.numel()} entries but logits have "
                    f"{logits.shape[1]} classes"
                )
            weight = self.class_weights.to(dtype=logits.dtype)
        if self.reduction == "mean" and bool((targets == self.ignore_index).all()):
            return logits.sum() * 0.0
        return F.cross_entropy(
            logits, targets, weight=weight, ignore_index=self.ignore_index, reduction=self.reduction
        )
