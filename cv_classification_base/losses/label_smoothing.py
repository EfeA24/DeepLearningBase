# Purpose: Computes label-smoothed cross-entropy from class logits with a validated smoothing factor for image classification.
"""Label-smoothed cross-entropy.

Target distribution for true class ``y`` with smoothing ``eps`` and ``C`` classes:
    q_k = (1 - eps) * [k == y] + eps / C
so the loss is ``(1 - eps) * NLL(y) + eps * mean_k(-log p_k)`` (torch's ``label_smoothing``
convention: the uniform mass includes the true class).

Contract: logits ``(B, C)`` raw scores; targets int64 ``(B,)`` in ``[0, C)`` or ``ignore_index``.
``smoothing`` must be in [0, 1); 0 equals plain cross-entropy. Optional ``class_weights (C,)``
follow torch's weighting semantics for ``F.cross_entropy``.

Requires: torch.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch import nn


def validate_smoothing(smoothing: float) -> float:
    if not 0.0 <= smoothing < 1.0:
        raise ValueError(f"smoothing must be in [0, 1), got {smoothing}")
    return float(smoothing)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(
        self,
        smoothing: float = 0.1,
        class_weights: Optional[Union[torch.Tensor, Sequence[float]]] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.smoothing = validate_smoothing(smoothing)
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum' or 'none', got {reduction!r}")
        if class_weights is not None:
            w = torch.as_tensor(class_weights, dtype=torch.float32)
            if w.dim() != 1 or torch.any(w < 0):
                raise ValueError("class_weights must be a non-negative 1D tensor")
            self.register_buffer("class_weights", w)
        else:
            self.class_weights = None
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.dim() != 2:
            raise ValueError(f"logits must be (B, C), got shape {tuple(logits.shape)}")
        if targets.shape != logits.shape[:1] or targets.dtype != torch.long:
            raise ValueError("targets must be int64 with shape (B,)")
        weight = None if self.class_weights is None else self.class_weights.to(logits.dtype)
        if self.reduction == "mean" and bool((targets == self.ignore_index).all()):
            return logits.sum() * 0.0
        return F.cross_entropy(
            logits,
            targets,
            weight=weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.smoothing,
        )


def label_smoothed_nll_reference(
    logits: torch.Tensor, targets: torch.Tensor, smoothing: float
) -> torch.Tensor:
    """Explicit unweighted mean computation of the formula above (for teaching/tests).

    No ignore_index handling; equals ``F.cross_entropy(..., label_smoothing=smoothing)``.
    """
    smoothing = validate_smoothing(smoothing)
    log_p = F.log_softmax(logits, dim=-1)
    nll = -log_p.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    uniform = -log_p.mean(dim=-1)
    return ((1.0 - smoothing) * nll + smoothing * uniform).mean()
