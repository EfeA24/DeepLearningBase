# Purpose: Wraps BCEWithLogits for binary and multi-label image classification with validated targets and optional pos_weight/element weights.
"""Binary / multi-label BCE-with-logits loss.

Contract:
    logits: raw scores, ``(B,)`` for binary or ``(B, L)`` for multi-label (no sigmoid applied).
    targets: same shape as logits, float values in [0, 1] (soft labels allowed); int/bool
        targets are converted to float.
    pos_weight: optional ``(L,)`` (or scalar for binary) multiplier on the positive term, e.g.
        ``num_negatives / num_positives`` per label to counter imbalance.
    weight: optional per-element rescaling broadcastable to the logits shape (e.g. ``(L,)``).

Requires: torch.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch import nn

TensorLike = Union[torch.Tensor, Sequence[float], float]


class BCEWithLogitsClassificationLoss(nn.Module):
    def __init__(
        self,
        pos_weight: Optional[TensorLike] = None,
        weight: Optional[TensorLike] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum' or 'none', got {reduction!r}")
        if pos_weight is not None:
            pw = torch.as_tensor(pos_weight, dtype=torch.float32)
            if torch.any(pw < 0):
                raise ValueError("pos_weight must be non-negative")
            self.register_buffer("pos_weight", pw)
        else:
            self.pos_weight = None
        if weight is not None:
            w = torch.as_tensor(weight, dtype=torch.float32)
            if torch.any(w < 0):
                raise ValueError("weight must be non-negative")
            self.register_buffer("weight", w)
        else:
            self.weight = None
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.shape != logits.shape:
            raise ValueError(
                f"targets shape {tuple(targets.shape)} must equal logits shape {tuple(logits.shape)}"
            )
        targets = targets.to(logits.dtype)
        if torch.any(targets < 0) or torch.any(targets > 1):
            raise ValueError("targets must be in [0, 1]")
        return F.binary_cross_entropy_with_logits(
            logits,
            targets,
            weight=None if self.weight is None else self.weight.to(logits.dtype),
            pos_weight=None if self.pos_weight is None else self.pos_weight.to(logits.dtype),
            reduction=self.reduction,
        )


def compute_pos_weight(multihot_targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-label ``num_negatives / num_positives`` from TRAINING targets ``(N, L)`` in {0, 1}.

    Labels with zero positives produce very large weights (``num_negatives / eps``); inspect or cap
    them explicitly before use.
    """
    if multihot_targets.dim() != 2:
        raise ValueError("multihot_targets must be (N, L)")
    t = multihot_targets.float()
    positives = t.sum(dim=0)
    negatives = t.shape[0] - positives
    return negatives / positives.clamp_min(eps)
