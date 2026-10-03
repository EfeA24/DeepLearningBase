# Purpose: Computes objectness BCE-with-logits loss over explicit positive and negative masks for dense object-detection heads.
"""Objectness loss.

Input contract:
    logits:        FloatTensor (...) raw objectness logits (e.g. (B, P) after flattening).
    targets:       FloatTensor (...) in [0, 1]; 1 for hard positives or a soft
                   quality score (e.g. IoU) at positives. Values at negatives are
                   forced to 0; values elsewhere are ignored.
    positive_mask: BoolTensor (...) True where a prediction was assigned to an object.
    negative_mask: BoolTensor (...) True where a prediction is background.
                   Positions in neither mask are IGNORED (e.g. IoU between ignore thresholds).
    positive_mask & negative_mask must not overlap.

Normalization ("mean" reduction):
    "num_positives" (default): sum / max(#positives, 1)  - common for dense detectors
    "num_valid":               sum / max(#pos + #neg, 1)
    ``normalizer`` argument overrides both (e.g. positives all-reduced across ranks).

Requires: torch only.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ObjectnessLoss(nn.Module):
    """BCE-with-logits objectness loss with explicit assignment masks.

    Args:
        pos_weight: Optional scalar weight for positive terms (BCE ``pos_weight``).
        negative_weight: Multiplier on negative terms (down-weight easy background).
        reduction: "mean" (normalized, see module docstring) or "sum".
        normalize_by: "num_positives" or "num_valid".
    """

    def __init__(
        self,
        pos_weight: Optional[float] = None,
        negative_weight: float = 1.0,
        reduction: str = "mean",
        normalize_by: str = "num_positives",
    ) -> None:
        super().__init__()
        if reduction not in ("mean", "sum"):
            raise ValueError("reduction must be 'mean' or 'sum'")
        if normalize_by not in ("num_positives", "num_valid"):
            raise ValueError("normalize_by must be 'num_positives' or 'num_valid'")
        if negative_weight < 0 or (pos_weight is not None and pos_weight <= 0):
            raise ValueError("negative_weight must be >= 0 and pos_weight > 0")
        self.pos_weight = pos_weight
        self.negative_weight = negative_weight
        self.reduction = reduction
        self.normalize_by = normalize_by

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        positive_mask: Tensor,
        negative_mask: Tensor,
        normalizer: Optional[float] = None,
    ) -> Tensor:
        if not (logits.shape == targets.shape == positive_mask.shape == negative_mask.shape):
            raise ValueError("logits, targets, positive_mask and negative_mask must share a shape")
        if positive_mask.dtype != torch.bool or negative_mask.dtype != torch.bool:
            raise TypeError("masks must be bool tensors")
        if (positive_mask & negative_mask).any():
            raise ValueError("positive_mask and negative_mask overlap")

        logits_f = logits.float()
        target_f = torch.where(positive_mask, targets.float().clamp(0.0, 1.0), torch.zeros_like(logits_f)).detach()
        pos_weight = None if self.pos_weight is None else logits_f.new_tensor(self.pos_weight)
        loss = F.binary_cross_entropy_with_logits(logits_f, target_f, reduction="none", pos_weight=pos_weight)
        weight = positive_mask.to(loss.dtype) + negative_mask.to(loss.dtype) * self.negative_weight
        total = (loss * weight).sum()
        if self.reduction == "sum":
            return total
        if normalizer is None:
            count = positive_mask.sum() if self.normalize_by == "num_positives" else (positive_mask | negative_mask).sum()
            normalizer = float(count)
        return total / max(float(normalizer), 1.0)
