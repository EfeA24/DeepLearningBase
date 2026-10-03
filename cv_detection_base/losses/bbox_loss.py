# Purpose: Provides a selectable box-regression loss (L1, SmoothL1, GIoU, DIoU, CIoU) over matched prediction/target pairs for object-detection training.
"""Box regression loss adapter.

Input contract (matched pairs only; assignment is done by the caller):
    pred, target: FloatTensor (N, 4)
      - "l1" / "smooth_l1": any consistent encoding (e.g. anchor deltas from
        models/anchors/anchor_utils.py, or xyxy pixels). Element-wise loss is
        summed over the 4 coordinates per box.
      - "giou" / "diou" / "ciou": decoded xyxy boxes in the same units
        (pixels or normalized). Per-box loss = 1 - IoU-variant.
    weights: optional (N,) per-box weights (e.g. mixup weights or IoU-aware weights).
    ``target`` is treated as a constant (detached); computation runs in float32.

Reduction:
    "mean" divides the (weighted) sum by ``normalizer`` if given (e.g. number of
    positives across the batch, possibly all-reduced), else by N (or by the sum
    of weights when weights are given). "sum" and "none" ((N,) per-box) also available.
    Empty input (N = 0) returns a zero that still depends on ``pred`` so autograd works.

IoU math is self-contained (copied from boxes/*.py) to keep the file copyable.

Requires: torch only.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

LOSS_TYPES = ("l1", "smooth_l1", "giou", "diou", "ciou")


def _iou_family(pred: Tensor, target: Tensor, kind: str, eps: float) -> Tensor:
    """Aligned IoU variants of (N, 4) xyxy tensors -> (N,)."""
    pw = (pred[:, 2] - pred[:, 0]).clamp(min=0)
    ph = (pred[:, 3] - pred[:, 1]).clamp(min=0)
    tw = (target[:, 2] - target[:, 0]).clamp(min=0)
    th = (target[:, 3] - target[:, 1]).clamp(min=0)
    inter_wh = (torch.minimum(pred[:, 2:], target[:, 2:]) - torch.maximum(pred[:, :2], target[:, :2])).clamp(min=0)
    inter = inter_wh[:, 0] * inter_wh[:, 1]
    union = pw * ph + tw * th - inter
    iou = inter / union.clamp(min=eps)
    enclose_wh = (torch.maximum(pred[:, 2:], target[:, 2:]) - torch.minimum(pred[:, :2], target[:, :2])).clamp(min=0)
    if kind == "giou":
        enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1]
        return iou - (enclose_area - union) / enclose_area.clamp(min=eps)
    diagonal_sq = enclose_wh[:, 0] ** 2 + enclose_wh[:, 1] ** 2 + eps
    center_dist_sq = (((pred[:, :2] + pred[:, 2:]) - (target[:, :2] + target[:, 2:])) ** 2).sum(dim=1) / 4.0
    diou = iou - center_dist_sq / diagonal_sq
    if kind == "diou":
        return diou
    v = (4.0 / math.pi**2) * (torch.atan(tw / (th + eps)) - torch.atan(pw / (ph + eps))) ** 2
    with torch.no_grad():
        alpha = v / ((1.0 - iou) + v + eps)
    return diou - alpha * v


class BoxRegressionLoss(nn.Module):
    """Selectable box regression loss. See module docstring for contracts.

    Args:
        loss_type: One of "l1", "smooth_l1", "giou", "diou", "ciou".
        reduction: "mean", "sum" or "none".
        beta: SmoothL1 transition point (> 0).
        eps: Numerical guard for IoU variants.
    """

    def __init__(self, loss_type: str = "giou", reduction: str = "mean", beta: float = 1.0 / 9.0, eps: float = 1e-7) -> None:
        super().__init__()
        if loss_type not in LOSS_TYPES:
            raise ValueError(f"loss_type must be one of {LOSS_TYPES}, got {loss_type!r}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be 'mean', 'sum' or 'none'")
        if beta <= 0:
            raise ValueError("beta must be > 0")
        self.loss_type = loss_type
        self.reduction = reduction
        self.beta = beta
        self.eps = eps

    def per_box(self, pred: Tensor, target: Tensor) -> Tensor:
        """Unreduced loss per box (N,)."""
        if pred.shape != target.shape or pred.ndim != 2 or pred.shape[-1] != 4:
            raise ValueError(f"pred and target must both be (N, 4), got {tuple(pred.shape)} and {tuple(target.shape)}")
        pred_f = pred.float()
        target_f = target.float().detach()
        if self.loss_type == "l1":
            return F.l1_loss(pred_f, target_f, reduction="none").sum(dim=1)
        if self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(pred_f, target_f, reduction="none", beta=self.beta).sum(dim=1)
        return 1.0 - _iou_family(pred_f, target_f, self.loss_type, self.eps)

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        weights: Optional[Tensor] = None,
        normalizer: Optional[float] = None,
    ) -> Tensor:
        losses = self.per_box(pred, target)
        if weights is not None:
            if weights.shape != losses.shape:
                raise ValueError(f"weights must be ({losses.shape[0]},), got {tuple(weights.shape)}")
            losses = losses * weights.to(losses.dtype)
        if self.reduction == "none":
            return losses
        if losses.numel() == 0:
            return pred.sum() * 0.0
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
