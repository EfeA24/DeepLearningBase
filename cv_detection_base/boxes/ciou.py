# Purpose: Computes Complete IoU (pairwise and aligned) with center-distance and aspect-ratio terms for detection box regression losses.
"""Complete IoU (Zheng et al., 2020) for xyxy boxes.

CIoU = IoU - rho^2 / c^2 - alpha * v
    v     = (4 / pi^2) * (atan(w_gt / h_gt) - atan(w / h))^2
    alpha = v / ((1 - IoU) + v)

Following the paper (and common implementations), ``alpha`` is treated as a
constant: it is computed under ``torch.no_grad()`` so gradients flow only
through IoU, the distance term and ``v``. Range: roughly [-1.5, 1]. Loss form:
``1 - ciou``. The value is symmetric in boxes1/boxes2.

Contract:
    boxes1: (N, 4), boxes2: (M, 4), xyxy, consistent units.
    - ``complete_box_iou`` -> (N, M); ``aligned_complete_box_iou`` -> (N,).
    Empty inputs return zero-sized tensors. ``eps`` is added to heights (atan),
    to c^2 and to the alpha denominator. Half precision is computed in float32.
    Self-contained.

Requires: torch only.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor


def _check_pair(boxes1: Tensor, boxes2: Tensor) -> None:
    for name, boxes in (("boxes1", boxes1), ("boxes2", boxes2)):
        if not isinstance(boxes, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if boxes.ndim != 2 or boxes.shape[-1] != 4:
            raise ValueError(f"{name} must have shape (N, 4), got {tuple(boxes.shape)}")


def _upcast(boxes: Tensor) -> Tensor:
    if boxes.dtype in (torch.float16, torch.bfloat16) or not boxes.is_floating_point():
        return boxes.float()
    return boxes


def elementwise_ciou(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tensor:
    """CIoU of broadcastable (..., 4) xyxy tensors -> broadcast shape (...)."""
    w1 = (boxes1[..., 2] - boxes1[..., 0]).clamp(min=0)
    h1 = (boxes1[..., 3] - boxes1[..., 1]).clamp(min=0)
    w2 = (boxes2[..., 2] - boxes2[..., 0]).clamp(min=0)
    h2 = (boxes2[..., 3] - boxes2[..., 1]).clamp(min=0)

    inter_wh = (torch.minimum(boxes1[..., 2:], boxes2[..., 2:]) - torch.maximum(boxes1[..., :2], boxes2[..., :2])).clamp(min=0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]
    union = w1 * h1 + w2 * h2 - inter
    iou = inter / union.clamp(min=eps)

    enclose_wh = (torch.maximum(boxes1[..., 2:], boxes2[..., 2:]) - torch.minimum(boxes1[..., :2], boxes2[..., :2])).clamp(min=0)
    diagonal_sq = enclose_wh[..., 0] ** 2 + enclose_wh[..., 1] ** 2 + eps
    center1 = (boxes1[..., :2] + boxes1[..., 2:]) * 0.5
    center2 = (boxes2[..., :2] + boxes2[..., 2:]) * 0.5
    center_dist_sq = ((center1 - center2) ** 2).sum(dim=-1)

    v = (4.0 / math.pi**2) * (torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))) ** 2
    with torch.no_grad():
        alpha = v / ((1.0 - iou) + v + eps)
    return iou - center_dist_sq / diagonal_sq - alpha * v


def complete_box_iou(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tensor:
    """Pairwise CIoU between (N, 4) and (M, 4) xyxy boxes -> (N, M)."""
    _check_pair(boxes1, boxes2)
    return elementwise_ciou(_upcast(boxes1)[:, None, :], _upcast(boxes2)[None, :, :], eps)


def aligned_complete_box_iou(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tensor:
    """Element-wise CIoU between matching rows of two (N, 4) xyxy tensors -> (N,)."""
    _check_pair(boxes1, boxes2)
    if boxes1.shape[0] != boxes2.shape[0]:
        raise ValueError(
            f"aligned CIoU needs the same number of boxes, got {boxes1.shape[0]} and {boxes2.shape[0]}"
        )
    return elementwise_ciou(_upcast(boxes1), _upcast(boxes2), eps)
