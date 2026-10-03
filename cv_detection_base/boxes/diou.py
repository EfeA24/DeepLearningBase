# Purpose: Computes Distance IoU (pairwise and aligned) with a center-distance penalty for detection box regression and NMS variants.
"""Distance IoU (Zheng et al., 2020) for xyxy boxes.

DIoU = IoU - rho^2(b, b_gt) / c^2, where rho is the Euclidean distance between
box centers and c is the diagonal length of the smallest enclosing box.
Range: (-1, 1]. Loss form: ``1 - diou``.

Contract:
    boxes1: (N, 4), boxes2: (M, 4), xyxy, consistent units.
    - ``distance_box_iou`` -> (N, M); ``aligned_distance_box_iou`` -> (N,).
    Empty inputs return zero-sized tensors. ``eps`` is added to c^2 and guards
    the union. Half-precision inputs are computed in float32. Self-contained.

Requires: torch only.
"""
from __future__ import annotations

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


def elementwise_diou(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tensor:
    """DIoU of broadcastable (..., 4) xyxy tensors -> broadcast shape (...)."""
    area1 = (boxes1[..., 2] - boxes1[..., 0]).clamp(min=0) * (boxes1[..., 3] - boxes1[..., 1]).clamp(min=0)
    area2 = (boxes2[..., 2] - boxes2[..., 0]).clamp(min=0) * (boxes2[..., 3] - boxes2[..., 1]).clamp(min=0)
    inter_wh = (torch.minimum(boxes1[..., 2:], boxes2[..., 2:]) - torch.maximum(boxes1[..., :2], boxes2[..., :2])).clamp(min=0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]
    union = area1 + area2 - inter
    iou = inter / union.clamp(min=eps)

    enclose_wh = (torch.maximum(boxes1[..., 2:], boxes2[..., 2:]) - torch.minimum(boxes1[..., :2], boxes2[..., :2])).clamp(min=0)
    diagonal_sq = enclose_wh[..., 0] ** 2 + enclose_wh[..., 1] ** 2 + eps
    center1 = (boxes1[..., :2] + boxes1[..., 2:]) * 0.5
    center2 = (boxes2[..., :2] + boxes2[..., 2:]) * 0.5
    center_dist_sq = ((center1 - center2) ** 2).sum(dim=-1)
    return iou - center_dist_sq / diagonal_sq


def distance_box_iou(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tensor:
    """Pairwise DIoU between (N, 4) and (M, 4) xyxy boxes -> (N, M)."""
    _check_pair(boxes1, boxes2)
    return elementwise_diou(_upcast(boxes1)[:, None, :], _upcast(boxes2)[None, :, :], eps)


def aligned_distance_box_iou(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tensor:
    """Element-wise DIoU between matching rows of two (N, 4) xyxy tensors -> (N,)."""
    _check_pair(boxes1, boxes2)
    if boxes1.shape[0] != boxes2.shape[0]:
        raise ValueError(
            f"aligned DIoU needs the same number of boxes, got {boxes1.shape[0]} and {boxes2.shape[0]}"
        )
    return elementwise_diou(_upcast(boxes1), _upcast(boxes2), eps)
