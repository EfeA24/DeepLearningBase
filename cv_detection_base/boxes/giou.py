# Purpose: Computes Generalized IoU (pairwise and aligned) for box regression losses and matching in detection projects.
"""Generalized IoU (Rezatofighi et al., 2019) for xyxy boxes.

GIoU = IoU - (|C| - |A u B|) / |C|, where C is the smallest enclosing box.
Range: [-1, 1]; equals IoU when one box encloses the other, approaches -1 for
far-apart small boxes. Loss form: ``1 - giou``.

Contract:
    boxes1: (N, 4), boxes2: (M, 4), xyxy, consistent units.
    - ``generalized_box_iou`` -> (N, M); ``aligned_generalized_box_iou`` -> (N,).
    Empty inputs return correspondingly zero-sized tensors. ``eps`` guards unions
    and enclosing areas. Half-precision inputs are computed in float32.
    Self-contained (IoU math duplicated from boxes/iou.py so the file can be copied alone).

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


def elementwise_giou(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tensor:
    """GIoU of broadcastable (..., 4) xyxy tensors -> broadcast shape (...)."""
    area1 = (boxes1[..., 2] - boxes1[..., 0]).clamp(min=0) * (boxes1[..., 3] - boxes1[..., 1]).clamp(min=0)
    area2 = (boxes2[..., 2] - boxes2[..., 0]).clamp(min=0) * (boxes2[..., 3] - boxes2[..., 1]).clamp(min=0)
    inter_wh = (torch.minimum(boxes1[..., 2:], boxes2[..., 2:]) - torch.maximum(boxes1[..., :2], boxes2[..., :2])).clamp(min=0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]
    union = area1 + area2 - inter
    iou = inter / union.clamp(min=eps)

    enclose_wh = (torch.maximum(boxes1[..., 2:], boxes2[..., 2:]) - torch.minimum(boxes1[..., :2], boxes2[..., :2])).clamp(min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    return iou - (enclose_area - union) / enclose_area.clamp(min=eps)


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tensor:
    """Pairwise GIoU between (N, 4) and (M, 4) xyxy boxes -> (N, M) in [-1, 1]."""
    _check_pair(boxes1, boxes2)
    return elementwise_giou(_upcast(boxes1)[:, None, :], _upcast(boxes2)[None, :, :], eps)


def aligned_generalized_box_iou(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tensor:
    """Element-wise GIoU between matching rows of two (N, 4) xyxy tensors -> (N,)."""
    _check_pair(boxes1, boxes2)
    if boxes1.shape[0] != boxes2.shape[0]:
        raise ValueError(
            f"aligned GIoU needs the same number of boxes, got {boxes1.shape[0]} and {boxes2.shape[0]}"
        )
    return elementwise_giou(_upcast(boxes1), _upcast(boxes2), eps)
