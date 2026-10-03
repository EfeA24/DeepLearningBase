# Purpose: Computes pairwise (N, M) and aligned (N,) intersection-over-union of axis-aligned boxes for detection training and matching.
"""Intersection over union (IoU) for xyxy boxes.

Contract:
    boxes1: (N, 4), boxes2: (M, 4), xyxy format, any consistent unit (pixels or
    normalized). Boxes with x2 < x1 or y2 < y1 are treated as zero-area.
    - ``box_iou(boxes1, boxes2)`` -> (N, M) pairwise IoU in [0, 1].
    - ``aligned_box_iou(boxes1, boxes2)`` -> (N,) IoU of row i with row i (N == M).
    Empty inputs: pairwise returns an (N, M) tensor with a zero-sized dimension;
    aligned returns shape (0,). Unions are clamped to ``eps`` so two degenerate
    boxes give IoU 0 instead of NaN. Half-precision inputs are computed in float32.
    Gradients flow through these functions (usable inside losses).

Requires: torch only.
"""
from __future__ import annotations

from typing import Tuple

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


def elementwise_inter_union(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
    """Intersection and union areas of broadcastable (..., 4) xyxy box tensors."""
    area1 = (boxes1[..., 2] - boxes1[..., 0]).clamp(min=0) * (boxes1[..., 3] - boxes1[..., 1]).clamp(min=0)
    area2 = (boxes2[..., 2] - boxes2[..., 0]).clamp(min=0) * (boxes2[..., 3] - boxes2[..., 1]).clamp(min=0)
    left_top = torch.maximum(boxes1[..., :2], boxes2[..., :2])
    right_bottom = torch.minimum(boxes1[..., 2:], boxes2[..., 2:])
    wh = (right_bottom - left_top).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1 + area2 - inter
    return inter, union


def box_iou(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tensor:
    """Pairwise IoU between (N, 4) and (M, 4) xyxy boxes -> (N, M)."""
    _check_pair(boxes1, boxes2)
    b1 = _upcast(boxes1)[:, None, :]
    b2 = _upcast(boxes2)[None, :, :]
    inter, union = elementwise_inter_union(b1, b2)
    return inter / union.clamp(min=eps)


def aligned_box_iou(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tensor:
    """Element-wise IoU between matching rows of two (N, 4) xyxy tensors -> (N,)."""
    _check_pair(boxes1, boxes2)
    if boxes1.shape[0] != boxes2.shape[0]:
        raise ValueError(
            f"aligned IoU needs the same number of boxes, got {boxes1.shape[0]} and {boxes2.shape[0]}"
        )
    inter, union = elementwise_inter_union(_upcast(boxes1), _upcast(boxes2))
    return inter / union.clamp(min=eps)
