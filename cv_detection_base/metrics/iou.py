# Purpose: Provides no-grad box-overlap metric helpers (pairwise IoU, best-match IoU, class-aware IoU) for detection evaluation, separate from training IoU losses.
"""Detection overlap metrics (evaluation only).

All functions run under ``torch.no_grad()``, compute in float32 (float64 inputs
stay float64) and are intended for evaluation/matching, not for losses (use
boxes/*.py or losses/bbox_loss.py for differentiable variants).

Contract:
    Boxes are (N, 4) / (M, 4) xyxy in consistent units.
    - ``pairwise_iou``        -> (N, M); zero-sized if N or M is 0.
    - ``best_match_iou``      -> (M,) best IoU of each ground-truth box with any
                                 prediction (0 where there are no predictions).
    - ``class_aware_iou``     -> (N, M) IoU, set to 0 where labels differ.
    - ``mean_best_match_iou`` -> float mean of ``best_match_iou`` (NaN if M == 0).

Requires: torch only.
"""
from __future__ import annotations

import torch
from torch import Tensor


def _check(boxes: Tensor, name: str) -> None:
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError(f"{name} must have shape (N, 4), got {tuple(boxes.shape)}")


@torch.no_grad()
def pairwise_iou(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-9) -> Tensor:
    """IoU matrix (N, M) between xyxy boxes."""
    _check(boxes1, "boxes1")
    _check(boxes2, "boxes2")
    dtype = torch.float64 if torch.float64 in (boxes1.dtype, boxes2.dtype) else torch.float32
    b1 = boxes1.to(dtype)
    b2 = boxes2.to(dtype)
    area1 = (b1[:, 2] - b1[:, 0]).clamp(min=0) * (b1[:, 3] - b1[:, 1]).clamp(min=0)
    area2 = (b2[:, 2] - b2[:, 0]).clamp(min=0) * (b2[:, 3] - b2[:, 1]).clamp(min=0)
    left_top = torch.maximum(b1[:, None, :2], b2[None, :, :2])
    right_bottom = torch.minimum(b1[:, None, 2:], b2[None, :, 2:])
    wh = (right_bottom - left_top).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=eps)


@torch.no_grad()
def class_aware_iou(boxes1: Tensor, labels1: Tensor, boxes2: Tensor, labels2: Tensor) -> Tensor:
    """IoU matrix (N, M) with entries zeroed where class labels differ."""
    iou = pairwise_iou(boxes1, boxes2)
    if labels1.shape != (boxes1.shape[0],) or labels2.shape != (boxes2.shape[0],):
        raise ValueError("labels must be 1-D and match their boxes")
    same = labels1[:, None].to(labels2.device) == labels2[None, :]
    return iou * same.to(iou.dtype)


@torch.no_grad()
def best_match_iou(pred_boxes: Tensor, gt_boxes: Tensor) -> Tensor:
    """Best IoU per ground-truth box (M,); 0 for every GT when there are no predictions."""
    _check(gt_boxes, "gt_boxes")
    if pred_boxes.shape[0] == 0:
        return torch.zeros(gt_boxes.shape[0], dtype=torch.float32, device=gt_boxes.device)
    if gt_boxes.shape[0] == 0:
        return torch.zeros(0, dtype=torch.float32, device=gt_boxes.device)
    return pairwise_iou(pred_boxes, gt_boxes).max(dim=0).values


@torch.no_grad()
def mean_best_match_iou(pred_boxes: Tensor, gt_boxes: Tensor) -> float:
    """Mean best-match IoU over GT boxes (localization quality); NaN if there are no GT boxes."""
    best = best_match_iou(pred_boxes, gt_boxes)
    return float("nan") if best.numel() == 0 else float(best.mean())
