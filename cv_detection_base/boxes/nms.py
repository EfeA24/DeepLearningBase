# Purpose: Provides reference non-maximum suppression (pure torch plus optional torchvision backend) and a class-aware batched variant for detection post-processing.
"""Non-maximum suppression (NMS) for xyxy boxes.

Contract:
    boxes: (N, 4) float xyxy (any consistent unit), scores: (N,) float,
    labels: (N,) int64 for the class-aware variant.
    Returns int64 indices into the ORIGINAL inputs, sorted by decreasing score.
    A box is suppressed when its IoU with an already kept box is strictly greater
    than ``iou_threshold`` (torchvision convention). ``score_threshold`` (optional)
    drops boxes with score <= threshold before suppression.
    Empty inputs return an empty int64 tensor on the input device.

Backends:
    "torch"       - reference greedy loop in pure torch (O(N^2) worst case, fine
                    for a few thousand boxes; deterministic for tied scores via stable sort).
    "torchvision" - ``torchvision.ops.nms`` imported lazily (fast C++/CUDA kernel).

Requires: torch; torchvision optional (only for backend="torchvision").
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def _validate(boxes: Tensor, scores: Tensor, iou_threshold: float) -> None:
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError(f"boxes must have shape (N, 4), got {tuple(boxes.shape)}")
    if scores.ndim != 1 or scores.shape[0] != boxes.shape[0]:
        raise ValueError(f"scores must have shape ({boxes.shape[0]},), got {tuple(scores.shape)}")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError(f"iou_threshold must be in [0, 1], got {iou_threshold}")


def _iou_one_to_many(box: Tensor, others: Tensor, eps: float = 1e-7) -> Tensor:
    area = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
    areas = (others[:, 2] - others[:, 0]).clamp(min=0) * (others[:, 3] - others[:, 1]).clamp(min=0)
    left_top = torch.maximum(box[:2], others[:, :2])
    right_bottom = torch.minimum(box[2:], others[:, 2:])
    wh = (right_bottom - left_top).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    return inter / (area + areas - inter).clamp(min=eps)


def _nms_torch(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    boxes = boxes.float() if boxes.dtype in (torch.float16, torch.bfloat16) else boxes
    order = torch.sort(scores, descending=True, stable=True).indices
    keep = []
    while order.numel() > 0:
        current = order[0]
        keep.append(current)
        if order.numel() == 1:
            break
        rest = order[1:]
        ious = _iou_one_to_many(boxes[current], boxes[rest])
        order = rest[ious <= iou_threshold]
    if not keep:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    return torch.stack(keep).to(torch.long)


def _nms_torchvision(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    try:
        from torchvision.ops import nms as tv_nms
    except ImportError as exc:
        raise ImportError("Install torchvision to use backend='torchvision': pip install torchvision") from exc
    return tv_nms(boxes, scores, iou_threshold)


def nms(
    boxes: Tensor,
    scores: Tensor,
    iou_threshold: float,
    score_threshold: Optional[float] = None,
    max_output: Optional[int] = None,
    backend: str = "torch",
) -> Tensor:
    """Class-agnostic NMS.

    Args:
        boxes: (N, 4) xyxy boxes.
        scores: (N,) confidence scores.
        iou_threshold: Suppress boxes whose IoU with a kept box is > this value.
        score_threshold: Optional pre-filter; keeps only scores > threshold.
        max_output: Optional cap on the number of returned indices.
        backend: "torch" (reference) or "torchvision" (lazy import).

    Returns:
        (K,) int64 indices into the original ``boxes``, sorted by decreasing score.
    """
    _validate(boxes, scores, iou_threshold)
    if max_output is not None and max_output < 0:
        raise ValueError(f"max_output must be >= 0, got {max_output}")
    candidate_idx = torch.arange(boxes.shape[0], device=boxes.device)
    if score_threshold is not None:
        mask = scores > score_threshold
        candidate_idx = candidate_idx[mask]
        boxes, scores = boxes[mask], scores[mask]
    if boxes.shape[0] == 0:
        return torch.empty(0, dtype=torch.long, device=candidate_idx.device)

    if backend == "torch":
        keep = _nms_torch(boxes, scores, iou_threshold)
    elif backend == "torchvision":
        keep = _nms_torchvision(boxes, scores, iou_threshold)
    else:
        raise ValueError(f"Unknown NMS backend {backend!r}; expected 'torch' or 'torchvision'")

    keep = candidate_idx[keep]
    if max_output is not None:
        keep = keep[:max_output]
    return keep


def batched_nms(
    boxes: Tensor,
    scores: Tensor,
    labels: Tensor,
    iou_threshold: float,
    score_threshold: Optional[float] = None,
    max_output: Optional[int] = None,
    backend: str = "torch",
) -> Tensor:
    """Class-aware NMS: boxes only suppress boxes with the same label.

    Uses the coordinate-offset trick: each class is shifted by
    ``label * (coordinate_range + 1)`` so boxes of different classes never overlap.
    The offset uses the full coordinate range, so negative coordinates are safe.

    Args:
        boxes: (N, 4) xyxy boxes.
        scores: (N,) scores.
        labels: (N,) integer class ids (>= 0).
        iou_threshold, score_threshold, max_output, backend: see ``nms``.

    Returns:
        (K,) int64 indices into the original inputs, sorted by decreasing score.
    """
    _validate(boxes, scores, iou_threshold)
    if labels.ndim != 1 or labels.shape[0] != boxes.shape[0]:
        raise ValueError(f"labels must have shape ({boxes.shape[0]},), got {tuple(labels.shape)}")
    if labels.is_floating_point():
        raise TypeError("labels must be an integer tensor")
    if boxes.shape[0] == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    work_boxes = boxes.float() if boxes.dtype in (torch.float16, torch.bfloat16) else boxes
    coord_range = work_boxes.max() - work_boxes.min()
    offsets = labels.to(work_boxes.dtype) * (coord_range + 1.0)
    shifted = work_boxes + offsets[:, None]
    return nms(shifted, scores, iou_threshold, score_threshold, max_output, backend)
