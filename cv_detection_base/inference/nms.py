# Purpose: Provides an inference-facing per-image batched NMS adapter with score filtering, class-aware/agnostic modes and max-detection caps for detection post-processing.
"""Inference NMS adapter.

Geometry lives in boxes/nms.py; this module only handles batching, score
filtering, top-k and output formatting. The NMS kernel is injectable:

    nms_fn(boxes (N, 4), scores (N,), labels (N,), iou_threshold) -> keep (K,) int64
        (indices sorted by decreasing score, class-aware on ``labels``)

If ``nms_fn`` is None, ``boxes.nms.batched_nms`` is imported lazily (requires the
template root on ``sys.path``). ``torchvision.ops.batched_nms`` has the same
signature and can be passed directly.

Contract:
    boxes:  (B, N, 4) xyxy tensor or list of B tensors (N_i, 4)
    scores: (B, N) or list of (N_i,)
    labels: (B, N) int64 or list of (N_i,)
    Returns a list of B dicts {"boxes": (K_i, 4), "scores": (K_i,), "labels": (K_i,)},
    sorted by decreasing score, K_i <= max_detections. Empty images give K_i = 0.

Requires: torch; sibling boxes/nms.py when ``nms_fn`` is not supplied.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Union

import torch
from torch import Tensor

NmsFn = Callable[[Tensor, Tensor, Tensor, float], Tensor]


def _default_nms_fn() -> NmsFn:
    try:
        from boxes.nms import batched_nms
    except ImportError as exc:
        raise ImportError(
            "No nms_fn supplied and boxes/nms.py is not importable; add the template root to sys.path "
            "or pass nms_fn (e.g. torchvision.ops.batched_nms)"
        ) from exc
    return lambda b, s, l, t: batched_nms(b, s, l, t)


def _as_list(value: Union[Tensor, Sequence[Tensor]], name: str) -> List[Tensor]:
    if isinstance(value, Tensor):
        return list(value.unbind(0))
    if isinstance(value, (list, tuple)):
        return list(value)
    raise TypeError(f"{name} must be a Tensor or a list of Tensors")


def nms_single_image(
    boxes: Tensor,
    scores: Tensor,
    labels: Tensor,
    iou_threshold: float,
    score_threshold: Optional[float] = None,
    max_detections: Optional[int] = 100,
    pre_nms_top_k: Optional[int] = None,
    class_agnostic: bool = False,
    nms_fn: Optional[NmsFn] = None,
) -> Dict[str, Tensor]:
    """Filter, suppress and cap detections of one image. See module docstring."""
    if boxes.ndim != 2 or boxes.shape[-1] != 4 or scores.shape != (boxes.shape[0],) or labels.shape != (boxes.shape[0],):
        raise ValueError("expected boxes (N, 4), scores (N,), labels (N,)")
    if score_threshold is not None:
        keep = scores > score_threshold
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    if pre_nms_top_k is not None and scores.numel() > pre_nms_top_k:
        top = torch.topk(scores, pre_nms_top_k).indices
        boxes, scores, labels = boxes[top], scores[top], labels[top]
    if boxes.shape[0] == 0:
        return {"boxes": boxes.reshape(0, 4), "scores": scores, "labels": labels.long()}
    kernel = nms_fn or _default_nms_fn()
    nms_labels = torch.zeros_like(labels) if class_agnostic else labels
    keep = kernel(boxes, scores, nms_labels.long(), iou_threshold)
    if max_detections is not None:
        keep = keep[:max_detections]
    return {"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep].long()}


def batched_image_nms(
    boxes: Union[Tensor, Sequence[Tensor]],
    scores: Union[Tensor, Sequence[Tensor]],
    labels: Union[Tensor, Sequence[Tensor]],
    iou_threshold: float,
    score_threshold: Optional[float] = None,
    max_detections: Optional[int] = 100,
    pre_nms_top_k: Optional[int] = None,
    class_agnostic: bool = False,
    nms_fn: Optional[NmsFn] = None,
) -> List[Dict[str, Tensor]]:
    """Run ``nms_single_image`` independently for every image in the batch."""
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    if max_detections is not None and max_detections <= 0:
        raise ValueError("max_detections must be > 0")
    boxes_l, scores_l, labels_l = _as_list(boxes, "boxes"), _as_list(scores, "scores"), _as_list(labels, "labels")
    if not (len(boxes_l) == len(scores_l) == len(labels_l)):
        raise ValueError("boxes, scores and labels must have the same batch size")
    kernel = nms_fn or _default_nms_fn()
    return [
        nms_single_image(b, s, l, iou_threshold, score_threshold, max_detections, pre_nms_top_k, class_agnostic, kernel)
        for b, s, l in zip(boxes_l, scores_l, labels_l)
    ]
