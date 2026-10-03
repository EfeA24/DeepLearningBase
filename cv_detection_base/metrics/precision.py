# Purpose: Accumulates detection true/false positives at explicit confidence and IoU thresholds and reports detection precision (micro and per class).
"""Detection precision at fixed thresholds.

Per image, predictions with score >= ``score_threshold`` are sorted by score
(descending) and greedily matched to the unmatched GT box with highest IoU >=
``iou_threshold`` (same class when ``class_aware=True``). Matched -> TP, else FP.
GT flagged by ``ignore_key`` (e.g. "difficult") cannot be matched; predictions
overlapping them (IoU >= threshold) are ignored (neither TP nor FP).

precision = TP / (TP + FP). With no counted predictions the result is
``zero_division`` (default 0.0; use float("nan") to mark "undefined").
Micro precision pools all classes; per-class values use the same policy.

Inputs follow metrics/map.py: predictions[i] {"boxes", "scores", "labels"},
targets[i] {"boxes", "labels", optional ignore flags}; boxes xyxy in matching units.

Requires: torch only.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor


def _pairwise_iou(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-9) -> Tensor:
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    wh = (torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:]) - torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area1[:, None] + area2[None, :] - inter).clamp(min=eps)


def greedy_match(
    p_boxes: Tensor,
    p_scores: Tensor,
    p_labels: Tensor,
    g_boxes: Tensor,
    g_labels: Tensor,
    g_ignore: Tensor,
    iou_threshold: float,
    class_aware: bool,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Greedy score-ordered matching for one image.

    Returns:
        (tp (P,) bool, ignored (P,) bool, gt_matched (G,) bool), in the input prediction order.
    """
    num_p, num_g = p_boxes.shape[0], g_boxes.shape[0]
    tp = torch.zeros(num_p, dtype=torch.bool)
    ignored = torch.zeros(num_p, dtype=torch.bool)
    matched = torch.zeros(num_g, dtype=torch.bool)
    if num_p == 0 or num_g == 0:
        return tp, ignored, matched
    iou = _pairwise_iou(p_boxes, g_boxes)
    if class_aware:
        iou = iou * (p_labels[:, None] == g_labels[None, :]).to(iou.dtype)
    for p_idx in torch.sort(p_scores, descending=True, stable=True).indices.tolist():
        candidates = iou[p_idx].clone()
        candidates[matched | g_ignore] = -1.0
        best = int(torch.argmax(candidates))
        if candidates[best] >= iou_threshold:
            matched[best] = True
            tp[p_idx] = True
        elif g_ignore.any() and bool((iou[p_idx][g_ignore] >= iou_threshold).any()):
            ignored[p_idx] = True
    return tp, ignored, matched


class DetectionPrecision:
    """TP/FP accumulator for detection precision.

    Args:
        num_classes: Number of classes (for per-class counts).
        iou_threshold: Minimum IoU for a match.
        score_threshold: Minimum score for a prediction to be counted.
        class_aware: Require equal labels for a match.
        ignore_key: Target key with per-GT ignore flags, or None.
        zero_division: Value returned when TP + FP == 0.
    """

    def __init__(
        self,
        num_classes: int,
        iou_threshold: float = 0.5,
        score_threshold: float = 0.5,
        class_aware: bool = True,
        ignore_key: Optional[str] = "difficult",
        zero_division: float = 0.0,
    ) -> None:
        if num_classes <= 0 or not 0.0 < iou_threshold <= 1.0:
            raise ValueError("num_classes must be > 0 and iou_threshold in (0, 1]")
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.class_aware = class_aware
        self.ignore_key = ignore_key
        self.zero_division = zero_division
        self.reset()

    def reset(self) -> None:
        self.tp = torch.zeros(self.num_classes, dtype=torch.long)
        self.fp = torch.zeros(self.num_classes, dtype=torch.long)

    @torch.no_grad()
    def update(self, predictions: Sequence[Mapping[str, Tensor]], targets: Sequence[Mapping[str, Any]]) -> None:
        if len(predictions) != len(targets):
            raise ValueError("predictions and targets must have the same length")
        for pred, target in zip(predictions, targets):
            keep = pred["scores"].detach().cpu().reshape(-1) >= self.score_threshold
            p_boxes = pred["boxes"].detach().cpu().double().reshape(-1, 4)[keep]
            p_scores = pred["scores"].detach().cpu().double().reshape(-1)[keep]
            p_labels = pred["labels"].detach().cpu().long().reshape(-1)[keep]
            g_boxes = target["boxes"].detach().cpu().double().reshape(-1, 4)
            g_labels = target["labels"].detach().cpu().long().reshape(-1)
            flags = target.get(self.ignore_key) if self.ignore_key else None
            g_ignore = flags.detach().cpu().bool().reshape(-1) if isinstance(flags, Tensor) else torch.zeros(g_boxes.shape[0], dtype=torch.bool)
            if p_labels.numel() and (p_labels.min() < 0 or p_labels.max() >= self.num_classes):
                raise ValueError(f"prediction labels must be in [0, {self.num_classes})")
            tp, ignored, _ = greedy_match(p_boxes, p_scores, p_labels, g_boxes, g_labels, g_ignore, self.iou_threshold, self.class_aware)
            counted = ~ignored
            self.tp += torch.bincount(p_labels[tp & counted], minlength=self.num_classes)
            self.fp += torch.bincount(p_labels[~tp & counted], minlength=self.num_classes)

    def compute(self) -> Dict[str, Any]:
        """Return {"precision": micro float, "precision_per_class": (C,) float64, "tp": int, "fp": int}."""
        tp_total, fp_total = int(self.tp.sum()), int(self.fp.sum())
        micro = tp_total / (tp_total + fp_total) if tp_total + fp_total > 0 else self.zero_division
        denom = (self.tp + self.fp).double()
        per_class = torch.where(denom > 0, self.tp.double() / denom.clamp(min=1), torch.full_like(denom, self.zero_division))
        return {"precision": float(micro), "precision_per_class": per_class, "tp": tp_total, "fp": fp_total}
