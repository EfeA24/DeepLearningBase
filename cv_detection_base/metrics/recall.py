# Purpose: Accumulates detection true positives and false negatives at explicit confidence and IoU thresholds and reports detection recall (micro and per class).
"""Detection recall at fixed thresholds.

Per image, predictions with score >= ``score_threshold`` are greedily matched by
descending score to the unmatched GT box with highest IoU >= ``iou_threshold``
(same class when ``class_aware=True``). Each non-ignored GT is either matched
(TP) or missed (FN). GT flagged by ``ignore_key`` (e.g. "difficult") are excluded
from both counts.

recall = TP / (TP + FN) = TP / #non-ignored GT. With no GT the result is
``zero_division`` (default 0.0; float("nan") marks "undefined"). Per-class
recall is attributed to the GT class.

Inputs follow metrics/map.py: predictions[i] {"boxes", "scores", "labels"},
targets[i] {"boxes", "labels", optional ignore flags}; boxes xyxy in matching units.

Requires: torch only.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import torch
from torch import Tensor


def _pairwise_iou(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-9) -> Tensor:
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    wh = (torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:]) - torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area1[:, None] + area2[None, :] - inter).clamp(min=eps)


class DetectionRecall:
    """TP/FN accumulator for detection recall.

    Args:
        num_classes: Number of classes (GT labels in [0, num_classes)).
        iou_threshold: Minimum IoU for a match.
        score_threshold: Minimum score for a prediction to be used.
        class_aware: Require equal labels for a match.
        ignore_key: Target key with per-GT ignore flags, or None.
        zero_division: Value returned when there are no non-ignored GT boxes.
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
        self.fn = torch.zeros(self.num_classes, dtype=torch.long)

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
            if g_labels.numel() and (g_labels.min() < 0 or g_labels.max() >= self.num_classes):
                raise ValueError(f"target labels must be in [0, {self.num_classes})")
            flags = target.get(self.ignore_key) if self.ignore_key else None
            g_ignore = flags.detach().cpu().bool().reshape(-1) if isinstance(flags, Tensor) else torch.zeros(g_boxes.shape[0], dtype=torch.bool)

            matched = torch.zeros(g_boxes.shape[0], dtype=torch.bool)
            if p_boxes.shape[0] and g_boxes.shape[0]:
                iou = _pairwise_iou(p_boxes, g_boxes)
                if self.class_aware:
                    iou = iou * (p_labels[:, None] == g_labels[None, :]).to(iou.dtype)
                for p_idx in torch.sort(p_scores, descending=True, stable=True).indices.tolist():
                    candidates = iou[p_idx].clone()
                    candidates[matched | g_ignore] = -1.0
                    best = int(torch.argmax(candidates))
                    if candidates[best] >= self.iou_threshold:
                        matched[best] = True
            counted = ~g_ignore
            self.tp += torch.bincount(g_labels[matched & counted], minlength=self.num_classes)
            self.fn += torch.bincount(g_labels[~matched & counted], minlength=self.num_classes)

    def compute(self) -> Dict[str, Any]:
        """Return {"recall": micro float, "recall_per_class": (C,) float64, "tp": int, "fn": int}."""
        tp_total, fn_total = int(self.tp.sum()), int(self.fn.sum())
        micro = tp_total / (tp_total + fn_total) if tp_total + fn_total > 0 else self.zero_division
        denom = (self.tp + self.fn).double()
        per_class = torch.where(denom > 0, self.tp.double() / denom.clamp(min=1), torch.full_like(denom, self.zero_division))
        return {"recall": float(micro), "recall_per_class": per_class, "tp": tp_total, "fn": fn_total}
