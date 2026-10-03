# Purpose: Accumulates detection results and computes per-class AP and mAP at multiple IoU thresholds (e.g. 0.5 and 0.5:0.95) with selectable all-point or COCO 101-point interpolation.
"""Mean Average Precision (mAP) accumulator.

Usage::

    metric = MeanAveragePrecision(num_classes=20)            # IoU 0.50:0.05:0.95
    metric.update(predictions, targets)                       # per batch
    result = metric.compute()                                 # {"map", "map_50", "map_75", ...}
    metric.reset()

Inputs (per image, lists of equal length):
    predictions[i]: {"boxes": (P, 4) xyxy, "scores": (P,), "labels": (P,) int64}
    targets[i]:     {"boxes": (G, 4) xyxy, "labels": (G,) int64,
                     optional "difficult" or "iscrowd": (G,) bool/int ignore flags}
    Boxes in the same units for predictions and targets (pixels recommended).

Matching (per image, per class, per IoU threshold): predictions sorted by score
(descending, stable); each prediction greedily takes the unmatched non-ignored
GT with the highest IoU >= threshold -> TP. Otherwise, if it overlaps an
ignored GT (difficult/crowd) with IoU >= threshold it is IGNORED (neither TP nor
FP; ignored GTs may absorb several predictions). Otherwise FP. Ignored GTs never
count as positives. This follows the VOC "difficult" protocol; COCO's crowd
handling (intersection-over-detection-area) is simplified to IoU here.

AP interpolation:
    "all_point" - area under the monotone (right-max) precision envelope at
                  every recall change (VOC2010+ style).
    "coco_101"  - mean of the envelope sampled at 101 recall points 0:0.01:1 (COCO style).

Edge cases: classes without any GT across the dataset are reported as NaN in
per-class AP and excluded from the mean; a class with GT but no predictions has
AP 0. If no class has GT, "map" is NaN. ``max_detections`` (optional) keeps the
top-scoring predictions per image (COCO uses 100).
Results are approximately, not bit-exactly, equal to pycocotools/VOC devkit.

Requires: torch only.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
from torch import Tensor


def _pairwise_iou(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-9) -> Tensor:
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    wh = (torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:]) - torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area1[:, None] + area2[None, :] - inter).clamp(min=eps)


def average_precision(recall: Tensor, precision: Tensor, method: str = "all_point") -> float:
    """AP from a recall/precision curve ordered by decreasing score.

    Args:
        recall: (K,) non-decreasing recall values.
        precision: (K,) precision values at the same ranks.
        method: "all_point" or "coco_101".
    """
    if recall.numel() == 0:
        return 0.0
    recall = recall.double()
    precision = precision.double()
    if method == "all_point":
        mrec = torch.cat([recall.new_zeros(1), recall, recall.new_ones(1)])
        mpre = torch.cat([precision.new_zeros(1), precision, precision.new_zeros(1)])
        mpre = torch.flip(torch.cummax(torch.flip(mpre, [0]), dim=0).values, [0])
        change = torch.nonzero(mrec[1:] != mrec[:-1]).flatten()
        return float(((mrec[change + 1] - mrec[change]) * mpre[change + 1]).sum())
    if method == "coco_101":
        envelope = torch.flip(torch.cummax(torch.flip(precision, [0]), dim=0).values, [0])
        thresholds = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)
        idx = torch.searchsorted(recall.contiguous(), thresholds, right=False)
        valid = idx < recall.numel()
        sampled = torch.zeros_like(thresholds)
        sampled[valid] = envelope[idx[valid]]
        return float(sampled.mean())
    raise ValueError(f"unknown AP method {method!r}; expected 'all_point' or 'coco_101'")


class MeanAveragePrecision:
    """Batch-wise mAP accumulator. See module docstring for semantics.

    Args:
        num_classes: Number of classes (labels in [0, num_classes)).
        iou_thresholds: IoU thresholds; default COCO 0.50:0.05:0.95.
        interpolation: "all_point" or "coco_101".
        max_detections: Optional per-image cap on predictions (top scores kept).
        ignore_key: Target key holding per-GT ignore flags ("difficult", "iscrowd"),
            or None to ignore nothing. Missing keys mean "no ignored GT".
    """

    def __init__(
        self,
        num_classes: int,
        iou_thresholds: Optional[Sequence[float]] = None,
        interpolation: str = "coco_101",
        max_detections: Optional[int] = None,
        ignore_key: Optional[str] = "difficult",
    ) -> None:
        if num_classes <= 0:
            raise ValueError("num_classes must be > 0")
        if interpolation not in ("all_point", "coco_101"):
            raise ValueError("interpolation must be 'all_point' or 'coco_101'")
        if iou_thresholds is None:
            iou_thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]
        if not iou_thresholds or any(not 0.0 < t <= 1.0 for t in iou_thresholds):
            raise ValueError("iou_thresholds must be non-empty values in (0, 1]")
        if max_detections is not None and max_detections <= 0:
            raise ValueError("max_detections must be > 0")
        self.num_classes = num_classes
        self.iou_thresholds = [float(t) for t in iou_thresholds]
        self.interpolation = interpolation
        self.max_detections = max_detections
        self.ignore_key = ignore_key
        self.reset()

    def reset(self) -> None:
        self._scores: List[List[Tensor]] = [[] for _ in range(self.num_classes)]
        self._tp: List[List[Tensor]] = [[] for _ in range(self.num_classes)]
        self._ignored: List[List[Tensor]] = [[] for _ in range(self.num_classes)]
        self._num_gt = torch.zeros(self.num_classes, dtype=torch.long)
        self._num_images = 0

    @torch.no_grad()
    def update(self, predictions: Sequence[Mapping[str, Tensor]], targets: Sequence[Mapping[str, Any]]) -> None:
        if len(predictions) != len(targets):
            raise ValueError(f"got {len(predictions)} predictions for {len(targets)} targets")
        num_thr = len(self.iou_thresholds)
        thresholds = torch.tensor(self.iou_thresholds, dtype=torch.float64)
        for pred, target in zip(predictions, targets):
            p_boxes = pred["boxes"].detach().cpu().double().reshape(-1, 4)
            p_scores = pred["scores"].detach().cpu().double().reshape(-1)
            p_labels = pred["labels"].detach().cpu().long().reshape(-1)
            g_boxes = target["boxes"].detach().cpu().double().reshape(-1, 4)
            g_labels = target["labels"].detach().cpu().long().reshape(-1)
            if not (p_boxes.shape[0] == p_scores.shape[0] == p_labels.shape[0]):
                raise ValueError("prediction boxes/scores/labels lengths differ")
            if g_boxes.shape[0] != g_labels.shape[0]:
                raise ValueError("target boxes/labels lengths differ")
            for labels in (p_labels, g_labels):
                if labels.numel() and (labels.min() < 0 or labels.max() >= self.num_classes):
                    raise ValueError(f"labels must be in [0, {self.num_classes})")
            ignore_flags = target.get(self.ignore_key) if self.ignore_key else None
            g_ignore = (
                ignore_flags.detach().cpu().bool().reshape(-1)
                if isinstance(ignore_flags, Tensor)
                else torch.zeros(g_boxes.shape[0], dtype=torch.bool)
            )

            if self.max_detections is not None and p_scores.numel() > self.max_detections:
                top = torch.sort(p_scores, descending=True, stable=True).indices[: self.max_detections]
                p_boxes, p_scores, p_labels = p_boxes[top], p_scores[top], p_labels[top]

            classes = torch.unique(torch.cat([p_labels, g_labels]))
            for c in classes.tolist():
                pm = p_labels == c
                gm = g_labels == c
                c_boxes, c_scores = p_boxes[pm], p_scores[pm]
                c_gt, c_ign = g_boxes[gm], g_ignore[gm]
                self._num_gt[c] += int((~c_ign).sum())
                if c_scores.numel() == 0:
                    continue
                order = torch.sort(c_scores, descending=True, stable=True).indices
                c_boxes, c_scores = c_boxes[order], c_scores[order]
                tp = torch.zeros(num_thr, c_scores.numel(), dtype=torch.bool)
                ignored = torch.zeros(num_thr, c_scores.numel(), dtype=torch.bool)
                if c_gt.shape[0] > 0:
                    iou = _pairwise_iou(c_boxes, c_gt)
                    for t_idx, thr in enumerate(thresholds.tolist()):
                        matched = torch.zeros(c_gt.shape[0], dtype=torch.bool)
                        for p_idx in range(c_scores.numel()):
                            candidates = iou[p_idx].clone()
                            candidates[matched | c_ign] = -1.0
                            best = int(torch.argmax(candidates)) if candidates.numel() else -1
                            if best >= 0 and candidates[best] >= thr:
                                matched[best] = True
                                tp[t_idx, p_idx] = True
                            elif c_ign.any() and bool((iou[p_idx][c_ign] >= thr).any()):
                                ignored[t_idx, p_idx] = True
                self._scores[c].append(c_scores)
                self._tp[c].append(tp)
                self._ignored[c].append(ignored)
            self._num_images += 1

    def compute(self) -> Dict[str, Any]:
        """Return {"map", "map_50" (if 0.5 used), "map_75" (if 0.75 used), "ap_per_class" (C,),
        "ap_per_threshold" (T,), "ap_matrix" (C, T), "num_gt_per_class" (C,)} (NaN where undefined)."""
        num_thr = len(self.iou_thresholds)
        ap = torch.full((self.num_classes, num_thr), float("nan"), dtype=torch.float64)
        for c in range(self.num_classes):
            n_gt = int(self._num_gt[c])
            if n_gt == 0:
                continue
            if not self._scores[c]:
                ap[c] = 0.0
                continue
            scores = torch.cat(self._scores[c])
            tp_all = torch.cat(self._tp[c], dim=1)
            ign_all = torch.cat(self._ignored[c], dim=1)
            order = torch.sort(scores, descending=True, stable=True).indices
            tp_all, ign_all = tp_all[:, order], ign_all[:, order]
            for t in range(num_thr):
                keep = ~ign_all[t]
                tp = tp_all[t][keep].double()
                fp = 1.0 - tp
                ctp, cfp = torch.cumsum(tp, 0), torch.cumsum(fp, 0)
                recall = ctp / n_gt
                precision = ctp / (ctp + cfp).clamp(min=1e-12)
                ap[c, t] = average_precision(recall, precision, self.interpolation)

        valid = ~torch.isnan(ap[:, 0])
        per_class = ap.mean(dim=1)
        per_threshold = ap[valid].mean(dim=0) if valid.any() else torch.full((num_thr,), float("nan"), dtype=torch.float64)
        result: Dict[str, Any] = {
            "map": float(per_class[valid].mean()) if valid.any() else float("nan"),
            "ap_per_class": per_class,
            "ap_per_threshold": per_threshold,
            "ap_matrix": ap,
            "num_gt_per_class": self._num_gt.clone(),
            "num_images": self._num_images,
        }
        for name, thr in (("map_50", 0.5), ("map_75", 0.75)):
            for t_idx, value in enumerate(self.iou_thresholds):
                if abs(value - thr) < 1e-9:
                    result[name] = float(per_threshold[t_idx])
        return result
