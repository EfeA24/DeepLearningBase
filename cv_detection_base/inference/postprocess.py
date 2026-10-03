# Purpose: Decodes raw dense detection-head outputs (documented A*(5+C) layout) into filtered xyxy boxes, class labels and scores under explicit thresholds with injectable NMS.
"""Detection post-processing.

Head layout (models/heads/detection_head.py): per level (B, A*(5+C), H, W), per
anchor channels [tx, ty, tw, th, objectness_logit, class_logits...]. Levels
are flattened to (B, H*W*A, 5+C) in (y, x, anchor) order and concatenated.

Box decoding conventions (pick the one your loss was trained with, or inject
``box_decoder``):
    anchor_free (A must be 1), grid cell (gx, gy), stride s, offset o (default 0.5):
        cx = (gx + o + tx) * s,  cy = (gy + o + ty) * s
        w  = exp(tw) * s,        h  = exp(th) * s        (tw/th clamped before exp)
    anchor_based: R-CNN deltas relative to supplied anchors (xyxy pixels,
        models/anchors/anchor_generator.py ordering) with ``weights`` as in
        models/anchors/anchor_utils.py.
Decoded boxes are xyxy in network-input pixels. Map them back to original
images yourself (undo resize/letterbox) - this module does not know your preprocessing.

Scores: sigmoid(objectness) * class_prob, where class_prob = sigmoid(class
logits) ("sigmoid", independent classes) or softmax ("softmax"). By default
each location keeps its best class; ``multi_label=True`` keeps every
(location, class) pair above ``score_threshold``.

NMS is injectable (``nms_fn(boxes, scores, labels, iou_threshold) -> keep``);
by default boxes/nms.py ``batched_nms`` is imported lazily.

Requires: torch; sibling boxes/nms.py only when ``nms_fn`` is not supplied.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

NmsFn = Callable[[Tensor, Tensor, Tensor, float], Tensor]
SCALE_CLAMP = math.log(1000.0 / 16)


def flatten_head_outputs(level_outputs: Sequence[Tensor], num_anchors: int, num_classes: int) -> Tuple[Tensor, List[Tuple[int, int]]]:
    """Concatenate levels into (B, P, 5+C) and return the per-level (H, W) sizes."""
    num_outputs = 5 + num_classes
    flat, sizes = [], []
    for i, out in enumerate(level_outputs):
        if out.ndim != 4 or out.shape[1] != num_anchors * num_outputs:
            raise ValueError(f"level {i}: expected (B, {num_anchors * num_outputs}, H, W), got {tuple(out.shape)}")
        b, _, h, w = out.shape
        flat.append(out.view(b, num_anchors, num_outputs, h, w).permute(0, 3, 4, 1, 2).reshape(b, h * w * num_anchors, num_outputs))
        sizes.append((h, w))
    return torch.cat(flat, dim=1), sizes


def decode_anchor_free(
    level_outputs: Sequence[Tensor],
    strides: Sequence[float],
    num_classes: int,
    offset: float = 0.5,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Decode anchor-free outputs -> (boxes (B, P, 4) xyxy pixels, obj logits (B, P), class logits (B, P, C))."""
    if len(strides) != len(level_outputs):
        raise ValueError("one stride per level is required")
    flat, sizes = flatten_head_outputs(level_outputs, 1, num_classes)
    flat = flat.float()
    grids, scale = [], []
    for (h, w), stride in zip(sizes, strides):
        ys, xs = torch.meshgrid(
            torch.arange(h, device=flat.device, dtype=flat.dtype),
            torch.arange(w, device=flat.device, dtype=flat.dtype),
            indexing="ij",
        )
        grids.append(torch.stack((xs.reshape(-1), ys.reshape(-1)), dim=-1))
        scale.append(torch.full((h * w, 1), float(stride), device=flat.device, dtype=flat.dtype))
    grid = torch.cat(grids)[None]
    stride_t = torch.cat(scale)[None]
    centers = (grid + offset + flat[..., 0:2]) * stride_t
    sizes_wh = torch.exp(flat[..., 2:4].clamp(max=SCALE_CLAMP)) * stride_t
    boxes = torch.cat((centers - 0.5 * sizes_wh, centers + 0.5 * sizes_wh), dim=-1)
    return boxes, flat[..., 4], flat[..., 5:]


def decode_anchor_based(
    level_outputs: Sequence[Tensor],
    anchors: Sequence[Tensor],
    num_anchors: int,
    num_classes: int,
    weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
) -> Tuple[Tensor, Tensor, Tensor]:
    """Decode anchor-based outputs with per-level anchors (H*W*A, 4) xyxy pixels."""
    if len(anchors) != len(level_outputs):
        raise ValueError("one anchor tensor per level is required")
    flat, sizes = flatten_head_outputs(level_outputs, num_anchors, num_classes)
    flat = flat.float()
    all_anchors = torch.cat([a.to(device=flat.device, dtype=flat.dtype) for a in anchors], dim=0)
    if all_anchors.shape[0] != flat.shape[1]:
        raise ValueError(f"anchors cover {all_anchors.shape[0]} positions but outputs have {flat.shape[1]}")
    wx, wy, ww, wh = weights
    aw = all_anchors[:, 2] - all_anchors[:, 0]
    ah = all_anchors[:, 3] - all_anchors[:, 1]
    ax = all_anchors[:, 0] + 0.5 * aw
    ay = all_anchors[:, 1] + 0.5 * ah
    cx = flat[..., 0] / wx * aw + ax
    cy = flat[..., 1] / wy * ah + ay
    w = torch.exp((flat[..., 2] / ww).clamp(max=SCALE_CLAMP)) * aw
    h = torch.exp((flat[..., 3] / wh).clamp(max=SCALE_CLAMP)) * ah
    boxes = torch.stack((cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h), dim=-1)
    return boxes, flat[..., 4], flat[..., 5:]


def _default_nms_fn() -> NmsFn:
    try:
        from boxes.nms import batched_nms
    except ImportError as exc:
        raise ImportError(
            "No nms_fn supplied and boxes/nms.py is not importable; add the template root to sys.path "
            "or pass nms_fn (e.g. torchvision.ops.batched_nms)"
        ) from exc
    return lambda b, s, l, t: batched_nms(b, s, l, t)


def select_detections(
    boxes: Tensor,
    objectness_logits: Tensor,
    class_logits: Tensor,
    score_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    max_detections: int = 100,
    image_sizes: Optional[Sequence[Tuple[int, int]]] = None,
    class_activation: str = "sigmoid",
    multi_label: bool = False,
    class_agnostic_nms: bool = False,
    pre_nms_top_k: Optional[int] = 1000,
    nms_fn: Optional[NmsFn] = None,
) -> List[Dict[str, Tensor]]:
    """Score, threshold, clip, NMS and cap decoded predictions.

    Args:
        boxes: (B, P, 4) xyxy pixels.
        objectness_logits: (B, P).
        class_logits: (B, P, C).
        score_threshold: Keep candidates with score > threshold.
        iou_threshold: NMS IoU threshold.
        max_detections: Max detections per image after NMS.
        image_sizes: Optional (H, W) per image; boxes are clipped to it.
        class_activation: "sigmoid" or "softmax".
        multi_label: Keep every class above threshold per location.
        class_agnostic_nms: Suppress across classes.
        pre_nms_top_k: Keep at most this many candidates per image before NMS.
        nms_fn: Injectable NMS kernel (see module docstring).

    Returns:
        List of B dicts {"boxes": (K, 4), "scores": (K,), "labels": (K,) int64}.
    """
    if class_activation not in ("sigmoid", "softmax"):
        raise ValueError("class_activation must be 'sigmoid' or 'softmax'")
    if boxes.ndim != 3 or objectness_logits.shape != boxes.shape[:2] or class_logits.shape[:2] != boxes.shape[:2]:
        raise ValueError("expected boxes (B, P, 4), objectness (B, P), class logits (B, P, C)")
    if image_sizes is not None and len(image_sizes) != boxes.shape[0]:
        raise ValueError("image_sizes must have one (H, W) per image")
    kernel = nms_fn or _default_nms_fn()
    obj = torch.sigmoid(objectness_logits.float())
    cls = torch.sigmoid(class_logits.float()) if class_activation == "sigmoid" else torch.softmax(class_logits.float(), dim=-1)
    scores_all = obj[..., None] * cls

    results: List[Dict[str, Tensor]] = []
    for i in range(boxes.shape[0]):
        img_boxes = boxes[i].float()
        if image_sizes is not None:
            h, w = image_sizes[i]
            img_boxes = torch.stack(
                (img_boxes[:, 0].clamp(0, w), img_boxes[:, 1].clamp(0, h), img_boxes[:, 2].clamp(0, w), img_boxes[:, 3].clamp(0, h)),
                dim=-1,
            )
        if multi_label:
            loc_idx, cls_idx = torch.nonzero(scores_all[i] > score_threshold, as_tuple=True)
            cand_boxes, cand_scores, cand_labels = img_boxes[loc_idx], scores_all[i][loc_idx, cls_idx], cls_idx
        else:
            best_scores, best_labels = scores_all[i].max(dim=-1)
            keep = best_scores > score_threshold
            cand_boxes, cand_scores, cand_labels = img_boxes[keep], best_scores[keep], best_labels[keep]
        valid = (cand_boxes[:, 2] > cand_boxes[:, 0]) & (cand_boxes[:, 3] > cand_boxes[:, 1])
        cand_boxes, cand_scores, cand_labels = cand_boxes[valid], cand_scores[valid], cand_labels[valid]
        if pre_nms_top_k is not None and cand_scores.numel() > pre_nms_top_k:
            top = torch.topk(cand_scores, pre_nms_top_k).indices
            cand_boxes, cand_scores, cand_labels = cand_boxes[top], cand_scores[top], cand_labels[top]
        if cand_scores.numel() == 0:
            results.append({"boxes": cand_boxes.reshape(0, 4), "scores": cand_scores, "labels": cand_labels.long()})
            continue
        nms_labels = torch.zeros_like(cand_labels) if class_agnostic_nms else cand_labels
        keep_idx = kernel(cand_boxes, cand_scores, nms_labels.long(), iou_threshold)[:max_detections]
        results.append({"boxes": cand_boxes[keep_idx], "scores": cand_scores[keep_idx], "labels": cand_labels[keep_idx].long()})
    return results


class DetectionPostprocessor:
    """Callable ``(outputs, images=None, targets=None) -> list of detections`` for validation/inference.

    Args:
        num_classes: C.
        strides: Per-level strides (anchor-free decoding) - required if ``anchor_generator`` is None.
        anchor_generator: Optional object with ``generate(feature_sizes, dtype, device)`` and
            ``num_anchors_per_location`` (models/anchors/anchor_generator.py) for anchor-based decoding.
        box_weights: Delta weights for anchor-based decoding.
        box_decoder: Optional custom ``level_outputs -> (boxes, obj_logits, cls_logits)``; overrides both.
        **select_kwargs: Forwarded to ``select_detections`` (thresholds, nms_fn, ...).

    Image sizes for clipping come from targets[i]["image_size"] (set by the
    detection collator) when available, else from the padded batch shape.
    """

    def __init__(
        self,
        num_classes: int,
        strides: Optional[Sequence[float]] = None,
        anchor_generator: Optional[Any] = None,
        box_weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
        box_decoder: Optional[Callable[[Sequence[Tensor]], Tuple[Tensor, Tensor, Tensor]]] = None,
        **select_kwargs: Any,
    ) -> None:
        if box_decoder is None and anchor_generator is None and strides is None:
            raise ValueError("provide strides (anchor-free), an anchor_generator, or a box_decoder")
        self.num_classes = num_classes
        self.strides = list(strides) if strides is not None else None
        self.anchor_generator = anchor_generator
        self.box_weights = tuple(box_weights)
        self.box_decoder = box_decoder
        self.select_kwargs = select_kwargs

    def decode(self, level_outputs: Sequence[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        if self.box_decoder is not None:
            return self.box_decoder(level_outputs)
        if self.anchor_generator is not None:
            sizes = [tuple(o.shape[-2:]) for o in level_outputs]
            anchors = self.anchor_generator.generate(sizes, dtype=torch.float32, device=level_outputs[0].device)
            num_anchors = self.anchor_generator.num_anchors_per_location
            if len(set(num_anchors)) != 1:
                raise ValueError("anchor-based decoding requires the same number of anchors on every level")
            return decode_anchor_based(level_outputs, anchors, num_anchors[0], self.num_classes, self.box_weights)
        return decode_anchor_free(level_outputs, self.strides, self.num_classes)

    def __call__(self, outputs: Sequence[Tensor], images: Any = None, targets: Optional[Sequence[Dict[str, Any]]] = None) -> List[Dict[str, Tensor]]:
        boxes, obj, cls = self.decode(outputs)
        image_sizes = None
        if targets is not None and all("image_size" in t for t in targets):
            image_sizes = [tuple(int(v) for v in t["image_size"]) for t in targets]
        elif isinstance(images, Tensor) and images.ndim == 4:
            image_sizes = [tuple(images.shape[-2:])] * images.shape[0]
        return select_detections(boxes, obj, cls, image_sizes=image_sizes, **self.select_kwargs)
