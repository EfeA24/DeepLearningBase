# Purpose: Provides torch box-geometry helpers for detection augmentation: clip, resize scaling, horizontal/vertical flip, degenerate-box removal and per-box target filtering.
"""Box geometry helpers used by detection transforms.

Contract:
    boxes: FloatTensor (N, 4) xyxy absolute pixels, continuous coordinates
    (a box covering the full image is (0, 0, W, H)). image_size is always
    ``(height, width)``. All functions return new tensors (inputs are not mutated).

Requires: torch only.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import torch
from torch import Tensor

PER_BOX_KEYS: Tuple[str, ...] = ("boxes", "labels", "iscrowd", "difficult", "area", "weights", "scores")


def _check(boxes: Tensor) -> None:
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError(f"boxes must have shape (N, 4), got {tuple(boxes.shape)}")


def clip_boxes(boxes: Tensor, image_size: Tuple[int, int]) -> Tensor:
    """Clamp xyxy boxes to [0, W] x [0, H]."""
    _check(boxes)
    height, width = image_size
    out = boxes.clone()
    out[:, 0::2] = out[:, 0::2].clamp(0, width)
    out[:, 1::2] = out[:, 1::2].clamp(0, height)
    return out


def scale_boxes(boxes: Tensor, scale_x: float, scale_y: float) -> Tensor:
    """Multiply x coordinates by ``scale_x`` and y coordinates by ``scale_y``."""
    _check(boxes)
    if scale_x <= 0 or scale_y <= 0:
        raise ValueError(f"scales must be positive, got ({scale_x}, {scale_y})")
    return boxes * boxes.new_tensor([scale_x, scale_y, scale_x, scale_y])


def resize_boxes(boxes: Tensor, original_size: Tuple[int, int], new_size: Tuple[int, int]) -> Tensor:
    """Rescale boxes for an image resized from ``original_size`` to ``new_size`` (both (H, W))."""
    (old_h, old_w), (new_h, new_w) = original_size, new_size
    if min(old_h, old_w, new_h, new_w) <= 0:
        raise ValueError(f"sizes must be positive, got {original_size} -> {new_size}")
    return scale_boxes(boxes, new_w / old_w, new_h / old_h)


def translate_boxes(boxes: Tensor, dx: float, dy: float) -> Tensor:
    """Shift boxes by (dx, dy) pixels (e.g. after cropping or padding)."""
    _check(boxes)
    return boxes + boxes.new_tensor([dx, dy, dx, dy])


def hflip_boxes(boxes: Tensor, image_width: int) -> Tensor:
    """Mirror boxes horizontally: x1' = W - x2, x2' = W - x1."""
    _check(boxes)
    out = boxes.clone()
    out[:, 0] = image_width - boxes[:, 2]
    out[:, 2] = image_width - boxes[:, 0]
    return out


def vflip_boxes(boxes: Tensor, image_height: int) -> Tensor:
    """Mirror boxes vertically: y1' = H - y2, y2' = H - y1."""
    _check(boxes)
    out = boxes.clone()
    out[:, 1] = image_height - boxes[:, 3]
    out[:, 3] = image_height - boxes[:, 1]
    return out


def degenerate_box_mask(boxes: Tensor, min_size: float = 1.0) -> Tensor:
    """Bool mask (N,) of boxes to KEEP: width and height >= ``min_size`` pixels and finite."""
    _check(boxes)
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    return torch.isfinite(boxes).all(dim=1) & (widths >= min_size) & (heights >= min_size)


def filter_target(target: Dict[str, Any], keep: Tensor, per_box_keys: Iterable[str] = PER_BOX_KEYS) -> Dict[str, Any]:
    """Return a shallow copy of ``target`` with per-box fields indexed by ``keep``.

    Only keys listed in ``per_box_keys`` whose first dimension equals the number
    of boxes are filtered; image-level fields ("image_id", "orig_size") are copied.
    """
    num_boxes = target["boxes"].shape[0]
    if keep.dtype == torch.bool and keep.shape != (num_boxes,):
        raise ValueError(f"keep mask must have shape ({num_boxes},), got {tuple(keep.shape)}")
    out = dict(target)
    for key in per_box_keys:
        value = target.get(key)
        if isinstance(value, Tensor) and value.ndim >= 1 and value.shape[0] == num_boxes:
            out[key] = value[keep]
    return out


def remove_degenerate_boxes(target: Dict[str, Any], min_size: float = 1.0) -> Dict[str, Any]:
    """Drop boxes smaller than ``min_size`` (and their per-box fields) from a target."""
    return filter_target(target, degenerate_box_mask(target["boxes"], min_size))


def box_visibility(original: Tensor, clipped: Tensor, eps: float = 1e-7) -> Tensor:
    """Fraction (N,) of each original box area that remains after clipping/cropping."""
    _check(original)
    _check(clipped)
    area_orig = (original[:, 2] - original[:, 0]).clamp(min=0) * (original[:, 3] - original[:, 1]).clamp(min=0)
    area_clip = (clipped[:, 2] - clipped[:, 0]).clamp(min=0) * (clipped[:, 3] - clipped[:, 1]).clamp(min=0)
    return area_clip / area_orig.clamp(min=eps)
