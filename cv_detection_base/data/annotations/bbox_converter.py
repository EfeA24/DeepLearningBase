# Purpose: Converts detection boxes between xyxy/xywh/cxcywh formats and between normalized and absolute pixel units using explicit image dimensions.
"""Box format and unit conversion for annotation pipelines.

Contract:
    Boxes are float tensors of shape (N, 4) (any leading dims allowed).
    Formats: "xyxy" (x1, y1, x2, y2), "xywh" (x1, y1, w, h; COCO),
    "cxcywh" (cx, cy, w, h; YOLO).
    Units: "pixel" (absolute, continuous coordinates where x2 = x1 + w) or
    "normalized" (divided by image width/height, nominally in [0, 1]).
    Image dimensions are always passed explicitly as ``image_width`` and
    ``image_height`` in pixels; they are never inferred from the boxes.
    Normalization scales x-like values by width and y-like values by height,
    which is valid for every supported format (all coordinates are x/y/w/h).

Requires: torch only (self-contained; overlaps with boxes/bbox.py on purpose).
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

BOX_FORMATS: Tuple[str, ...] = ("xyxy", "xywh", "cxcywh")
BOX_UNITS: Tuple[str, ...] = ("pixel", "normalized")


def _check(boxes: Tensor) -> None:
    if not isinstance(boxes, Tensor):
        raise TypeError(f"boxes must be a torch.Tensor, got {type(boxes).__name__}")
    if boxes.ndim < 1 or boxes.shape[-1] != 4:
        raise ValueError(f"boxes must have shape (..., 4), got {tuple(boxes.shape)}")
    if not boxes.is_floating_point():
        raise TypeError(f"boxes must be floating point, got {boxes.dtype}")


def _check_dims(image_width: float, image_height: float) -> None:
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"image dimensions must be positive, got width={image_width}, height={image_height}")


def xyxy_to_xywh(boxes: Tensor) -> Tensor:
    _check(boxes)
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack((x1, y1, x2 - x1, y2 - y1), dim=-1)


def xywh_to_xyxy(boxes: Tensor) -> Tensor:
    _check(boxes)
    x, y, w, h = boxes.unbind(-1)
    return torch.stack((x, y, x + w, y + h), dim=-1)


def xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    _check(boxes)
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack(((x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1), dim=-1)


def cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    _check(boxes)
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack((cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h), dim=-1)


def convert_format(boxes: Tensor, in_fmt: str, out_fmt: str) -> Tensor:
    """Convert between "xyxy", "xywh" and "cxcywh" (units unchanged)."""
    for fmt in (in_fmt, out_fmt):
        if fmt not in BOX_FORMATS:
            raise ValueError(f"Unknown box format {fmt!r}; expected one of {BOX_FORMATS}")
    _check(boxes)
    if in_fmt == out_fmt:
        return boxes.clone()
    if in_fmt == "xywh":
        xyxy = xywh_to_xyxy(boxes)
    elif in_fmt == "cxcywh":
        xyxy = cxcywh_to_xyxy(boxes)
    else:
        xyxy = boxes
    if out_fmt == "xyxy":
        return xyxy.clone() if xyxy is boxes else xyxy
    if out_fmt == "xywh":
        return xyxy_to_xywh(xyxy)
    return xyxy_to_cxcywh(xyxy)


def normalize_boxes(boxes: Tensor, image_width: float, image_height: float) -> Tensor:
    """Pixel boxes -> normalized boxes (any supported format)."""
    _check(boxes)
    _check_dims(image_width, image_height)
    scale = boxes.new_tensor([image_width, image_height, image_width, image_height])
    return boxes / scale


def denormalize_boxes(boxes: Tensor, image_width: float, image_height: float) -> Tensor:
    """Normalized boxes -> pixel boxes (any supported format)."""
    _check(boxes)
    _check_dims(image_width, image_height)
    scale = boxes.new_tensor([image_width, image_height, image_width, image_height])
    return boxes * scale


def convert_boxes(
    boxes: Tensor,
    in_fmt: str,
    out_fmt: str,
    in_units: str = "pixel",
    out_units: str = "pixel",
    image_width: float = 0.0,
    image_height: float = 0.0,
) -> Tensor:
    """Convert format and units in one call.

    ``image_width``/``image_height`` are required (positive) whenever
    ``in_units != out_units``.

    Example:
        YOLO normalized cxcywh -> target-schema xyxy pixels::

            convert_boxes(b, "cxcywh", "xyxy", "normalized", "pixel", image_width=640, image_height=480)
    """
    for units in (in_units, out_units):
        if units not in BOX_UNITS:
            raise ValueError(f"Unknown units {units!r}; expected one of {BOX_UNITS}")
    out = convert_format(boxes, in_fmt, out_fmt)
    if in_units == out_units:
        return out
    if in_units == "normalized":
        return denormalize_boxes(out, image_width, image_height)
    return normalize_boxes(out, image_width, image_height)
