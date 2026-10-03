# Purpose: Provides small box helpers (area, format conversion, validity checks) for object-detection projects.
"""Basic axis-aligned box helpers.

Contract:
    Boxes are float tensors of shape (..., 4) in one of three formats:
      - "xyxy":   (x1, y1, x2, y2) top-left and bottom-right corners
      - "xywh":   (x1, y1, w, h) top-left corner plus width/height (COCO style)
      - "cxcywh": (cx, cy, w, h) center plus width/height (YOLO style)
    Units are whatever the caller uses (absolute pixels or normalized [0, 1]);
    these helpers never change units. The warehouse target schema uses xyxy
    absolute pixels with continuous coordinates (x2 = x1 + w).

Requires: torch only.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

BOX_FORMATS: Tuple[str, ...] = ("xyxy", "xywh", "cxcywh")


def check_boxes_shape(boxes: Tensor, name: str = "boxes") -> None:
    """Raise if ``boxes`` is not a floating tensor with a trailing dimension of 4."""
    if not isinstance(boxes, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(boxes).__name__}")
    if boxes.ndim < 1 or boxes.shape[-1] != 4:
        raise ValueError(f"{name} must have shape (..., 4), got {tuple(boxes.shape)}")
    if not boxes.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor, got {boxes.dtype}")


def _check_format(fmt: str) -> None:
    if fmt not in BOX_FORMATS:
        raise ValueError(f"Unknown box format {fmt!r}; expected one of {BOX_FORMATS}")


def box_convert(boxes: Tensor, in_fmt: str, out_fmt: str) -> Tensor:
    """Convert boxes between "xyxy", "xywh" and "cxcywh" (units unchanged).

    Args:
        boxes: (..., 4) float tensor in ``in_fmt``.
        in_fmt: Source format.
        out_fmt: Target format.

    Returns:
        New (..., 4) tensor in ``out_fmt``.
    """
    check_boxes_shape(boxes)
    _check_format(in_fmt)
    _check_format(out_fmt)
    if in_fmt == out_fmt:
        return boxes.clone()

    a, b, c, d = boxes.unbind(-1)
    if in_fmt == "xyxy":
        x1, y1, x2, y2 = a, b, c, d
    elif in_fmt == "xywh":
        x1, y1, x2, y2 = a, b, a + c, b + d
    else:
        x1, y1, x2, y2 = a - 0.5 * c, b - 0.5 * d, a + 0.5 * c, b + 0.5 * d

    if out_fmt == "xyxy":
        out = (x1, y1, x2, y2)
    elif out_fmt == "xywh":
        out = (x1, y1, x2 - x1, y2 - y1)
    else:
        out = ((x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1)
    return torch.stack(out, dim=-1)


def box_area(boxes: Tensor, fmt: str = "xyxy") -> Tensor:
    """Area of boxes, shape (...,). Negative extents are clamped to zero area."""
    check_boxes_shape(boxes)
    _check_format(fmt)
    if fmt == "xyxy":
        widths = boxes[..., 2] - boxes[..., 0]
        heights = boxes[..., 3] - boxes[..., 1]
    else:
        widths = boxes[..., 2]
        heights = boxes[..., 3]
    return widths.clamp(min=0) * heights.clamp(min=0)


def box_widths_heights(boxes: Tensor) -> Tuple[Tensor, Tensor]:
    """Return (widths, heights) of xyxy boxes, each of shape (...,)."""
    check_boxes_shape(boxes)
    return boxes[..., 2] - boxes[..., 0], boxes[..., 3] - boxes[..., 1]


def valid_boxes_mask(boxes: Tensor, min_size: float = 0.0) -> Tensor:
    """Bool mask (...,) of finite xyxy boxes with width and height strictly greater than ``min_size``."""
    check_boxes_shape(boxes)
    if min_size < 0:
        raise ValueError(f"min_size must be >= 0, got {min_size}")
    widths, heights = box_widths_heights(boxes)
    finite = torch.isfinite(boxes).all(dim=-1)
    return finite & (widths > min_size) & (heights > min_size)


def assert_valid_boxes(boxes: Tensor, name: str = "boxes") -> None:
    """Raise ValueError if any xyxy box is non-finite or has x2 < x1 / y2 < y1.

    Zero-area boxes are allowed here; use ``valid_boxes_mask`` to filter them.
    """
    check_boxes_shape(boxes, name)
    if boxes.numel() == 0:
        return
    if not torch.isfinite(boxes).all():
        raise ValueError(f"{name} contains non-finite values")
    widths, heights = box_widths_heights(boxes)
    if (widths < 0).any() or (heights < 0).any():
        raise ValueError(f"{name} must satisfy x2 >= x1 and y2 >= y1 (xyxy format)")


def boxes_within_image(boxes: Tensor, image_size: Tuple[int, int], tolerance: float = 0.0) -> Tensor:
    """Bool mask (...,) of xyxy pixel boxes lying inside an image of size (height, width)."""
    check_boxes_shape(boxes)
    height, width = image_size
    if height <= 0 or width <= 0:
        raise ValueError(f"image_size must be positive (height, width), got {image_size}")
    return (
        (boxes[..., 0] >= -tolerance)
        & (boxes[..., 1] >= -tolerance)
        & (boxes[..., 2] <= width + tolerance)
        & (boxes[..., 3] <= height + tolerance)
    )
