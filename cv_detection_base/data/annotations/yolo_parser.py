# Purpose: Parses YOLO-format text annotations ("class xc yc w h", normalized) and converts them to the xyxy-pixel detection target schema.
"""YOLO text annotation parsing.

Input format (one object per non-empty line, whitespace separated):
    ``class_id x_center y_center width height``
    - class_id: non-negative integer (0-based)
    - coordinates: normalized to [0, 1] by image width/height (cxcywh)
    Lines starting with '#' and blank lines are ignored. Segmentation polygons or
    keypoints (more than 5 values per row) are NOT supported and raise ValueError.

Outputs:
    ``parse_yolo_text`` -> (labels LongTensor (N,), boxes FloatTensor (N, 4) cxcywh normalized)
    ``yolo_to_target``  -> {"boxes": (N, 4) xyxy absolute pixels, "labels": (N,)}
    Empty files give labels (0,) and boxes (0, 4).

Requires: torch only (pathlib imported lazily for file reading).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


def parse_yolo_line(line: str, line_number: int = 0, tolerance: float = 1e-6) -> Tuple[int, float, float, float, float]:
    """Parse one "class xc yc w h" row and validate normalized ranges.

    Args:
        line: Row text.
        line_number: Used in error messages only.
        tolerance: Allowed slack outside [0, 1] for float rounding in exported labels.
    """
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"line {line_number}: expected 5 values 'class xc yc w h', got {len(parts)}: {line!r}")
    try:
        class_value = float(parts[0])
        xc, yc, w, h = (float(p) for p in parts[1:])
    except ValueError as exc:
        raise ValueError(f"line {line_number}: non-numeric value in {line!r}") from exc
    if class_value < 0 or class_value != int(class_value):
        raise ValueError(f"line {line_number}: class id must be a non-negative integer, got {parts[0]!r}")
    for name, value in (("x_center", xc), ("y_center", yc), ("width", w), ("height", h)):
        if not (-tolerance <= value <= 1.0 + tolerance):
            raise ValueError(f"line {line_number}: {name}={value} is outside normalized range [0, 1]")
    if w <= 0 or h <= 0:
        raise ValueError(f"line {line_number}: width and height must be > 0, got w={w}, h={h}")
    return int(class_value), xc, yc, w, h


def parse_yolo_text(
    text: str,
    num_classes: Optional[int] = None,
    tolerance: float = 1e-6,
) -> Tuple[Tensor, Tensor]:
    """Parse the full content of a YOLO label file.

    Args:
        text: File content.
        num_classes: If given, class ids must be < num_classes.
        tolerance: See ``parse_yolo_line``.

    Returns:
        labels (N,) int64 and boxes (N, 4) float32 cxcywh normalized.
    """
    labels: List[int] = []
    boxes: List[Tuple[float, float, float, float]] = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        class_id, xc, yc, w, h = parse_yolo_line(line, line_number, tolerance)
        if num_classes is not None and class_id >= num_classes:
            raise ValueError(f"line {line_number}: class id {class_id} >= num_classes={num_classes}")
        labels.append(class_id)
        boxes.append((xc, yc, w, h))
    labels_t = torch.tensor(labels, dtype=torch.long)
    boxes_t = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    return labels_t, boxes_t


def read_yolo_label_file(path: str, num_classes: Optional[int] = None, missing_ok: bool = False) -> Tuple[Tensor, Tensor]:
    """Read and parse a YOLO label file from an explicit path.

    Args:
        path: Label file path.
        num_classes: Optional class-count validation.
        missing_ok: If True a missing file yields empty tensors (image without objects).
    """
    from pathlib import Path

    file_path = Path(path)
    if not file_path.is_file():
        if missing_ok:
            return torch.zeros(0, dtype=torch.long), torch.zeros(0, 4, dtype=torch.float32)
        raise FileNotFoundError(f"YOLO label file not found: {file_path}")
    return parse_yolo_text(file_path.read_text(encoding="utf-8"), num_classes=num_classes)


def yolo_boxes_to_xyxy_pixels(boxes_cxcywh_norm: Tensor, image_width: int, image_height: int, clip: bool = True) -> Tensor:
    """Convert normalized cxcywh boxes (N, 4) to absolute-pixel xyxy boxes (N, 4)."""
    if boxes_cxcywh_norm.ndim != 2 or boxes_cxcywh_norm.shape[-1] != 4:
        raise ValueError(f"boxes must have shape (N, 4), got {tuple(boxes_cxcywh_norm.shape)}")
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"image dimensions must be positive, got {image_width}x{image_height}")
    cx, cy, w, h = boxes_cxcywh_norm.float().unbind(-1)
    xyxy = torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), dim=-1)
    xyxy = xyxy * xyxy.new_tensor([image_width, image_height, image_width, image_height])
    if clip:
        xyxy[:, 0::2] = xyxy[:, 0::2].clamp(0, image_width)
        xyxy[:, 1::2] = xyxy[:, 1::2].clamp(0, image_height)
    return xyxy


def yolo_to_target(labels: Tensor, boxes_cxcywh_norm: Tensor, image_width: int, image_height: int, clip: bool = True) -> Dict[str, Tensor]:
    """Build the warehouse target dict {"boxes": xyxy pixels (N, 4), "labels": (N,)}."""
    if labels.shape[0] != boxes_cxcywh_norm.shape[0]:
        raise ValueError(f"labels ({labels.shape[0]}) and boxes ({boxes_cxcywh_norm.shape[0]}) count mismatch")
    return {
        "boxes": yolo_boxes_to_xyxy_pixels(boxes_cxcywh_norm, image_width, image_height, clip=clip),
        "labels": labels.to(torch.long),
    }


def target_to_yolo_lines(boxes_xyxy: Tensor, labels: Tensor, image_width: int, image_height: int, precision: int = 6) -> List[str]:
    """Inverse conversion: xyxy pixel boxes + labels -> YOLO text rows (for exporting labels)."""
    if boxes_xyxy.shape[0] != labels.shape[0]:
        raise ValueError("boxes and labels count mismatch")
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"image dimensions must be positive, got {image_width}x{image_height}")
    x1, y1, x2, y2 = boxes_xyxy.float().unbind(-1)
    cx = (x1 + x2) / 2 / image_width
    cy = (y1 + y2) / 2 / image_height
    w = (x2 - x1) / image_width
    h = (y2 - y1) / image_height
    rows = []
    for c, a, b, d, e in zip(labels.tolist(), cx.tolist(), cy.tolist(), w.tolist(), h.tolist()):
        rows.append(f"{int(c)} {a:.{precision}f} {b:.{precision}f} {d:.{precision}f} {e:.{precision}f}")
    return rows
