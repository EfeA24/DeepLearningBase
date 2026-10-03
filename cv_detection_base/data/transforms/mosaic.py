# Purpose: Provides an optional reference 4-image mosaic augmentation for object detection with explicit box offsetting, clipping and label concatenation.
"""Reference 4-image mosaic augmentation (optional; training only).

Algorithm (reference version, not tied to any specific YOLO release):
    1. Create a canvas of ``output_size`` (H, W) filled with ``fill_value``.
    2. Pick a mosaic center (xc, yc) uniformly in the central region
       ``[center_range[0], center_range[1]] * (W, H)`` using the supplied generator.
    3. Place image 0 so its bottom-right corner touches the center (top-left
       quadrant), image 1 bottom-left corner (top-right quadrant), image 2
       top-right corner (bottom-left quadrant), image 3 top-left corner
       (bottom-right quadrant). Images are NOT resized here; pre-resize them
       (e.g. to roughly half the canvas) with box-aware transforms.
    4. Shift each image's boxes by its placement offset, clip to the canvas,
       drop boxes with < ``min_visibility`` of their area remaining or a side
       < ``min_size`` pixels, and concatenate labels and per-box fields.

Contract:
    images:  4 Tensors (C, H_i, W_i) with identical C and dtype.
    targets: 4 dicts with "boxes" (N_i, 4) float xyxy pixels and "labels" (N_i,).
    Returns (canvas (C, H, W), target) where the target holds concatenated
    "boxes"/"labels" (+ "iscrowd"/"difficult"/"weights" if present in ALL inputs).
    Image-level fields (image_id/orig_size) are dropped because the mosaic has no
    single source; add your own bookkeeping if needed.

Requires: torch only.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

OPTIONAL_PER_BOX_KEYS: Tuple[str, ...] = ("iscrowd", "difficult", "weights")


def _rand(generator: Optional[torch.Generator]) -> float:
    return float(torch.rand((), generator=generator).item())


def mosaic4(
    images: Sequence[Tensor],
    targets: Sequence[Dict[str, Any]],
    output_size: Tuple[int, int],
    generator: torch.Generator,
    center_range: Tuple[float, float] = (0.25, 0.75),
    fill_value: float = 0.5,
    min_visibility: float = 0.2,
    min_size: float = 2.0,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Combine 4 images and their targets into one mosaic sample. See module docstring."""
    if len(images) != 4 or len(targets) != 4:
        raise ValueError("mosaic4 needs exactly 4 images and 4 targets")
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator")
    channels, dtype = images[0].shape[0], images[0].dtype
    for img in images:
        if img.ndim != 3 or img.shape[0] != channels or img.dtype != dtype:
            raise ValueError("all images must be (C, H, W) with the same C and dtype")
    lo, hi = center_range
    if not 0.0 <= lo <= hi <= 1.0:
        raise ValueError("center_range must satisfy 0 <= low <= high <= 1")

    out_h, out_w = int(output_size[0]), int(output_size[1])
    canvas = torch.full((channels, out_h, out_w), fill_value, dtype=dtype, device=images[0].device)
    xc = int(out_w * (lo + (hi - lo) * _rand(generator)))
    yc = int(out_h * (lo + (hi - lo) * _rand(generator)))

    all_boxes: List[Tensor] = []
    all_labels: List[Tensor] = []
    extras: Dict[str, List[Tensor]] = {k: [] for k in OPTIONAL_PER_BOX_KEYS if all(k in t for t in targets)}

    for quadrant, (img, target) in enumerate(zip(images, targets)):
        h, w = img.shape[-2:]
        if quadrant == 0:
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            x1b, y1b = w - (x2a - x1a), h - (y2a - y1a)
        elif quadrant == 1:
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, out_w), yc
            x1b, y1b = 0, h - (y2a - y1a)
        elif quadrant == 2:
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(yc + h, out_h)
            x1b, y1b = w - (x2a - x1a), 0
        else:
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, out_w), min(yc + h, out_h)
            x1b, y1b = 0, 0
        region_w, region_h = x2a - x1a, y2a - y1a
        if region_w > 0 and region_h > 0:
            canvas[:, y1a:y2a, x1a:x2a] = img[:, y1b : y1b + region_h, x1b : x1b + region_w]

        boxes = target["boxes"].to(torch.float32)
        offset_x, offset_y = x1a - x1b, y1a - y1b
        shifted = boxes + boxes.new_tensor([offset_x, offset_y, offset_x, offset_y])
        region = shifted.new_tensor([x1a, y1a, x2a, y2a])
        clipped = shifted.clone()
        clipped[:, 0::2] = torch.maximum(torch.minimum(clipped[:, 0::2], region[2]), region[0])
        clipped[:, 1::2] = torch.maximum(torch.minimum(clipped[:, 1::2], region[3]), region[1])
        area = (shifted[:, 2] - shifted[:, 0]).clamp(min=0) * (shifted[:, 3] - shifted[:, 1]).clamp(min=0)
        kept_w = clipped[:, 2] - clipped[:, 0]
        kept_h = clipped[:, 3] - clipped[:, 1]
        visibility = (kept_w * kept_h) / area.clamp(min=1e-7)
        keep = (visibility >= min_visibility) & (kept_w >= min_size) & (kept_h >= min_size)
        all_boxes.append(clipped[keep])
        all_labels.append(target["labels"][keep])
        for key in extras:
            extras[key].append(target[key][keep])

    out: Dict[str, Tensor] = {
        "boxes": torch.cat(all_boxes, dim=0).reshape(-1, 4),
        "labels": torch.cat(all_labels, dim=0),
    }
    for key, values in extras.items():
        out[key] = torch.cat(values, dim=0)
    return canvas, out
