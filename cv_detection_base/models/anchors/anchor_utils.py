# Purpose: Encodes and decodes anchor-relative box offsets (dx, dy, dw, dh with weights) for anchor-based object detectors, validating shapes and units.
"""Anchor box coding (R-CNN style parameterization).

Encoding (boxes and anchors both xyxy, SAME units - normally absolute pixels):
    dx = wx * (gx - ax) / aw        dy = wy * (gy - ay) / ah
    dw = ww * log(gw / aw)          dh = wh * log(gh / ah)
where (gx, gy, gw, gh) / (ax, ay, aw, ah) are center/size of box / anchor and
``weights = (wx, wy, ww, wh)`` (e.g. (1, 1, 1, 1), or (10, 10, 5, 5) as in Faster R-CNN).

Decoding inverts this; dw/dh are clamped to ``log(1000 / 16)`` before exp so
early-training predictions cannot overflow.

Validation: shapes must match (N, 4); anchors must have positive width/height.
A heuristic warning is emitted when one tensor looks normalized (max <= 1) and
the other looks like pixels (max > 2), the most common unit mix-up.

Requires: torch only.
"""
from __future__ import annotations

import math
import warnings
from typing import Sequence

import torch
from torch import Tensor

DEFAULT_SCALE_CLAMP = math.log(1000.0 / 16)


def _check_units(boxes: Tensor, anchors: Tensor) -> None:
    if boxes.numel() == 0 or anchors.numel() == 0:
        return
    boxes_max = float(boxes.detach().abs().max())
    anchors_max = float(anchors.detach().abs().max())
    if (boxes_max <= 1.0 < 2.0 < anchors_max) or (anchors_max <= 1.0 < 2.0 < boxes_max):
        warnings.warn(
            "boxes and anchors appear to use different units (normalized vs pixels); "
            "convert them to the same unit before encoding/decoding",
            stacklevel=3,
        )


def _validate(boxes: Tensor, anchors: Tensor, weights: Sequence[float]) -> None:
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError(f"boxes/deltas must be (N, 4), got {tuple(boxes.shape)}")
    if anchors.shape != boxes.shape:
        raise ValueError(f"anchors must match shape {tuple(boxes.shape)}, got {tuple(anchors.shape)}")
    if len(weights) != 4 or any(w <= 0 for w in weights):
        raise ValueError("weights must be 4 positive numbers (wx, wy, ww, wh)")
    if anchors.numel() and (((anchors[:, 2] - anchors[:, 0]) <= 0).any() or ((anchors[:, 3] - anchors[:, 1]) <= 0).any()):
        raise ValueError("anchors must have positive width and height (xyxy)")


def encode_boxes(
    boxes: Tensor,
    anchors: Tensor,
    weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
    eps: float = 1e-6,
) -> Tensor:
    """Encode matched ground-truth boxes (N, 4) relative to anchors (N, 4) -> deltas (N, 4)."""
    _validate(boxes, anchors, weights)
    _check_units(boxes, anchors)
    wx, wy, ww, wh = weights
    boxes = boxes.float() if boxes.dtype in (torch.float16, torch.bfloat16) else boxes
    anchors = anchors.to(boxes.dtype)
    aw = anchors[:, 2] - anchors[:, 0]
    ah = anchors[:, 3] - anchors[:, 1]
    ax = anchors[:, 0] + 0.5 * aw
    ay = anchors[:, 1] + 0.5 * ah
    gw = (boxes[:, 2] - boxes[:, 0]).clamp(min=eps)
    gh = (boxes[:, 3] - boxes[:, 1]).clamp(min=eps)
    gx = boxes[:, 0] + 0.5 * (boxes[:, 2] - boxes[:, 0])
    gy = boxes[:, 1] + 0.5 * (boxes[:, 3] - boxes[:, 1])
    dx = wx * (gx - ax) / aw
    dy = wy * (gy - ay) / ah
    dw = ww * torch.log(gw / aw)
    dh = wh * torch.log(gh / ah)
    return torch.stack((dx, dy, dw, dh), dim=1)


def decode_boxes(
    deltas: Tensor,
    anchors: Tensor,
    weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
    scale_clamp: float = DEFAULT_SCALE_CLAMP,
) -> Tensor:
    """Decode deltas (N, 4) relative to anchors (N, 4) -> xyxy boxes (N, 4) in the anchors' units."""
    _validate(deltas, anchors, weights)
    wx, wy, ww, wh = weights
    deltas = deltas.float() if deltas.dtype in (torch.float16, torch.bfloat16) else deltas
    anchors = anchors.to(deltas.dtype)
    aw = anchors[:, 2] - anchors[:, 0]
    ah = anchors[:, 3] - anchors[:, 1]
    ax = anchors[:, 0] + 0.5 * aw
    ay = anchors[:, 1] + 0.5 * ah
    dx = deltas[:, 0] / wx
    dy = deltas[:, 1] / wy
    dw = (deltas[:, 2] / ww).clamp(max=scale_clamp)
    dh = (deltas[:, 3] / wh).clamp(max=scale_clamp)
    cx = dx * aw + ax
    cy = dy * ah + ay
    w = torch.exp(dw) * aw
    h = torch.exp(dh) * ah
    return torch.stack((cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h), dim=1)
