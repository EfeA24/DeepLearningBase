# Purpose: Generates xyxy pixel anchors (and anchor-free grid centers) from explicit feature sizes, strides, scales and aspect ratios for dense object detectors.
"""Anchor generation.

Contract:
    Per level i the caller supplies stride s_i (input pixels per feature cell),
    anchor sizes (side length in pixels of a square anchor of equal area) and
    aspect ratios r = height / width. Base anchors: w = size / sqrt(r),
    h = size * sqrt(r), centered on the cell center ((x + offset) * s, (y + offset) * s).
    ``offset=0.5`` centers anchors in the cell; ``offset=0.0`` reproduces the
    torchvision convention (anchor centers at cell top-left corners).

Output ordering (matches models/heads/detection_head.py ``flatten_level_outputs``):
    level i -> Tensor (H_i * W_i * A_i, 4) xyxy absolute pixels, row-major over
    (y, x) and then over anchors (sizes outer loop, ratios inner loop).
    A_i = len(sizes[i]) * len(aspect_ratios[i]).

Anchors are computed on demand from explicit feature sizes; nothing is cached globally.

Requires: torch only.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor


def make_base_anchors(sizes: Sequence[float], aspect_ratios: Sequence[float], dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None) -> Tensor:
    """Zero-centered base anchors (A, 4) xyxy for one level, sizes outer / ratios inner."""
    if not sizes or not aspect_ratios:
        raise ValueError("sizes and aspect_ratios must be non-empty")
    if any(s <= 0 for s in sizes) or any(r <= 0 for r in aspect_ratios):
        raise ValueError("sizes and aspect_ratios must be positive")
    anchors = []
    for size in sizes:
        for ratio in aspect_ratios:
            w = size / math.sqrt(ratio)
            h = size * math.sqrt(ratio)
            anchors.append([-w / 2, -h / 2, w / 2, h / 2])
    return torch.tensor(anchors, dtype=dtype, device=device)


def make_grid_centers(
    feature_size: Tuple[int, int],
    stride: float,
    offset: float = 0.5,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Cell centers (H*W, 2) as (x, y) pixels, row-major over (y, x); used by anchor-free decoders."""
    height, width = feature_size
    if height <= 0 or width <= 0 or stride <= 0:
        raise ValueError("feature_size and stride must be positive")
    ys = (torch.arange(height, dtype=dtype, device=device) + offset) * stride
    xs = (torch.arange(width, dtype=dtype, device=device) + offset) * stride
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)


class AnchorGenerator:
    """Multi-level anchor generator (stateless apart from its configuration).

    Args:
        strides: Stride per level (high-res first).
        sizes: Anchor sizes per level, e.g. ((32,), (64,), (128,)).
        aspect_ratios: Ratios (h / w) per level, or one sequence shared by all levels.
        offset: Cell-center offset in cells (0.5 = center, 0.0 = torchvision convention).
    """

    def __init__(
        self,
        strides: Sequence[float],
        sizes: Sequence[Sequence[float]],
        aspect_ratios: Sequence[Sequence[float]] | Sequence[float] = (0.5, 1.0, 2.0),
        offset: float = 0.5,
    ) -> None:
        if not strides or len(sizes) != len(strides):
            raise ValueError("sizes must provide one sequence per stride level")
        if len(aspect_ratios) > 0 and not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = [list(aspect_ratios)] * len(strides)  # type: ignore[list-item]
        if len(aspect_ratios) != len(strides):
            raise ValueError("aspect_ratios must be shared or provided per level")
        self.strides = [float(s) for s in strides]
        self.sizes = [list(s) for s in sizes]
        self.aspect_ratios = [list(r) for r in aspect_ratios]  # type: ignore[arg-type]
        self.offset = offset

    @property
    def num_anchors_per_location(self) -> List[int]:
        return [len(s) * len(r) for s, r in zip(self.sizes, self.aspect_ratios)]

    def generate(
        self,
        feature_sizes: Sequence[Tuple[int, int]],
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> List[Tensor]:
        """Anchors per level, each (H_i * W_i * A_i, 4) xyxy pixels.

        Args:
            feature_sizes: (H_i, W_i) per level, e.g. ``[f.shape[-2:] for f in features]``.
        """
        if len(feature_sizes) != len(self.strides):
            raise ValueError(f"expected {len(self.strides)} feature sizes, got {len(feature_sizes)}")
        anchors = []
        for (h, w), stride, sizes, ratios in zip(feature_sizes, self.strides, self.sizes, self.aspect_ratios):
            base = make_base_anchors(sizes, ratios, dtype=dtype, device=device)
            centers = make_grid_centers((int(h), int(w)), stride, self.offset, dtype=dtype, device=device)
            shifts = torch.cat([centers, centers], dim=1)
            anchors.append((shifts[:, None, :] + base[None, :, :]).reshape(-1, 4))
        return anchors

    def generate_concatenated(
        self,
        feature_sizes: Sequence[Tuple[int, int]],
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """All levels concatenated in level order -> (sum_i H_i*W_i*A_i, 4)."""
        return torch.cat(self.generate(feature_sizes, dtype, device), dim=0)
