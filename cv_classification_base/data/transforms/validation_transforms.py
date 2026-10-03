# Purpose: Composes deterministic resize, center-crop, tensor conversion and normalization for image-classification validation and test data.
"""Validation / test transforms (no randomness).

Input: PIL.Image in RGB mode. Output: float32 ``(3, image_size, image_size)`` normalized.

Pipeline: Resize(resize_size) [shorter side] -> CenterCrop(image_size) -> ToTensor -> Normalize.
``resize_size`` defaults to ``round(image_size / crop_ratio)`` with ``crop_ratio=0.875``
(224 -> 256, the common ImageNet evaluation convention). Use ``crop_ratio=1.0`` to skip the margin.

Requires: torchvision.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence


def build_validation_transforms(
    image_size: int,
    mean: Sequence[float],
    std: Sequence[float],
    resize_size: Optional[int] = None,
    crop_ratio: float = 0.875,
    interpolation: str = "bilinear",
) -> Any:
    """Return a deterministic ``torchvision.transforms.Compose``."""
    try:
        import torchvision.transforms as T
    except ImportError as exc:
        raise ImportError("Install torchvision to build validation transforms: pip install torchvision") from exc
    if not isinstance(image_size, int) or image_size < 1:
        raise ValueError(f"image_size must be an int >= 1, got {image_size!r}")
    if len(mean) != 3 or len(std) != 3 or any(s <= 0 for s in std):
        raise ValueError("mean/std must have 3 values with std > 0")
    if not 0.0 < crop_ratio <= 1.0:
        raise ValueError(f"crop_ratio must be in (0, 1], got {crop_ratio}")
    if resize_size is None:
        resize_size = int(round(image_size / crop_ratio))
    if resize_size < image_size:
        raise ValueError("resize_size must be >= image_size")
    modes = {"bilinear": T.InterpolationMode.BILINEAR, "bicubic": T.InterpolationMode.BICUBIC}
    if interpolation not in modes:
        raise ValueError(f"interpolation must be one of {sorted(modes)}")
    return T.Compose(
        [
            T.Resize(resize_size, interpolation=modes[interpolation]),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=list(mean), std=list(std)),
        ]
    )
