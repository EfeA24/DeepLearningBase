# Purpose: Provides shared torchvision (lazy import) construction helpers for image-classification preprocessing with explicit sizes and statistics.
"""Shared transform construction helpers.

All builders import ``torchvision.transforms`` (v1 API, PIL or tensor input) lazily, so importing
this module does not require torchvision. Image size and mean/std are always explicit arguments;
``IMAGENET_MEAN``/``IMAGENET_STD`` are provided as named constants for torchvision pretrained
weights, not applied implicitly.

Input format expected by the composed pipelines: PIL.Image in RGB mode (H, W, 3, uint8).
``to_tensor_and_normalize`` produces float32 ``(3, H, W)`` normalized tensors.

Requires: torchvision.
"""
from __future__ import annotations

from typing import Any, List, Sequence, Tuple

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


def require_torchvision_transforms() -> Any:
    """Return the ``torchvision.transforms`` module or raise an informative ImportError."""
    try:
        import torchvision.transforms as T
    except ImportError as exc:
        raise ImportError("Install torchvision to build image transforms: pip install torchvision") from exc
    return T


def validate_stats(mean: Sequence[float], std: Sequence[float], channels: int = 3) -> None:
    if len(mean) != channels or len(std) != channels:
        raise ValueError(f"mean and std must each have {channels} values")
    if any(s <= 0 for s in std):
        raise ValueError("std values must be > 0")


def validate_size(size: int, name: str = "image_size") -> None:
    if not isinstance(size, int) or size < 1:
        raise ValueError(f"{name} must be an int >= 1, got {size!r}")


def interpolation_mode(name: str) -> Any:
    """Map "bilinear"/"bicubic"/"nearest" to ``torchvision.transforms.InterpolationMode``."""
    T = require_torchvision_transforms()
    modes = {
        "bilinear": T.InterpolationMode.BILINEAR,
        "bicubic": T.InterpolationMode.BICUBIC,
        "nearest": T.InterpolationMode.NEAREST,
    }
    if name not in modes:
        raise ValueError(f"interpolation must be one of {sorted(modes)}, got {name!r}")
    return modes[name]


def to_tensor_and_normalize(mean: Sequence[float], std: Sequence[float]) -> List[Any]:
    """``[ToTensor(), Normalize(mean, std)]``: PIL uint8 [0,255] -> float [0,1] -> normalized."""
    validate_stats(mean, std)
    T = require_torchvision_transforms()
    return [T.ToTensor(), T.Normalize(mean=list(mean), std=list(std))]


def compose(transforms: Sequence[Any]) -> Any:
    """``torchvision.transforms.Compose`` over the given callables."""
    T = require_torchvision_transforms()
    return T.Compose(list(transforms))
