# Purpose: Composes deterministic joint image+box transforms (resize, flip, normalize, clip, degenerate filtering) for object-detection preprocessing.
"""Joint (image, target) transforms for detection.

Every transform is a callable ``(image, target) -> (image, target)`` so image
geometry and boxes always change together. Randomized training augmentation
lives in augmentation.py; the transforms here are deterministic and suitable
for validation/test preprocessing as well.

Contract:
    image:  Tensor (C, H, W). Float images are assumed RGB in [0, 1] unless a
            Normalize was already applied; uint8 images are converted by ConvertImageDtype.
    target: {"boxes": (N, 4) float xyxy absolute pixels, "labels": (N,), ...}
            Per-box fields ("labels", "iscrowd", "difficult", "area", "weights")
            are filtered together with boxes. Targets are shallow-copied, never mutated.

Requires: torch only.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

Target = Dict[str, Any]
JointTransform = Callable[[Tensor, Target], Tuple[Tensor, Target]]
PER_BOX_KEYS: Tuple[str, ...] = ("boxes", "labels", "iscrowd", "difficult", "area", "weights", "scores")


def _check_image(image: Tensor) -> None:
    if not isinstance(image, Tensor) or image.ndim != 3:
        raise TypeError(f"image must be a Tensor (C, H, W), got {type(image).__name__} {getattr(image, 'shape', '')}")


class Compose:
    """Apply joint transforms in order."""

    def __init__(self, transforms: Sequence[JointTransform]) -> None:
        self.transforms = list(transforms)

    def __call__(self, image: Tensor, target: Target) -> Tuple[Tensor, Target]:
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target

    def __repr__(self) -> str:
        inner = ", ".join(type(t).__name__ for t in self.transforms)
        return f"Compose([{inner}])"


class ConvertImageDtype:
    """Convert uint8 [0, 255] images to float [0, 1] (float inputs are cast only)."""

    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        if not dtype.is_floating_point:
            raise ValueError("ConvertImageDtype targets floating dtypes only")
        self.dtype = dtype

    def __call__(self, image: Tensor, target: Target) -> Tuple[Tensor, Target]:
        _check_image(image)
        if image.dtype == torch.uint8:
            return image.to(self.dtype).div_(255.0), target
        return image.to(self.dtype), target


class Resize:
    """Resize the image and scale boxes accordingly (aspect ratio may change).

    Args:
        size: Output (height, width).
        mode: ``F.interpolate`` mode ("bilinear", "nearest", ...).
    """

    def __init__(self, size: Tuple[int, int], mode: str = "bilinear") -> None:
        if len(size) != 2 or min(size) <= 0:
            raise ValueError(f"size must be (height, width) with positive values, got {size}")
        self.size = (int(size[0]), int(size[1]))
        self.mode = mode

    def __call__(self, image: Tensor, target: Target) -> Tuple[Tensor, Target]:
        _check_image(image)
        old_h, old_w = image.shape[-2:]
        new_h, new_w = self.size
        kwargs = {"align_corners": False} if self.mode in ("bilinear", "bicubic") else {}
        resized = F.interpolate(image[None].float(), size=self.size, mode=self.mode, **kwargs)[0].to(image.dtype)
        sx, sy = new_w / old_w, new_h / old_h
        out = dict(target)
        out["boxes"] = target["boxes"] * target["boxes"].new_tensor([sx, sy, sx, sy])
        if isinstance(target.get("area"), Tensor):
            out["area"] = target["area"] * (sx * sy)
        return resized, out


class ResizeShortestEdge:
    """Resize keeping aspect ratio so the short side equals ``short_size`` and the long side <= ``max_size``."""

    def __init__(self, short_size: int, max_size: Optional[int] = None, mode: str = "bilinear") -> None:
        if short_size <= 0 or (max_size is not None and max_size < short_size):
            raise ValueError("require short_size > 0 and max_size >= short_size")
        self.short_size = short_size
        self.max_size = max_size
        self.mode = mode

    def __call__(self, image: Tensor, target: Target) -> Tuple[Tensor, Target]:
        _check_image(image)
        h, w = image.shape[-2:]
        scale = self.short_size / min(h, w)
        if self.max_size is not None and max(h, w) * scale > self.max_size:
            scale = self.max_size / max(h, w)
        new_size = (max(1, round(h * scale)), max(1, round(w * scale)))
        return Resize(new_size, self.mode)(image, target)


class HorizontalFlip:
    """Deterministic horizontal flip of image and boxes (use augmentation.py for random flips)."""

    def __call__(self, image: Tensor, target: Target) -> Tuple[Tensor, Target]:
        _check_image(image)
        width = image.shape[-1]
        boxes = target["boxes"]
        flipped = boxes.clone()
        flipped[:, 0] = width - boxes[:, 2]
        flipped[:, 2] = width - boxes[:, 0]
        out = dict(target)
        out["boxes"] = flipped
        return image.flip(-1), out


class VerticalFlip:
    """Deterministic vertical flip of image and boxes."""

    def __call__(self, image: Tensor, target: Target) -> Tuple[Tensor, Target]:
        _check_image(image)
        height = image.shape[-2]
        boxes = target["boxes"]
        flipped = boxes.clone()
        flipped[:, 1] = height - boxes[:, 3]
        flipped[:, 3] = height - boxes[:, 1]
        out = dict(target)
        out["boxes"] = flipped
        return image.flip(-2), out


class Normalize:
    """Channel-wise ``(image - mean) / std`` on a float image; boxes untouched."""

    def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None:
        if len(mean) != len(std) or any(s <= 0 for s in std):
            raise ValueError("mean and std must have equal length and std must be positive")
        self.mean = tuple(float(m) for m in mean)
        self.std = tuple(float(s) for s in std)

    def __call__(self, image: Tensor, target: Target) -> Tuple[Tensor, Target]:
        _check_image(image)
        if not image.is_floating_point():
            raise TypeError("Normalize expects a float image; apply ConvertImageDtype first")
        if image.shape[0] != len(self.mean):
            raise ValueError(f"image has {image.shape[0]} channels but mean/std have {len(self.mean)}")
        mean = image.new_tensor(self.mean)[:, None, None]
        std = image.new_tensor(self.std)[:, None, None]
        return (image - mean) / std, target


class ClipBoxes:
    """Clamp boxes to the current image bounds."""

    def __call__(self, image: Tensor, target: Target) -> Tuple[Tensor, Target]:
        _check_image(image)
        height, width = image.shape[-2:]
        boxes = target["boxes"].clone()
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, width)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, height)
        out = dict(target)
        out["boxes"] = boxes
        return image, out


class RemoveDegenerateBoxes:
    """Drop boxes with width or height < ``min_size`` pixels, filtering all per-box fields."""

    def __init__(self, min_size: float = 1.0, per_box_keys: Sequence[str] = PER_BOX_KEYS) -> None:
        if min_size < 0:
            raise ValueError("min_size must be >= 0")
        self.min_size = min_size
        self.per_box_keys = tuple(per_box_keys)

    def __call__(self, image: Tensor, target: Target) -> Tuple[Tensor, Target]:
        boxes = target["boxes"]
        keep = (
            torch.isfinite(boxes).all(dim=1)
            & ((boxes[:, 2] - boxes[:, 0]) >= self.min_size)
            & ((boxes[:, 3] - boxes[:, 1]) >= self.min_size)
        )
        num_boxes = boxes.shape[0]
        out = dict(target)
        for key in self.per_box_keys:
            value = target.get(key)
            if isinstance(value, Tensor) and value.ndim >= 1 and value.shape[0] == num_boxes:
                out[key] = value[keep]
        return image, out


class Lambda:
    """Wrap an image-only callable (boxes unchanged), e.g. a photometric op that never moves pixels."""

    def __init__(self, image_fn: Callable[[Tensor], Tensor]) -> None:
        self.image_fn = image_fn

    def __call__(self, image: Tensor, target: Target) -> Tuple[Tensor, Target]:
        new_image = self.image_fn(image)
        if isinstance(new_image, Tensor) and new_image.shape[-2:] != image.shape[-2:]:
            raise ValueError("Lambda image_fn changed the spatial size; use a box-aware transform instead")
        return new_image, target


def build_eval_transforms(
    size: Optional[Union[int, Tuple[int, int]]] = None,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
) -> Compose:
    """Deterministic validation/test preprocessing: dtype -> optional resize -> optional normalize.

    ``size`` may be an int (shortest edge) or (height, width). Statistics must be
    supplied explicitly (e.g. computed on the training set or from the backbone's pretraining).
    """
    transforms: list = [ConvertImageDtype()]
    if isinstance(size, int):
        transforms.append(ResizeShortestEdge(size))
    elif size is not None:
        transforms.append(Resize(size))
    if (mean is None) != (std is None):
        raise ValueError("pass both mean and std, or neither")
    if mean is not None and std is not None:
        transforms.append(Normalize(mean, std))
    return Compose(transforms)
