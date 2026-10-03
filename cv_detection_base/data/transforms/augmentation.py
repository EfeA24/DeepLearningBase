# Purpose: Provides randomized, box-aware training augmentations for object detection driven by an explicit torch.Generator.
"""Detection-safe training augmentation.

Every op is a joint callable ``(image, target) -> (image, target)`` and draws
randomness ONLY from the ``torch.Generator`` passed to its constructor (CPU
generator), so runs are reproducible from one seed. Use for training only;
keep validation/test preprocessing deterministic (detection_transforms.py).

DataLoader caveat: each worker process receives a copy of the generator in the
same state, so workers would produce identical random streams. Reseed per
worker, e.g. ``worker_init_fn=lambda wid: aug_generator.manual_seed(base_seed + wid)``
(and include the epoch in the seed if you want different streams per epoch).

Contract:
    image:  float Tensor (C, H, W), RGB, values in [0, 1] (photometric ops clamp to [0, 1]).
    target: {"boxes": (N, 4) float xyxy absolute pixels, "labels": (N,), ...};
            per-box fields are filtered together with boxes; targets are never mutated.

Requires: torch only.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

Target = Dict[str, Any]
PER_BOX_KEYS: Tuple[str, ...] = ("boxes", "labels", "iscrowd", "difficult", "area", "weights", "scores")


def _check_generator(generator: torch.Generator) -> None:
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator (e.g. torch.Generator().manual_seed(0))")


def _rand(generator: torch.Generator) -> float:
    return float(torch.rand((), generator=generator).item())


def _uniform(generator: torch.Generator, low: float, high: float) -> float:
    return low + (high - low) * _rand(generator)


def _filter(target: Target, keep: Tensor) -> Target:
    num_boxes = target["boxes"].shape[0]
    out = dict(target)
    for key in PER_BOX_KEYS:
        value = target.get(key)
        if isinstance(value, Tensor) and value.ndim >= 1 and value.shape[0] == num_boxes:
            out[key] = value[keep]
    return out


class RandomHorizontalFlip:
    """Flip image and boxes horizontally with probability ``p``."""

    def __init__(self, p: float, generator: torch.Generator) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1]")
        _check_generator(generator)
        self.p = p
        self.generator = generator

    def __call__(self, image: Tensor, target: Target) -> Tuple[Tensor, Target]:
        if _rand(self.generator) >= self.p:
            return image, target
        width = image.shape[-1]
        boxes = target["boxes"]
        flipped = boxes.clone()
        flipped[:, 0] = width - boxes[:, 2]
        flipped[:, 2] = width - boxes[:, 0]
        out = dict(target)
        out["boxes"] = flipped
        return image.flip(-1), out


class RandomVerticalFlip:
    """Flip image and boxes vertically with probability ``p`` (only if vertical flips are semantically valid)."""

    def __init__(self, p: float, generator: torch.Generator) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1]")
        _check_generator(generator)
        self.p = p
        self.generator = generator

    def __call__(self, image: Tensor, target: Target) -> Tuple[Tensor, Target]:
        if _rand(self.generator) >= self.p:
            return image, target
        height = image.shape[-2]
        boxes = target["boxes"]
        flipped = boxes.clone()
        flipped[:, 1] = height - boxes[:, 3]
        flipped[:, 3] = height - boxes[:, 1]
        out = dict(target)
        out["boxes"] = flipped
        return image.flip(-2), out


class RandomPhotometricDistort:
    """Random brightness/contrast/saturation jitter (pixels move nowhere, boxes unchanged).

    Each factor is drawn uniformly from ``[1 - x, 1 + x]``. Saturation needs 3 RGB channels.
    """

    def __init__(self, generator: torch.Generator, brightness: float = 0.2, contrast: float = 0.2, saturation: float = 0.2) -> None:
        _check_generator(generator)
        for name, value in (("brightness", brightness), ("contrast", contrast), ("saturation", saturation)):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        self.generator = generator
        self.brightness, self.contrast, self.saturation = brightness, contrast, saturation

    def __call__(self, image: Tensor, target: Target) -> Tuple[Tensor, Target]:
        if not image.is_floating_point():
            raise TypeError("RandomPhotometricDistort expects a float image in [0, 1]")
        g = self.generator
        out = image * _uniform(g, 1 - self.brightness, 1 + self.brightness)
        mean = out.mean()
        out = (out - mean) * _uniform(g, 1 - self.contrast, 1 + self.contrast) + mean
        if out.shape[0] == 3 and self.saturation > 0:
            gray = (0.299 * out[0] + 0.587 * out[1] + 0.114 * out[2])[None]
            out = (out - gray) * _uniform(g, 1 - self.saturation, 1 + self.saturation) + gray
        return out.clamp(0.0, 1.0), target


class RandomScale:
    """Resize by a random factor in ``scale_range`` (aspect ratio preserved), scaling boxes."""

    def __init__(self, scale_range: Tuple[float, float], generator: torch.Generator, mode: str = "bilinear") -> None:
        low, high = scale_range
        if not 0 < low <= high:
            raise ValueError("scale_range must satisfy 0 < low <= high")
        _check_generator(generator)
        self.scale_range = (float(low), float(high))
        self.generator = generator
        self.mode = mode

    def __call__(self, image: Tensor, target: Target) -> Tuple[Tensor, Target]:
        scale = _uniform(self.generator, *self.scale_range)
        h, w = image.shape[-2:]
        new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
        kwargs = {"align_corners": False} if self.mode in ("bilinear", "bicubic") else {}
        resized = F.interpolate(image[None].float(), size=(new_h, new_w), mode=self.mode, **kwargs)[0].to(image.dtype)
        sx, sy = new_w / w, new_h / h
        out = dict(target)
        out["boxes"] = target["boxes"] * target["boxes"].new_tensor([sx, sy, sx, sy])
        if isinstance(target.get("area"), Tensor):
            out["area"] = target["area"] * (sx * sy)
        return resized, out


class RandomCropWithBoxes:
    """Random crop that clips boxes and drops those with too little remaining area.

    Args:
        generator: RNG.
        min_scale, max_scale: Crop side length as a fraction of the image side.
        min_visibility: Keep a box only if this fraction of its area survives the crop.
        min_size: Also drop boxes smaller than this (pixels) after clipping.
        p: Probability of cropping at all.
    """

    def __init__(
        self,
        generator: torch.Generator,
        min_scale: float = 0.5,
        max_scale: float = 1.0,
        min_visibility: float = 0.3,
        min_size: float = 1.0,
        p: float = 0.5,
    ) -> None:
        _check_generator(generator)
        if not 0 < min_scale <= max_scale <= 1.0:
            raise ValueError("require 0 < min_scale <= max_scale <= 1")
        if not 0.0 <= min_visibility <= 1.0 or not 0.0 <= p <= 1.0:
            raise ValueError("min_visibility and p must be in [0, 1]")
        self.generator = generator
        self.min_scale, self.max_scale = min_scale, max_scale
        self.min_visibility = min_visibility
        self.min_size = min_size
        self.p = p

    def __call__(self, image: Tensor, target: Target) -> Tuple[Tensor, Target]:
        g = self.generator
        if _rand(g) >= self.p:
            return image, target
        h, w = image.shape[-2:]
        crop_h = max(1, round(h * _uniform(g, self.min_scale, self.max_scale)))
        crop_w = max(1, round(w * _uniform(g, self.min_scale, self.max_scale)))
        top = int(_rand(g) * (h - crop_h + 1))
        left = int(_rand(g) * (w - crop_w + 1))
        top, left = min(top, h - crop_h), min(left, w - crop_w)

        boxes = target["boxes"]
        shifted = boxes - boxes.new_tensor([left, top, left, top])
        clipped = shifted.clone()
        clipped[:, 0::2] = clipped[:, 0::2].clamp(0, crop_w)
        clipped[:, 1::2] = clipped[:, 1::2].clamp(0, crop_h)
        area_orig = (shifted[:, 2] - shifted[:, 0]).clamp(min=0) * (shifted[:, 3] - shifted[:, 1]).clamp(min=0)
        area_clip = (clipped[:, 2] - clipped[:, 0]) * (clipped[:, 3] - clipped[:, 1])
        visibility = area_clip / area_orig.clamp(min=1e-7)
        keep = (
            (visibility >= self.min_visibility)
            & ((clipped[:, 2] - clipped[:, 0]) >= self.min_size)
            & ((clipped[:, 3] - clipped[:, 1]) >= self.min_size)
        )
        out = dict(target)
        out["boxes"] = clipped
        if isinstance(target.get("area"), Tensor):
            out["area"] = target["area"] * visibility
        out = _filter(out, keep)
        return image[:, top : top + crop_h, left : left + crop_w], out


class RandomApply:
    """Apply a joint transform with probability ``p``."""

    def __init__(self, transform: Callable[[Tensor, Target], Tuple[Tensor, Target]], p: float, generator: torch.Generator) -> None:
        _check_generator(generator)
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1]")
        self.transform, self.p, self.generator = transform, p, generator

    def __call__(self, image: Tensor, target: Target) -> Tuple[Tensor, Target]:
        if _rand(self.generator) < self.p:
            return self.transform(image, target)
        return image, target


class AugmentationPipeline:
    """Ordered composition of joint augmentations (same semantics as detection_transforms.Compose)."""

    def __init__(self, transforms: Sequence[Callable[[Tensor, Target], Tuple[Tensor, Target]]]) -> None:
        self.transforms = list(transforms)

    def __call__(self, image: Tensor, target: Target) -> Tuple[Tensor, Target]:
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


def build_train_augmentation(
    generator: torch.Generator,
    hflip_p: float = 0.5,
    scale_range: Optional[Tuple[float, float]] = None,
    crop_p: float = 0.0,
    photometric: bool = True,
    extra: Optional[Sequence[Callable[[Tensor, Target], Tuple[Tensor, Target]]]] = None,
) -> AugmentationPipeline:
    """Build a small, explicit training recipe: photometric -> scale -> crop -> flip -> extras.

    Normalization and final resizing/padding are NOT included; append them
    explicitly (they belong to the shared train/eval preprocessing).
    """
    _check_generator(generator)
    ops: List[Callable[[Tensor, Target], Tuple[Tensor, Target]]] = []
    if photometric:
        ops.append(RandomPhotometricDistort(generator))
    if scale_range is not None:
        ops.append(RandomScale(scale_range, generator))
    if crop_p > 0:
        ops.append(RandomCropWithBoxes(generator, p=crop_p))
    if hflip_p > 0:
        ops.append(RandomHorizontalFlip(hflip_p, generator))
    if extra:
        ops.extend(extra)
    return AugmentationPipeline(ops)
