# Purpose: Composes training-time preprocessing and augmentation (random resized crop, flip, optional color jitter) for image classification.
"""Training transforms.

Input: PIL.Image in RGB mode. Output: float32 ``(3, image_size, image_size)`` normalized with the
caller's ``mean``/``std``.

Pipeline:
    RandomResizedCrop(image_size, scale, ratio) -> RandomHorizontalFlip(hflip_prob)
    -> [ColorJitter(b, c, s, h) if color_jitter] -> [extra PIL-level transforms]
    -> ToTensor -> Normalize -> [RandomErasing(p) if random_erasing_prob > 0]

Only for the TRAIN dataset. Validation/test must use validation_transforms.py. Horizontal flips
assume class semantics are mirror-invariant (not true for e.g. text or left/right labels; set
``hflip_prob=0`` then).

Requires: torchvision.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple


def build_train_transforms(
    image_size: int,
    mean: Sequence[float],
    std: Sequence[float],
    scale: Tuple[float, float] = (0.08, 1.0),
    ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
    hflip_prob: float = 0.5,
    color_jitter: Optional[Tuple[float, float, float, float]] = None,
    extra_pil_transforms: Sequence[Any] = (),
    random_erasing_prob: float = 0.0,
    interpolation: str = "bilinear",
) -> Any:
    """Return a ``torchvision.transforms.Compose`` for training images."""
    try:
        import torchvision.transforms as T
    except ImportError as exc:
        raise ImportError("Install torchvision to build train transforms: pip install torchvision") from exc
    if not isinstance(image_size, int) or image_size < 1:
        raise ValueError(f"image_size must be an int >= 1, got {image_size!r}")
    if len(mean) != 3 or len(std) != 3 or any(s <= 0 for s in std):
        raise ValueError("mean/std must have 3 values with std > 0")
    if not 0.0 < scale[0] <= scale[1] <= 1.0:
        raise ValueError(f"scale must satisfy 0 < min <= max <= 1, got {scale}")
    if not 0.0 <= hflip_prob <= 1.0 or not 0.0 <= random_erasing_prob <= 1.0:
        raise ValueError("probabilities must be in [0, 1]")
    modes = {"bilinear": T.InterpolationMode.BILINEAR, "bicubic": T.InterpolationMode.BICUBIC}
    if interpolation not in modes:
        raise ValueError(f"interpolation must be one of {sorted(modes)}")

    steps = [
        T.RandomResizedCrop(image_size, scale=scale, ratio=ratio, interpolation=modes[interpolation]),
    ]
    if hflip_prob > 0:
        steps.append(T.RandomHorizontalFlip(p=hflip_prob))
    if color_jitter is not None:
        brightness, contrast, saturation, hue = color_jitter
        steps.append(T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue))
    steps.extend(extra_pil_transforms)
    steps += [T.ToTensor(), T.Normalize(mean=list(mean), std=list(std))]
    if random_erasing_prob > 0:
        steps.append(T.RandomErasing(p=random_erasing_prob))
    return T.Compose(steps)
