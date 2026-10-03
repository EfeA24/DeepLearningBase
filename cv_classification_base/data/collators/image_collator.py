# Purpose: Collates (image tensor, label) examples into stacked image and label batches with clear errors for inconsistent sizes or types.
"""Image classification collator.

Input: a list of ``(image, target)`` pairs where ``image`` is a tensor ``(C, H, W)`` (all with the
same shape and dtype - resize/crop in the transforms first) and ``target`` is either
    - an int / 0-d integer tensor (single-label) -> int64 ``(B,)``, or
    - a 1D float tensor ``(L,)`` multi-hot (multi-label) -> float ``(B, L)``.
Output: ``(images (B, C, H, W), targets)``.

Errors name the offending example index so data issues are easy to locate.

Requires: torch.
"""
from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import torch


def collate_image_classification(batch: Sequence[Tuple[Any, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(batch) == 0:
        raise ValueError("cannot collate an empty batch")
    images: List[torch.Tensor] = []
    targets: List[Any] = []
    for i, example in enumerate(batch):
        if not isinstance(example, (tuple, list)) or len(example) != 2:
            raise TypeError(f"example {i} must be an (image, target) pair, got {type(example).__name__}")
        image, target = example
        if not isinstance(image, torch.Tensor):
            raise TypeError(
                f"example {i}: image is {type(image).__name__}; apply a ToTensor-style transform first"
            )
        if image.dim() != 3:
            raise ValueError(f"example {i}: image must be (C, H, W), got {tuple(image.shape)}")
        images.append(image)
        targets.append(target)

    ref = images[0]
    for i, image in enumerate(images[1:], start=1):
        if image.shape != ref.shape:
            raise ValueError(
                f"example {i}: image shape {tuple(image.shape)} differs from example 0 "
                f"{tuple(ref.shape)}; resize/crop to a fixed size in the transforms"
            )
        if image.dtype != ref.dtype:
            raise TypeError(f"example {i}: dtype {image.dtype} differs from example 0 dtype {ref.dtype}")

    return torch.stack(images), _collate_targets(targets)


def _collate_targets(targets: List[Any]) -> torch.Tensor:
    first = targets[0]
    if isinstance(first, torch.Tensor) and first.dim() == 1:
        for i, t in enumerate(targets):
            if not isinstance(t, torch.Tensor) or t.shape != first.shape:
                raise ValueError(f"example {i}: multi-label target shape mismatch")
        return torch.stack(targets).float()
    out = []
    for i, t in enumerate(targets):
        if isinstance(t, torch.Tensor):
            if t.dim() != 0 or t.is_floating_point():
                raise TypeError(f"example {i}: single-label target must be an int or 0-d int tensor")
            out.append(int(t))
        elif isinstance(t, int) and not isinstance(t, bool):
            out.append(t)
        else:
            raise TypeError(f"example {i}: unsupported target type {type(t).__name__}")
    return torch.tensor(out, dtype=torch.long)
