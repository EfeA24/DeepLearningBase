# Purpose: Batches object-detection samples with variable box counts, stacking or bottom/right-padding images and returning a list of target dicts.
"""Detection batch collation.

Input: a list of ``(image, target)`` samples where image is a Tensor (C, H_i, W_i)
and target follows the warehouse schema ("boxes" (N_i, 4) xyxy pixels, "labels" (N_i,)).

Output: ``(images, targets)``
    mode="stack": images is a Tensor (B, C, H, W); all images must share a size.
    mode="pad":   images is a Tensor (B, C, H_max, W_max) (optionally rounded up
                  to a multiple of ``size_divisible``). Padding is added at the
                  bottom/right only, filled with ``pad_value``, so box coordinates
                  stay valid without any shift. Pick ``pad_value`` consistent with
                  your normalization (0.0 after mean subtraction ~ "mean color").
    mode="list":  images is a list of Tensors (no stacking).
    targets is always a list of B dicts (shallow copies). Each copy gains
    "image_size": LongTensor (2,) = (H_i, W_i), the un-padded size, so
    post-processing can clip boxes to real image content. Variable box counts
    are never padded; per-image tensors keep their own N_i (possibly 0).

Requires: torch only.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple, Union

import torch
from torch import Tensor

Sample = Tuple[Tensor, Dict[str, Any]]


class DetectionCollator:
    """Collate detection samples. See module docstring for the output contract.

    Args:
        mode: "stack", "pad" or "list".
        pad_value: Fill value for padded pixels (mode="pad").
        size_divisible: Round padded H/W up to a multiple of this (mode="pad"),
            e.g. 32 for backbones with total stride 32.
    """

    def __init__(self, mode: str = "pad", pad_value: float = 0.0, size_divisible: int = 1) -> None:
        if mode not in ("stack", "pad", "list"):
            raise ValueError(f"mode must be 'stack', 'pad' or 'list', got {mode!r}")
        if size_divisible < 1:
            raise ValueError("size_divisible must be >= 1")
        self.mode = mode
        self.pad_value = pad_value
        self.size_divisible = size_divisible

    def __call__(self, batch: Sequence[Sample]) -> Tuple[Union[Tensor, List[Tensor]], List[Dict[str, Any]]]:
        if len(batch) == 0:
            raise ValueError("cannot collate an empty batch")
        images: List[Tensor] = []
        targets: List[Dict[str, Any]] = []
        for index, sample in enumerate(batch):
            if not isinstance(sample, (tuple, list)) or len(sample) != 2:
                raise TypeError(f"sample {index} must be an (image, target) pair")
            image, target = sample
            if not isinstance(image, Tensor) or image.ndim != 3:
                raise TypeError(f"sample {index}: image must be a Tensor (C, H, W)")
            if not isinstance(target, dict) or "boxes" not in target or "labels" not in target:
                raise TypeError(f"sample {index}: target must be a dict with 'boxes' and 'labels'")
            images.append(image)
            copied = dict(target)
            copied["image_size"] = torch.tensor([image.shape[-2], image.shape[-1]], dtype=torch.long)
            targets.append(copied)

        channels = {img.shape[0] for img in images}
        dtypes = {img.dtype for img in images}
        if self.mode != "list" and (len(channels) != 1 or len(dtypes) != 1):
            raise ValueError(f"images must share channels and dtype to be batched, got C={channels}, dtypes={dtypes}")

        if self.mode == "list":
            return images, targets
        if self.mode == "stack":
            sizes = {tuple(img.shape[-2:]) for img in images}
            if len(sizes) != 1:
                raise ValueError(f"mode='stack' needs equal image sizes, got {sorted(sizes)}; use mode='pad'")
            return torch.stack(images, dim=0), targets

        max_h = max(img.shape[-2] for img in images)
        max_w = max(img.shape[-1] for img in images)
        div = self.size_divisible
        max_h = int(math.ceil(max_h / div) * div)
        max_w = int(math.ceil(max_w / div) * div)
        padded = images[0].new_full((len(images), images[0].shape[0], max_h, max_w), self.pad_value)
        for i, img in enumerate(images):
            padded[i, :, : img.shape[-2], : img.shape[-1]].copy_(img)
        return padded, targets


def detection_collate_list(batch: Sequence[Sample]) -> Tuple[List[Tensor], List[Dict[str, Any]]]:
    """Minimal collate_fn: returns (list of images, list of targets) without copying or padding."""
    images = [sample[0] for sample in batch]
    targets = [sample[1] for sample in batch]
    return images, targets
