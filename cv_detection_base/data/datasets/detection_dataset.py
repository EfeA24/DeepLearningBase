# Purpose: Defines the generic map-style object-detection dataset contract (abstract base) returning an image tensor and a target dict.
"""Detection dataset contract.

Each item is ``(image, target)``:
    image:  FloatTensor (C, H, W), RGB, values in [0, 1] (before normalization)
    target: {"boxes": FloatTensor (N, 4) xyxy absolute pixels,
             "labels": LongTensor (N,),
             optional "image_id": LongTensor scalar,
             optional "orig_size": LongTensor (2,) = (height, width) before transforms,
             optional per-box extras such as "iscrowd"/"difficult"/"area" (N,)}
    N may be 0 (boxes shape (0, 4), labels shape (0,)).

Subclasses implement ``__len__``, ``load_image`` and ``load_target``.
``__getitem__`` validates the raw target and then applies the injected joint
transform ``transforms(image, target) -> (image, target)``, which must update
boxes consistently with any geometric image change.

Requires: torch only.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

Target = Dict[str, Any]
JointTransform = Callable[[Any, Target], Tuple[Any, Target]]


def validate_target(target: Target, image_size: Optional[Tuple[int, int]] = None) -> None:
    """Raise if ``target`` violates the warehouse detection target schema.

    Args:
        target: Target dict to check.
        image_size: Optional (height, width); when given, boxes must lie inside
            the image (with a small tolerance for float rounding).
    """
    if not isinstance(target, dict):
        raise TypeError(f"target must be a dict, got {type(target).__name__}")
    for key in ("boxes", "labels"):
        if key not in target:
            raise KeyError(f"target is missing required key {key!r}")
    boxes, labels = target["boxes"], target["labels"]
    if not isinstance(boxes, Tensor) or boxes.ndim != 2 or boxes.shape[-1] != 4 or not boxes.is_floating_point():
        raise ValueError(f"target['boxes'] must be a float tensor (N, 4), got {getattr(boxes, 'shape', type(boxes))}")
    if not isinstance(labels, Tensor) or labels.ndim != 1 or labels.dtype != torch.long:
        raise ValueError("target['labels'] must be an int64 tensor of shape (N,)")
    if boxes.shape[0] != labels.shape[0]:
        raise ValueError(f"boxes ({boxes.shape[0]}) and labels ({labels.shape[0]}) count mismatch")
    if boxes.numel() and ((boxes[:, 2] < boxes[:, 0]).any() or (boxes[:, 3] < boxes[:, 1]).any()):
        raise ValueError("target['boxes'] must be xyxy with x2 >= x1 and y2 >= y1")
    if image_size is not None and boxes.numel():
        height, width = image_size
        tol = 1e-3
        if (boxes[:, 0::2] < -tol).any() or (boxes[:, 1::2] < -tol).any() or (boxes[:, 0::2] > width + tol).any() or (
            boxes[:, 1::2] > height + tol
        ).any():
            raise ValueError(f"target['boxes'] extend outside the image of size (H={height}, W={width})")


def empty_target() -> Target:
    """Target with zero objects in the warehouse schema."""
    return {"boxes": torch.zeros(0, 4, dtype=torch.float32), "labels": torch.zeros(0, dtype=torch.long)}


class DetectionDataset(Dataset, ABC):
    """Abstract map-style detection dataset.

    Args:
        transforms: Optional joint transform ``(image, target) -> (image, target)``.
        validate: Validate each raw target before transforms (cheap; recommended).
    """

    def __init__(self, transforms: Optional[JointTransform] = None, validate: bool = True) -> None:
        self.transforms = transforms
        self.validate = validate

    @abstractmethod
    def __len__(self) -> int:
        """Number of images."""

    @abstractmethod
    def load_image(self, index: int) -> Any:
        """Return the raw image for ``index`` (FloatTensor (C, H, W) in [0, 1] recommended)."""

    @abstractmethod
    def load_target(self, index: int) -> Target:
        """Return the raw target dict for ``index`` in the warehouse schema (boxes in pixels)."""

    def __getitem__(self, index: int) -> Tuple[Any, Target]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(f"index {index} out of range for dataset of size {len(self)}")
        image = self.load_image(index)
        target = self.load_target(index)
        if self.validate:
            size = (int(image.shape[-2]), int(image.shape[-1])) if isinstance(image, Tensor) else None
            validate_target(target, size)
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target
