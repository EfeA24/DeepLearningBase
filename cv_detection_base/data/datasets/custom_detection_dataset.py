# Purpose: Provides an adaptation scaffold for a project-specific object-detection dataset with required hooks and the xyxy-pixel target schema.
"""Custom detection dataset scaffold (template - fill in the hooks).

Copy this file into a project and implement the three hooks. Everything else
(validation, target assembly, joint transform application) is already wired.

Required target schema returned by ``__getitem__``:
    image:  FloatTensor (C, H, W), RGB, [0, 1] before normalization
    target: {"boxes": FloatTensor (N, 4) xyxy absolute pixels,
             "labels": LongTensor (N,) class ids,
             "image_id": LongTensor scalar, "orig_size": LongTensor (2,) = (H, W)}
    Images without objects use boxes (0, 4) and labels (0,).

Hook contracts:
    ``__len__()`` -> int
    ``load_image(index)`` -> FloatTensor (C, H, W) in [0, 1]
    ``load_annotations(index)`` -> (boxes_xyxy_pixels FloatTensor (N, 4), labels LongTensor (N,))
        Convert your storage format here (see data/annotations/* for converters).

Requires: torch only (plus whatever your storage backend needs, imported lazily in the hooks).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset


class CustomDetectionDataset(Dataset):
    """Scaffold: subclass or edit and implement ``__len__``, ``load_image``, ``load_annotations``.

    Args:
        transforms: Optional joint transform ``(image, target) -> (image, target)``.
    """

    def __init__(self, transforms: Optional[Callable[[Any, Dict[str, Any]], Tuple[Any, Dict[str, Any]]]] = None) -> None:
        self.transforms = transforms

    def __len__(self) -> int:
        raise NotImplementedError("Return the number of samples, e.g. len(self.records).")

    def load_image(self, index: int) -> Tensor:
        raise NotImplementedError(
            "Load image `index` and return FloatTensor (C, H, W) RGB in [0, 1]; import image libraries inside this method."
        )

    def load_annotations(self, index: int) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError(
            "Return (boxes FloatTensor (N, 4) xyxy absolute pixels, labels LongTensor (N,)) for sample `index`."
        )

    def __getitem__(self, index: int) -> Tuple[Tensor, Dict[str, Any]]:
        image = self.load_image(index)
        boxes, labels = self.load_annotations(index)
        if boxes.ndim != 2 or boxes.shape[-1] != 4:
            raise ValueError(f"load_annotations must return boxes (N, 4), got {tuple(boxes.shape)}")
        if labels.shape != (boxes.shape[0],):
            raise ValueError("load_annotations must return labels (N,) matching boxes")
        height, width = int(image.shape[-2]), int(image.shape[-1])
        target: Dict[str, Any] = {
            "boxes": boxes.to(torch.float32),
            "labels": labels.to(torch.long),
            "image_id": torch.tensor(index, dtype=torch.long),
            "orig_size": torch.tensor([height, width], dtype=torch.long),
        }
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target
