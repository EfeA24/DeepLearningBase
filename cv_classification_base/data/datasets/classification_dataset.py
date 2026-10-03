# Purpose: Provides a single-label image dataset over explicit in-memory records (tensors, images or paths) with validated class ids.
"""Single-label dataset from explicit records.

Inputs:
    images: a sequence of examples (tensors ``(C, H, W)``, PIL images, file paths, ...) or a
        stacked tensor ``(N, C, H, W)``.
    labels: int class ids ``(N,)`` (sequence or tensor), each in ``[0, num_classes)``.
    loader: optional ``loader(item) -> image`` applied first (e.g. a PIL path loader). Without it,
        items are returned as stored.
Returns ``(transform(image), target)`` where target is a Python int (or
``target_transform(int)``). The default collate turns ints into an int64 ``(B,)`` tensor.

Requires: torch.
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset


class ClassificationDataset(Dataset):
    def __init__(
        self,
        images: Union[Sequence[Any], torch.Tensor],
        labels: Union[Sequence[int], torch.Tensor],
        num_classes: int,
        transform: Optional[Callable[[Any], Any]] = None,
        target_transform: Optional[Callable[[int], Any]] = None,
        loader: Optional[Callable[[Any], Any]] = None,
        class_names: Optional[Sequence[str]] = None,
    ) -> None:
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        label_list: List[int]
        if isinstance(labels, torch.Tensor):
            if labels.dim() != 1 or labels.is_floating_point():
                raise ValueError("labels tensor must be 1D integer class ids")
            label_list = [int(v) for v in labels.tolist()]
        else:
            label_list = []
            for v in labels:
                if isinstance(v, bool) or int(v) != v:
                    raise TypeError(f"labels must be integer class ids, got {v!r}")
                label_list.append(int(v))
        if len(images) != len(label_list):
            raise ValueError(f"len(images)={len(images)} != len(labels)={len(label_list)}")
        bad = [v for v in label_list if not 0 <= v < num_classes]
        if bad:
            raise ValueError(f"labels outside [0, {num_classes}): {sorted(set(bad))[:10]}")
        if class_names is not None and len(class_names) != num_classes:
            raise ValueError("class_names length must equal num_classes")

        self.images = images
        self.targets = label_list
        self.num_classes = num_classes
        self.class_names = list(class_names) if class_names is not None else None
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = self.images[index]
        if self.loader is not None:
            image = self.loader(image)
        if self.transform is not None:
            image = self.transform(image)
        label = self.targets[index]
        target: Any = label if self.target_transform is None else self.target_transform(label)
        return image, target
