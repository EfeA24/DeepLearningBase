# Purpose: Provides a multi-label image dataset returning images with multi-hot float label vectors in a documented class order.
"""Multi-label image dataset.

Class ordering: ``class_names[j]`` is the label of vector position ``j``. Keep this list fixed
across train/val/test and the model head (models/heads/multilabel_head.py).

Label inputs (one per image), either:
    - label index lists, e.g. ``[[0, 3], [], [2]]`` (empty list = no positive labels), or
    - a multi-hot tensor ``(N, L)`` with values in {0, 1}.
Returns ``(transform(image), target)`` with target float32 ``(L,)`` multi-hot (or
``target_transform(vector)``). Use with BCEWithLogits-style losses.

Requires: torch.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset


class MultilabelImageDataset(Dataset):
    def __init__(
        self,
        images: Union[Sequence[Any], torch.Tensor],
        labels: Union[Sequence[Sequence[int]], torch.Tensor],
        class_names: Sequence[str],
        transform: Optional[Callable[[Any], Any]] = None,
        target_transform: Optional[Callable[[torch.Tensor], Any]] = None,
        loader: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        num_labels = len(class_names)
        if num_labels < 1:
            raise ValueError("class_names must not be empty")
        if len(set(class_names)) != num_labels:
            raise ValueError("class_names must be unique")
        if isinstance(labels, torch.Tensor):
            if labels.dim() != 2 or labels.shape[1] != num_labels:
                raise ValueError(f"multi-hot labels must be (N, {num_labels}), got {tuple(labels.shape)}")
            if not bool(((labels == 0) | (labels == 1)).all()):
                raise ValueError("multi-hot labels must contain only 0/1")
            targets = labels.to(torch.float32).clone()
        else:
            targets = torch.zeros(len(labels), num_labels, dtype=torch.float32)
            for row, indices in enumerate(labels):
                for j in indices:
                    if not 0 <= int(j) < num_labels:
                        raise ValueError(f"example {row}: label index {j} outside [0, {num_labels})")
                    targets[row, int(j)] = 1.0
        if len(images) != targets.shape[0]:
            raise ValueError(f"len(images)={len(images)} != number of label rows={targets.shape[0]}")

        self.images = images
        self.targets = targets
        self.class_names = list(class_names)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    @property
    def num_labels(self) -> int:
        return len(self.class_names)

    def label_frequencies(self) -> torch.Tensor:
        """Positive count per label ``(L,)`` (compute on the TRAIN split only for pos_weight)."""
        return self.targets.sum(dim=0)

    def __len__(self) -> int:
        return self.targets.shape[0]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = self.images[index]
        if self.loader is not None:
            image = self.loader(image)
        if self.transform is not None:
            image = self.transform(image)
        target: Any = self.targets[index].clone()
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target
