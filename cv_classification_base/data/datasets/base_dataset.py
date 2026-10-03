# Purpose: Defines a minimal abstract map-style dataset contract for image/label classification examples independent of storage format.
"""Abstract image-classification dataset contract.

Subclasses implement:
    ``__len__()`` -> number of examples
    ``load_example(index)`` -> ``(image, target)`` in raw form (e.g. PIL RGB image or uint8
        tensor, and an int class id / multi-hot vector). No augmentation here.

``__getitem__`` applies the injected ``transform`` to the image and ``target_transform`` to the
target and returns ``(image, target)``. Keep train augmentation and val/test preprocessing as
separate transform objects passed to separate dataset instances.

Optional ``class_names`` defines the label order (index i -> name).

Requires: torch.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Sequence, Tuple

from torch.utils.data import Dataset


class ImageClassificationDataset(Dataset, ABC):
    def __init__(
        self,
        transform: Optional[Callable[[Any], Any]] = None,
        target_transform: Optional[Callable[[Any], Any]] = None,
        class_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.transform = transform
        self.target_transform = target_transform
        self.class_names: Optional[List[str]] = list(class_names) if class_names is not None else None

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def load_example(self, index: int) -> Tuple[Any, Any]:
        """Return the raw ``(image, target)`` pair for ``index``."""
        raise NotImplementedError

    @property
    def num_classes(self) -> Optional[int]:
        return len(self.class_names) if self.class_names is not None else None

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(f"index {index} out of range for dataset of size {len(self)}")
        image, target = self.load_example(index)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target
