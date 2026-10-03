# Purpose: Provides an adaptation template for a project-specific image-classification dataset with abstract index/load/label hooks (scaffold).
"""Custom dataset adaptation template (scaffold).

Copy this file and implement the three hooks; everything else (bounds checks, transform
application, output contract) is already wired:
    ``_build_index()`` -> list of opaque records (paths, DB keys, offsets...). Called once in
        ``__init__`` with the caller's explicit ``source`` (never at import).
    ``_load_image(record)`` -> raw image (PIL RGB / uint8 tensor (C, H, W)).
    ``_load_target(record)`` -> int class id (or multi-hot vector for multi-label).

Output contract: ``(transform(image), target_transform(target))``.

Requires: torch; plus whatever your hooks need (import those lazily inside the hooks).
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(
        self,
        source: Any,
        transform: Optional[Callable[[Any], Any]] = None,
        target_transform: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        self.source = source
        self.transform = transform
        self.target_transform = target_transform
        self.records: List[Any] = list(self._build_index())

    def _build_index(self) -> List[Any]:
        raise NotImplementedError(
            "Implement _build_index(): enumerate examples from self.source and return a list of "
            "records (e.g. [(path, label), ...])."
        )

    def _load_image(self, record: Any) -> Any:
        raise NotImplementedError("Implement _load_image(record) -> PIL RGB image or (C, H, W) tensor.")

    def _load_target(self, record: Any) -> Any:
        raise NotImplementedError("Implement _load_target(record) -> int class id (or multi-hot vector).")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if not -len(self.records) <= index < len(self.records):
            raise IndexError(f"index {index} out of range for dataset of size {len(self.records)}")
        record = self.records[index]
        image = self._load_image(record)
        target = self._load_target(record)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target
