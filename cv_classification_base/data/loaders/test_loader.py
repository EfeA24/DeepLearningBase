# Purpose: Builds a deterministic, order-preserving test DataLoader for final image-classification evaluation, separate from validation policy.
"""Test DataLoader.

Kept separate from validation_loader.py so test-time policy can diverge (e.g. larger batches,
returning file paths for per-sample reports) without touching model-selection code.

Fixed policy: no shuffling, no sampler, ``drop_last=False``: output order equals dataset index
order, so prediction row ``i`` corresponds to ``dataset[i]``. Evaluate the test split only after
model selection is finished.

Requires: torch.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from torch.utils.data import DataLoader, Dataset


def build_test_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    collate_fn: Optional[Callable[[list], Any]] = None,
) -> DataLoader:
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {num_workers}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_fn,
    )
