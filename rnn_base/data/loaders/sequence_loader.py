# Purpose: Creates a DataLoader for a supplied sequence dataset with explicit sampler, collator and worker settings.
"""Sequence DataLoader construction.

- The dataset is already constructed by the caller; nothing is read here.
- ``shuffle`` and ``sampler`` are mutually exclusive (ValueError if both are set).
- Stream-ordered training that carries hidden state across batches (TBPTT over a batchified
  stream) needs ``shuffle=False`` so batch row b continues the same stream every step.
- Pass a seeded ``torch.Generator`` for reproducible shuffling.
- ``collate_fn`` is injected, e.g. ``PaddedCollator`` for variable lengths; None uses PyTorch's default.

Requires: torch only. No sibling files.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from torch.utils.data import DataLoader, Dataset, Sampler


def build_sequence_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    sampler: Optional[Sampler] = None,
    collate_fn: Optional[Callable[[Any], Any]] = None,
    num_workers: int = 0,
    drop_last: bool = False,
    pin_memory: bool = False,
    generator: Optional[torch.Generator] = None,
    persistent_workers: bool = False,
) -> DataLoader:
    """Create a DataLoader with validated, explicit options."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if shuffle and sampler is not None:
        raise ValueError("shuffle=True cannot be combined with a sampler; the sampler defines the order.")
    if num_workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {num_workers}.")
    if persistent_workers and num_workers == 0:
        raise ValueError("persistent_workers=True requires num_workers > 0.")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        generator=generator,
        persistent_workers=persistent_workers,
    )
