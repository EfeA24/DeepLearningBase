# Purpose: Builds a deterministic validation DataLoader (no shuffling, no dropped samples) for image classification.
"""Validation DataLoader.

Fixed policy: ``shuffle=False`` and ``drop_last=False`` so every validation example is evaluated
exactly once in a stable order, making metrics comparable across epochs. An optional sampler is
accepted only for sharded evaluation (e.g. ``DistributedSampler(..., shuffle=False)``); note that
DistributedSampler pads to equal shard sizes, so a few samples may be counted twice unless the
caller de-duplicates.

Requires: torch.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from torch.utils.data import DataLoader, Dataset, Sampler


def build_validation_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    collate_fn: Optional[Callable[[list], Any]] = None,
    sampler: Optional[Sampler] = None,
) -> DataLoader:
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {num_workers}")
    if sampler is not None and getattr(sampler, "shuffle", False):
        raise ValueError("validation sampler must not shuffle")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_fn,
    )
