# Purpose: Creates a torch DataLoader from an already constructed image-classification dataset and explicit, validated loader options.
"""Generic DataLoader factory.

All options are explicit keyword arguments; nothing is inferred from the environment (e.g. the
worker count is NOT derived from CPU count). Validation:
    - ``shuffle=True`` and ``sampler`` are mutually exclusive (torch rule).
    - ``persistent_workers`` and ``prefetch_factor`` require ``num_workers > 0``.
    - ``pin_memory`` only helps when batches are later copied to CUDA.

For reproducible shuffling pass a seeded ``generator`` (see utils/seed.py ``make_generator``) and
``worker_init_fn=seed_worker`` when using workers.

Requires: torch.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from torch.utils.data import DataLoader, Dataset, Sampler


def create_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool = False,
    sampler: Optional[Sampler] = None,
    num_workers: int = 0,
    collate_fn: Optional[Callable[[list], Any]] = None,
    pin_memory: bool = False,
    drop_last: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    worker_init_fn: Optional[Callable[[int], None]] = None,
    timeout: float = 0.0,
) -> DataLoader:
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"batch_size must be an int >= 1, got {batch_size!r}")
    if shuffle and sampler is not None:
        raise ValueError("shuffle=True cannot be combined with a sampler; the sampler defines the order")
    if num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {num_workers}")
    if num_workers == 0 and (persistent_workers or prefetch_factor is not None):
        raise ValueError("persistent_workers/prefetch_factor require num_workers > 0")
    if prefetch_factor is not None and prefetch_factor < 1:
        raise ValueError(f"prefetch_factor must be >= 1, got {prefetch_factor}")

    kwargs: dict = {}
    if prefetch_factor is not None:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        generator=generator,
        worker_init_fn=worker_init_fn,
        timeout=timeout,
        **kwargs,
    )
