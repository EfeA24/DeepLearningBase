# Purpose: Builds an object-detection DataLoader from an already constructed dataset with explicit batch, sampler, worker and collate settings.
"""Detection DataLoader construction.

Default collate: ``list_collate`` returns ``(list of images, list of targets)``
because detection targets have a variable number of boxes per image and
images may differ in size. Inject ``collate_fn`` (e.g.
data/collators/detection_collator.py ``DetectionCollator(mode="pad")``) to get
a padded (B, C, H, W) tensor instead.

Policy notes:
    - ``shuffle`` and ``sampler`` are mutually exclusive (DataLoader rule made explicit).
    - Pass a seeded ``generator`` for reproducible shuffling; seed augmentation
      RNGs per worker with ``worker_init_fn``.
    - Nothing is created at import time; the dataset is never inspected beyond len().

Requires: torch only.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Sampler


def list_collate(batch: Sequence[Tuple[Any, Dict[str, Any]]]) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """Collate (image, target) pairs into (list of images, list of targets)."""
    images = [sample[0] for sample in batch]
    targets = [sample[1] for sample in batch]
    return images, targets


def build_detection_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    sampler: Optional[Sampler] = None,
    num_workers: int = 0,
    collate_fn: Optional[Callable[[Sequence[Any]], Any]] = None,
    pin_memory: bool = False,
    drop_last: bool = False,
    generator: Optional[torch.Generator] = None,
    worker_init_fn: Optional[Callable[[int], None]] = None,
    persistent_workers: bool = False,
    prefetch_factor: Optional[int] = None,
) -> DataLoader:
    """Create a DataLoader for detection.

    Args:
        dataset: Map-style dataset returning (image, target).
        batch_size: Images per batch (> 0).
        shuffle: Shuffle each epoch (training); must be False when ``sampler`` is given.
        sampler: Optional sampler (e.g. DistributedSampler configured by the caller).
        num_workers: Worker processes (0 = main process).
        collate_fn: Batch assembly function; defaults to ``list_collate``.
        pin_memory: Pin host memory (useful with CUDA).
        drop_last: Drop the final incomplete batch.
        generator: RNG for shuffling / random samplers.
        worker_init_fn: Per-worker init hook (reseed augmentation generators here).
        persistent_workers: Keep workers alive across epochs (requires num_workers > 0).
        prefetch_factor: Batches prefetched per worker (requires num_workers > 0).
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {num_workers}")
    if shuffle and sampler is not None:
        raise ValueError("shuffle=True and a sampler are mutually exclusive; shuffle inside the sampler instead")
    if num_workers == 0 and (persistent_workers or prefetch_factor is not None):
        raise ValueError("persistent_workers/prefetch_factor require num_workers > 0")

    kwargs: Dict[str, Any] = {}
    if prefetch_factor is not None:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn or list_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        generator=generator,
        worker_init_fn=worker_init_fn,
        persistent_workers=persistent_workers,
        **kwargs,
    )
