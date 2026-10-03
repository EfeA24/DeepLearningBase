# Purpose: Builds the training DataLoader for image classification from an explicit config, validating shuffle versus caller-supplied sampler.
"""Training DataLoader.

Policy:
    - Without a sampler, ``shuffle`` defaults to True.
    - With a sampler (weighted/balanced/distributed), ``shuffle`` must be False: the sampler owns
      ordering. A ``DistributedSampler`` needs ``sampler.set_epoch(epoch)`` each epoch (caller).
    - ``drop_last`` defaults to True so BatchNorm never sees a size-1 final batch; set False if
      every sample must be seen each epoch.

Requires: torch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
from torch.utils.data import DataLoader, Dataset, Sampler


@dataclass(frozen=True)
class TrainLoaderConfig:
    batch_size: int
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = True
    persistent_workers: bool = False

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")
        if self.persistent_workers and self.num_workers == 0:
            raise ValueError("persistent_workers requires num_workers > 0")


def build_train_loader(
    dataset: Dataset,
    config: TrainLoaderConfig,
    sampler: Optional[Sampler] = None,
    collate_fn: Optional[Callable[[list], Any]] = None,
    generator: Optional[torch.Generator] = None,
    worker_init_fn: Optional[Callable[[int], None]] = None,
) -> DataLoader:
    if sampler is not None and config.shuffle:
        raise ValueError(
            "A sampler was supplied but config.shuffle=True; set shuffle=False (the sampler "
            "controls ordering and randomness)."
        )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
        persistent_workers=config.persistent_workers,
        collate_fn=collate_fn,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )
