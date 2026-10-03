# Purpose: Configures a DistributedSampler for image-classification data without initializing distributed state (explicit or already-initialized group only).
"""DistributedSampler configuration.

``num_replicas``/``rank`` are taken from the arguments; if omitted they are read from an ALREADY
initialized default process group. This module never calls ``init_process_group``; if no group is
initialized and values are missing, it raises.

Notes:
    - Call ``sampler.set_epoch(epoch)`` at the start of every epoch, otherwise each epoch uses the
      same shuffle order.
    - ``drop_last=False`` pads shards by repeating samples so all ranks get equal counts;
      ``drop_last=True`` drops the tail instead.
    - Use ``shuffle=False`` for validation/test sharding.
    - Pass the sampler to the DataLoader with ``shuffle=False``.

Requires: torch.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch.distributed as dist
from torch.utils.data import Dataset, DistributedSampler


def _resolve_world(num_replicas: Optional[int], rank: Optional[int]) -> Tuple[int, int]:
    if num_replicas is None or rank is None:
        if not (dist.is_available() and dist.is_initialized()):
            raise ValueError(
                "num_replicas and rank must be passed explicitly when no process group is initialized"
            )
        num_replicas = dist.get_world_size() if num_replicas is None else num_replicas
        rank = dist.get_rank() if rank is None else rank
    if num_replicas < 1 or not 0 <= rank < num_replicas:
        raise ValueError(f"invalid num_replicas={num_replicas}, rank={rank}")
    return num_replicas, rank


def build_distributed_sampler(
    dataset: Dataset,
    shuffle: bool,
    seed: int = 0,
    num_replicas: Optional[int] = None,
    rank: Optional[int] = None,
    drop_last: bool = False,
) -> DistributedSampler:
    world, rank_ = _resolve_world(num_replicas, rank)
    return DistributedSampler(
        dataset, num_replicas=world, rank=rank_, shuffle=shuffle, seed=seed, drop_last=drop_last
    )
