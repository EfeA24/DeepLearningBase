# Purpose: Provides rank/world-size queries and all-reduce helpers that degrade gracefully when torch.distributed is unavailable or uninitialized.
"""Distributed helpers safe in single-process runs.

Every function checks ``torch.distributed.is_available() and is_initialized()`` at CALL time and
falls back to single-process semantics (rank 0, world size 1, identity reductions) otherwise.
Nothing is initialized here; see process_group.py for explicit init/teardown.

Tensor contracts:
    - ``all_reduce_sum`` / ``all_reduce_mean`` return a NEW tensor (inputs are not mutated).
    - With the NCCL backend tensors must be on the current CUDA device; with Gloo, CPU tensors.
    - ``all_reduce_mean`` divides by world size: correct only when each rank contributes equally.
      For sample-weighted means reduce (sum, count) pairs with ``weighted_mean_across_ranks``.

Requires: torch.
"""
from __future__ import annotations

from typing import Any, List

import torch
import torch.distributed as dist


def is_dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_ready() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_ready() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    """Synchronize all ranks; no-op when not distributed."""
    if is_dist_ready():
        dist.barrier()


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """Sum ``tensor`` across ranks and return the result as a new tensor."""
    result = tensor.detach().clone()
    if is_dist_ready() and get_world_size() > 1:
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Average ``tensor`` across ranks (equal-weight mean). Integer tensors are cast to float."""
    result = all_reduce_sum(tensor if tensor.is_floating_point() else tensor.float())
    return result / get_world_size()


def weighted_mean_across_ranks(value_sum: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
    """Global ``sum(value_sum) / sum(count)``; raises if the global count is zero."""
    stacked = torch.stack([value_sum.detach().double().reshape(()), count.detach().double().reshape(())])
    total = all_reduce_sum(stacked)
    if total[1].item() == 0:
        raise ValueError("Global count is zero; cannot compute weighted mean")
    return total[0] / total[1]


def all_gather_objects(obj: Any) -> List[Any]:
    """Gather a picklable object from every rank (list ordered by rank). Uses pickling; trusted data only."""
    if not is_dist_ready() or get_world_size() == 1:
        return [obj]
    gathered: List[Any] = [None] * get_world_size()
    dist.all_gather_object(gathered, obj)
    return gathered
