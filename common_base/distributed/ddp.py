# Purpose: Shows minimal DistributedDataParallel wrapping and per-rank data sharding for multi-process PyTorch training where the caller owns the process lifecycle.
"""Minimal DDP wrapping example.

Caller responsibilities:
    - Initialize the process group BEFORE calling ``wrap_ddp`` (see process_group.py) and
      destroy it afterwards.
    - Move the model to this rank's device before wrapping.
    - Use a ``DistributedSampler`` and call ``sampler.set_epoch(epoch)`` each epoch.
    - Save checkpoints from rank 0 using ``model.module.state_dict()``.

Example per-rank flow::

    init_process_group("nccl", "env://", rank, world_size, local_device_index=local_rank)
    device = torch.device("cuda", local_rank)
    model = wrap_ddp(model.to(device), device_ids=[local_rank])
    sampler = make_distributed_sampler(dataset, shuffle=True, seed=0)
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)
    ...
    destroy_process_group()

Requires: torch.
"""
from __future__ import annotations

from typing import Any, List, Optional

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DistributedSampler


def wrap_ddp(
    model: nn.Module,
    device_ids: Optional[List[int]] = None,
    find_unused_parameters: bool = False,
    broadcast_buffers: bool = True,
    gradient_as_bucket_view: bool = True,
    **kwargs: Any,
) -> DistributedDataParallel:
    """Wrap an already placed model in DDP.

    Args:
        model: Model already on its target device.
        device_ids: ``[local_rank]`` for single-GPU-per-process CUDA; None for CPU models.
        find_unused_parameters: Set True only if some parameters do not receive gradients every
            step (costs an extra graph traversal).
        broadcast_buffers: Sync buffers (e.g. BatchNorm stats) from rank 0 each forward.
        gradient_as_bucket_view: Memory optimization; grads become views into comm buckets.
    """
    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError("Initialize a process group before wrapping with DDP")
    return DistributedDataParallel(
        model,
        device_ids=device_ids,
        find_unused_parameters=find_unused_parameters,
        broadcast_buffers=broadcast_buffers,
        gradient_as_bucket_view=gradient_as_bucket_view,
        **kwargs,
    )


def make_distributed_sampler(
    dataset: Dataset,
    shuffle: bool,
    seed: int,
    drop_last: bool = False,
    num_replicas: Optional[int] = None,
    rank: Optional[int] = None,
) -> DistributedSampler:
    """Create a DistributedSampler; replicas/rank default to the initialized group's values.

    Without ``drop_last`` the sampler pads by repeating samples so all ranks get equal counts;
    this slightly biases validation metrics.
    """
    if (num_replicas is None or rank is None) and not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError("Pass num_replicas and rank explicitly when no process group is initialized")
    return DistributedSampler(
        dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last
    )


def unwrap_ddp(model: nn.Module) -> nn.Module:
    """Return the underlying module of a DDP-wrapped model (identity otherwise)."""
    return model.module if isinstance(model, DistributedDataParallel) else model


def ddp_no_sync(model: nn.Module, skip_sync: bool):
    """Context that skips gradient all-reduce (for accumulation micro-batches) when requested."""
    from contextlib import nullcontext

    if skip_sync and isinstance(model, DistributedDataParallel):
        return model.no_sync()
    return nullcontext()
