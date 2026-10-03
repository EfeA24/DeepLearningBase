# Purpose: Explicitly initializes and tears down a torch.distributed process group from caller-supplied backend, init method, rank and world size.
"""Explicit process-group lifecycle.

Nothing runs on import. The caller supplies every value; this module never reads environment
variables. If you launch with ``torchrun``, read ``RANK``/``WORLD_SIZE``/``LOCAL_RANK`` yourself
(in your entry script) and pass them in, or pass ``init_method="env://"`` knowingly (then torch
reads MASTER_ADDR/MASTER_PORT from the environment itself).

Backends:
    - "nccl": multi-GPU CUDA (Linux). Set the CUDA device BEFORE init (``device_id`` argument).
    - "gloo": CPU or fallback; works on Windows.

Usage::

    init_process_group("gloo", init_method="tcp://127.0.0.1:29500", rank=rank, world_size=ws)
    try:
        ...training...
    finally:
        destroy_process_group()

Requires: torch.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist


def init_process_group(
    backend: str,
    init_method: str,
    rank: int,
    world_size: int,
    local_device_index: Optional[int] = None,
    timeout_seconds: Optional[float] = None,
) -> None:
    """Initialize the default process group.

    Args:
        backend: "nccl", "gloo", or another backend supported by your build.
        init_method: e.g. "tcp://host:port", "file:///path/to/shared_file" or "env://".
        rank: This process's global rank in [0, world_size).
        world_size: Total number of processes (>= 1).
        local_device_index: If given and backend is "nccl", calls ``torch.cuda.set_device``
            before init so each rank uses its own GPU.
        timeout_seconds: Optional collective timeout.

    Raises:
        RuntimeError: torch.distributed unavailable or already initialized.
        ValueError: invalid rank/world_size.
    """
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this torch build")
    if dist.is_initialized():
        raise RuntimeError("A default process group is already initialized")
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    if backend == "nccl" and local_device_index is not None:
        torch.cuda.set_device(local_device_index)
    kwargs = {}
    if timeout_seconds is not None:
        from datetime import timedelta

        kwargs["timeout"] = timedelta(seconds=timeout_seconds)
    dist.init_process_group(backend=backend, init_method=init_method, rank=rank, world_size=world_size, **kwargs)


def destroy_process_group() -> None:
    """Destroy the default process group if one exists (safe to call in ``finally``)."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
