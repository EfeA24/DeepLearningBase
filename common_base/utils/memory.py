# Purpose: Provides opt-in CUDA memory reporting and cleanup helpers for diagnosing memory use in PyTorch training and inference.
"""Memory reporting/cleanup (opt-in; nothing runs on import).

All CUDA functions take an explicit device and return plain dicts of bytes (or MiB via
``to_mib``). On machines without CUDA, ``cuda_memory_stats`` raises; ``free_memory`` just runs
garbage collection. Note ``torch.cuda.empty_cache`` returns cached blocks to the driver but
does not free tensors that are still referenced.

Requires: torch.
"""
from __future__ import annotations

from typing import Dict, Optional, Union

import torch

DeviceLike = Union[str, int, torch.device]


def _cuda_device(device: DeviceLike) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    dev = torch.device(f"cuda:{device}") if isinstance(device, int) else torch.device(device)
    if dev.type != "cuda":
        raise ValueError(f"Expected a CUDA device, got {dev}")
    return dev


def cuda_memory_stats(device: DeviceLike) -> Dict[str, int]:
    """Allocated/reserved (current and peak) bytes for ``device``."""
    dev = _cuda_device(device)
    return {
        "allocated": torch.cuda.memory_allocated(dev),
        "reserved": torch.cuda.memory_reserved(dev),
        "max_allocated": torch.cuda.max_memory_allocated(dev),
        "max_reserved": torch.cuda.max_memory_reserved(dev),
    }


def reset_peak_memory(device: DeviceLike) -> None:
    torch.cuda.reset_peak_memory_stats(_cuda_device(device))


def to_mib(stats: Dict[str, int]) -> Dict[str, float]:
    return {k: v / (1024**2) for k, v in stats.items()}


def free_memory(device: Optional[DeviceLike] = None) -> int:
    """Run ``gc.collect()`` and, for a CUDA ``device``, ``torch.cuda.empty_cache()``.

    Returns the number of unreachable objects found by the garbage collector.
    """
    import gc

    collected = gc.collect()
    if device is None:
        return collected
    dev = torch.device(f"cuda:{device}") if isinstance(device, int) else torch.device(device)
    if dev.type == "cuda":
        with torch.cuda.device(_cuda_device(dev)):
            torch.cuda.empty_cache()
    return collected


def tensor_nbytes(tensor: torch.Tensor) -> int:
    """Bytes used by the tensor's elements (not the full underlying storage)."""
    return tensor.numel() * tensor.element_size()


def module_memory_bytes(module: torch.nn.Module, include_buffers: bool = True) -> int:
    """Bytes of parameters (and buffers) held by ``module``; shared tensors counted once."""
    seen = set()
    total = 0
    tensors = list(module.parameters()) + (list(module.buffers()) if include_buffers else [])
    for t in tensors:
        if id(t) in seen:
            continue
        seen.add(id(t))
        total += tensor_nbytes(t)
    return total
