# Purpose: Sets caller-requested Python, NumPy and PyTorch random seeds and builds seeded generators/worker init functions for reproducible experiments.
"""Seeding helpers.

``seed_everything(seed)`` seeds Python ``random``, NumPy (if installed) and torch (CPU and all
CUDA devices). It does NOT make runs bit-for-bit deterministic: non-deterministic kernels,
thread scheduling, DataLoader workers and library versions still matter (see deterministic.py).
It also does not set ``PYTHONHASHSEED`` (must be set before interpreter start to matter).

DataLoader pattern::

    g = make_generator(seed)
    DataLoader(ds, shuffle=True, generator=g, worker_init_fn=seed_worker, num_workers=4)

Requires: torch; optional numpy.
"""
from __future__ import annotations

from typing import Dict

import torch

_MAX_SEED = 2**32 - 1


def seed_everything(seed: int, seed_numpy: bool = True, seed_cuda: bool = True) -> Dict[str, bool]:
    """Seed the requested RNGs; returns which libraries were seeded.

    Args:
        seed: Integer in [0, 2**32 - 1].
        seed_numpy: Seed NumPy's global RNG if numpy is installed.
        seed_cuda: Seed all CUDA devices (no-op without CUDA; torch.manual_seed also does this).
    """
    import random

    if not isinstance(seed, int) or not 0 <= seed <= _MAX_SEED:
        raise ValueError(f"seed must be an int in [0, {_MAX_SEED}], got {seed!r}")
    seeded = {"python": True, "torch": True, "numpy": False, "cuda": False}
    random.seed(seed)
    torch.manual_seed(seed)
    if seed_numpy:
        try:
            import numpy as np
        except ImportError:
            np = None
        if np is not None:
            np.random.seed(seed)
            seeded["numpy"] = True
    if seed_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        seeded["cuda"] = True
    return seeded


def make_generator(seed: int, device: str = "cpu") -> torch.Generator:
    """Create an explicitly seeded ``torch.Generator`` (pass to DataLoader / random_split)."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: derive Python/NumPy seeds from torch's per-worker seed."""
    import random

    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    try:
        import numpy as np
    except ImportError:
        return
    np.random.seed(worker_seed)
