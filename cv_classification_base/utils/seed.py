# Purpose: Seeds Python, optional NumPy and torch RNGs on explicit request and provides DataLoader worker/generator seeding for image-classification runs.
"""Seeding helpers (consistent with common_base/reproducibility/seed.py).

``seed_everything(seed)`` seeds ``random``, NumPy (only if installed, unless ``require_numpy``) and
torch (CPU + all CUDA devices). Seeding alone does NOT make runs bit-exact: nondeterministic
kernels, cuDNN autotuning, worker scheduling and library versions still matter (see
utils/reproducibility.py).

DataLoader pattern::

    g = make_generator(seed)
    DataLoader(ds, shuffle=True, generator=g, worker_init_fn=seed_worker, num_workers=4)

Requires: torch; numpy optional.
"""
from __future__ import annotations

from typing import Dict

import torch


def seed_everything(seed: int, require_numpy: bool = False) -> Dict[str, bool]:
    """Seed available RNGs; returns which libraries were seeded."""
    if not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError(f"seed must be an int in [0, 2**32), got {seed!r}")
    import random

    random.seed(seed)
    seeded = {"python": True, "numpy": False, "torch": True}
    try:
        import numpy as np
    except ImportError:
        if require_numpy:
            raise ImportError("numpy seeding requested but numpy is not installed: pip install numpy")
    else:
        np.random.seed(seed)
        seeded["numpy"] = True
    torch.manual_seed(seed)  # also seeds all CUDA devices
    return seeded


def make_generator(seed: int) -> torch.Generator:
    """Return a CPU ``torch.Generator`` seeded with ``seed`` (for DataLoader shuffling/splits)."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: derive Python/NumPy seeds from the worker's torch seed."""
    import random

    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    try:
        import numpy as np
    except ImportError:
        return
    np.random.seed(worker_seed)
