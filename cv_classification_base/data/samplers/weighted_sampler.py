# Purpose: Builds a WeightedRandomSampler from explicit per-sample weights with validated replacement and epoch-length semantics for image classification.
"""Weighted random sampling.

Semantics:
    - ``weights[i]`` is the relative probability of drawing sample ``i`` (need not sum to 1).
    - ``num_samples`` = draws per epoch ("epoch length"); defaults to ``len(weights)``.
    - ``replacement=True``: samples can repeat within an epoch (needed for oversampling).
      ``replacement=False``: each index at most once, so ``num_samples`` must not exceed the
      number of positive weights.
Use with ``shuffle=False`` in the DataLoader. Pass a seeded ``generator`` for reproducibility.

Requires: torch.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
from torch.utils.data import WeightedRandomSampler


def build_weighted_sampler(
    weights: Union[Sequence[float], torch.Tensor],
    num_samples: Optional[int] = None,
    replacement: bool = True,
    generator: Optional[torch.Generator] = None,
) -> WeightedRandomSampler:
    w = torch.as_tensor(weights, dtype=torch.double)
    if w.dim() != 1 or w.numel() == 0:
        raise ValueError("weights must be a non-empty 1D sequence")
    if not bool(torch.isfinite(w).all()) or bool((w < 0).any()):
        raise ValueError("weights must be finite and non-negative")
    positive = int((w > 0).sum())
    if positive == 0:
        raise ValueError("at least one weight must be positive")
    n = w.numel() if num_samples is None else num_samples
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"num_samples must be an int >= 1, got {num_samples!r}")
    if not replacement and n > positive:
        raise ValueError(
            f"replacement=False requires num_samples <= number of positive weights ({positive}), got {n}"
        )
    return WeightedRandomSampler(w, num_samples=n, replacement=replacement, generator=generator)
