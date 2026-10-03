# Purpose: Computes class-balanced sampling weights from supplied single-label targets and builds a balanced sampler, handling absent classes explicitly.
"""Class-balanced sampling.

Per-sample weight: ``w_i = 1 / count[y_i] ** power`` (``power=1`` -> every present class has equal
expected frequency; ``power=0.5`` -> softened "square-root" balancing; ``power=0`` -> uniform).

Absent classes (count 0 within ``num_classes``) cannot be sampled. ``on_absent`` controls the
response: "error" (default), "warn", or "ignore". Compute weights from the TRAIN split labels only.

Requires: torch.
"""
from __future__ import annotations

import warnings
from typing import Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import WeightedRandomSampler


def class_counts(labels: Union[Sequence[int], torch.Tensor], num_classes: int) -> torch.Tensor:
    y = torch.as_tensor(labels, dtype=torch.long)
    if y.dim() != 1 or y.numel() == 0:
        raise ValueError("labels must be a non-empty 1D sequence of class ids")
    if int(y.min()) < 0 or int(y.max()) >= num_classes:
        raise ValueError(f"labels must lie in [0, {num_classes})")
    return torch.bincount(y, minlength=num_classes)


def class_balanced_weights(
    labels: Union[Sequence[int], torch.Tensor],
    num_classes: int,
    power: float = 1.0,
    on_absent: str = "error",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(sample_weights (N,), class_weights (num_classes,))``; absent classes get 0."""
    if on_absent not in ("error", "warn", "ignore"):
        raise ValueError("on_absent must be 'error', 'warn' or 'ignore'")
    if power < 0:
        raise ValueError(f"power must be >= 0, got {power}")
    y = torch.as_tensor(labels, dtype=torch.long)
    counts = class_counts(y, num_classes)
    absent = (counts == 0).nonzero().flatten().tolist()
    if absent:
        message = f"classes with no samples: {absent}; they will never be drawn"
        if on_absent == "error":
            raise ValueError(message)
        if on_absent == "warn":
            warnings.warn(message, stacklevel=2)
    class_weights = torch.zeros(num_classes, dtype=torch.double)
    present = counts > 0
    class_weights[present] = counts[present].double().pow(-power)
    return class_weights[y], class_weights


def build_balanced_sampler(
    labels: Union[Sequence[int], torch.Tensor],
    num_classes: int,
    num_samples: Optional[int] = None,
    power: float = 1.0,
    on_absent: str = "error",
    generator: Optional[torch.Generator] = None,
) -> WeightedRandomSampler:
    """Sampler drawing ``num_samples`` (default ``len(labels)``) indices WITH replacement."""
    sample_weights, _ = class_balanced_weights(labels, num_classes, power, on_absent)
    n = len(sample_weights) if num_samples is None else num_samples
    if n < 1:
        raise ValueError(f"num_samples must be >= 1, got {n}")
    return WeightedRandomSampler(sample_weights, num_samples=n, replacement=True, generator=generator)
