# Purpose: Accumulates named metric sums and counts across batches and reports properly weighted means for PyTorch training and evaluation loops.
"""Weighted running-mean metric logger.

``update(name, value, n)`` adds ``value * n`` to the sum and ``n`` to the count, where ``value``
is the batch MEAN and ``n`` the number of samples/tokens it covers. ``update_sum(name, total, n)``
is for callers that already have a sum. Means never assume equal batch sizes.

``state()`` returns plain dicts; to aggregate across ranks, all-reduce the sums and counts.

Requires: torch is optional (tensor values are converted with ``.item()``).
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Dict


def _to_float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


class MetricLogger:
    """Named sum/count accumulators."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._sums: Dict[str, float] = defaultdict(float)
        self._counts: Dict[str, float] = defaultdict(float)

    def update(self, name: str, value: Any, n: float = 1) -> None:
        """Add a batch mean ``value`` covering ``n`` (> 0) items."""
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")
        self._sums[name] += _to_float(value) * n
        self._counts[name] += n

    def update_sum(self, name: str, total: Any, n: float) -> None:
        """Add an already summed ``total`` covering ``n`` (>= 0) items."""
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        self._sums[name] += _to_float(total)
        self._counts[name] += n

    def update_many(self, values: Mapping[str, Any], n: float = 1) -> None:
        for name, value in values.items():
            self.update(name, value, n)

    def mean(self, name: str) -> float:
        if self._counts.get(name, 0) == 0:
            raise KeyError(f"No observations for metric '{name}'")
        return self._sums[name] / self._counts[name]

    def compute(self) -> Dict[str, float]:
        """Means of all metrics with a non-zero count."""
        return {name: self._sums[name] / c for name, c in self._counts.items() if c > 0}

    def state(self) -> Dict[str, Dict[str, float]]:
        return {"sums": dict(self._sums), "counts": dict(self._counts)}
