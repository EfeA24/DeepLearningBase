# Purpose: Holds resumable scalar training progress (epoch, global step, best metric, metadata) for any PyTorch training loop.
"""Resumable scalar training state.

``TrainState`` stores only plain Python scalars and a JSON-friendly metadata dict so it can
be saved inside a checkpoint (``torch.save`` with ``weights_only=True`` compatible) or as JSON.

Conventions:
    - ``epoch`` counts COMPLETED epochs (0 before training starts).
    - ``global_step`` counts optimizer updates (not micro-batches).
    - ``best_metric`` is ``None`` until the first evaluation; ``mode`` is "min" or "max".

Requires: torch is not needed by this module itself (stdlib only).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class TrainState:
    """Plain resumable training counters and best-metric tracking."""

    epoch: int = 0
    global_step: int = 0
    best_metric: Optional[float] = None
    best_epoch: Optional[int] = None
    mode: str = "min"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {self.mode!r}")
        if self.epoch < 0 or self.global_step < 0:
            raise ValueError("epoch and global_step must be non-negative")

    def is_improvement(self, value: float, min_delta: float = 0.0) -> bool:
        """Return True if ``value`` beats ``best_metric`` by more than ``min_delta``."""
        if min_delta < 0:
            raise ValueError(f"min_delta must be >= 0, got {min_delta}")
        if math.isnan(value):
            return False
        if self.best_metric is None:
            return True
        if self.mode == "min":
            return value < self.best_metric - min_delta
        return value > self.best_metric + min_delta

    def update_best(self, value: float, min_delta: float = 0.0) -> bool:
        """Record ``value`` as best (at the current epoch) if it improves; return whether it did."""
        improved = self.is_improvement(value, min_delta)
        if improved:
            self.best_metric = float(value)
            self.best_epoch = self.epoch
        return improved

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "mode": self.mode,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainState":
        """Rebuild from ``to_dict`` output; unknown keys raise to surface schema drift."""
        known = {"epoch", "global_step", "best_metric", "best_epoch", "mode", "metadata"}
        unknown = set(data) - known
        if unknown:
            raise KeyError(f"Unknown TrainState keys: {sorted(unknown)}")
        return cls(
            epoch=int(data.get("epoch", 0)),
            global_step=int(data.get("global_step", 0)),
            best_metric=None if data.get("best_metric") is None else float(data["best_metric"]),
            best_epoch=None if data.get("best_epoch") is None else int(data["best_epoch"]),
            mode=str(data.get("mode", "min")),
            metadata=dict(data.get("metadata", {})),
        )
