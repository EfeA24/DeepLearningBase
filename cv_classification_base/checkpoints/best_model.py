# Purpose: Tracks the best validation metric for image-classification training and invokes a caller-supplied persistence callback on improvement.
"""Best-model selection helper.

The caller chooses the metric direction (``mode``), the improvement threshold (``min_delta``) and
how to persist (``on_improvement`` callback, e.g. a lambda calling
``save_classification_checkpoint(path, model, ...)``). This class never writes files itself.

Improvement rule: mode="max" -> ``value > best + min_delta``; mode="min" -> ``value < best - min_delta``.
NaN never counts as an improvement.

Requires: Python standard library only.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional

ImprovementCallback = Callable[[float, int], None]


class BestModelTracker:
    def __init__(
        self,
        mode: str = "max",
        min_delta: float = 0.0,
        on_improvement: Optional[ImprovementCallback] = None,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        if min_delta < 0:
            raise ValueError(f"min_delta must be >= 0, got {min_delta}")
        self.mode = mode
        self.min_delta = float(min_delta)
        self.on_improvement = on_improvement
        self.best: Optional[float] = None
        self.best_epoch: Optional[int] = None

    def is_improvement(self, value: float) -> bool:
        if math.isnan(value):
            return False
        if self.best is None:
            return True
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def update(self, value: float, epoch: int) -> bool:
        """Record ``value`` for ``epoch``; call the callback and return True on improvement."""
        value = float(value)
        if not self.is_improvement(value):
            return False
        self.best = value
        self.best_epoch = epoch
        if self.on_improvement is not None:
            self.on_improvement(value, epoch)
        return True

    def state_dict(self) -> Dict[str, Any]:
        return {"mode": self.mode, "min_delta": self.min_delta, "best": self.best, "best_epoch": self.best_epoch}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state.get("mode") != self.mode:
            raise ValueError(f"state mode {state.get('mode')!r} does not match tracker mode {self.mode!r}")
        self.best = state["best"]
        self.best_epoch = state["best_epoch"]
