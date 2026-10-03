# Purpose: Tracks a caller-selected validation metric with mode, patience and min_delta to decide early stopping in image-classification training.
"""Early stopping tracker (no checkpoint writing; pair with checkpoints/best_model.py if needed).

Improvement rule:
    mode="min": ``value < best - min_delta``
    mode="max": ``value > best + min_delta``
``step(value)`` returns True once ``patience`` consecutive non-improving steps have occurred.
NaN values never count as improvement. ``state_dict``/``load_state_dict`` allow resuming.

Requires: Python standard library only (metric values are plain floats).
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional


class EarlyStopping:
    def __init__(self, mode: str = "min", patience: int = 10, min_delta: float = 0.0) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        if not isinstance(patience, int) or patience < 1:
            raise ValueError(f"patience must be an int >= 1, got {patience!r}")
        if min_delta < 0:
            raise ValueError(f"min_delta must be >= 0, got {min_delta}")
        self.mode = mode
        self.patience = patience
        self.min_delta = float(min_delta)
        self.reset()

    def reset(self) -> None:
        self.best: Optional[float] = None
        self.best_step: int = -1
        self.num_bad_steps: int = 0
        self.num_steps: int = 0
        self.should_stop: bool = False

    def is_improvement(self, value: float) -> bool:
        if math.isnan(value):
            return False
        if self.best is None:
            return True
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def step(self, value: float) -> bool:
        """Register a new metric value; return True if training should stop."""
        value = float(value)
        if self.is_improvement(value):
            self.best = value
            self.best_step = self.num_steps
            self.num_bad_steps = 0
        else:
            self.num_bad_steps += 1
        self.num_steps += 1
        self.should_stop = self.num_bad_steps >= self.patience
        return self.should_stop

    def state_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best": self.best,
            "best_step": self.best_step,
            "num_bad_steps": self.num_bad_steps,
            "num_steps": self.num_steps,
            "should_stop": self.should_stop,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state.get("mode") != self.mode:
            raise ValueError(f"state mode {state.get('mode')!r} does not match tracker mode {self.mode!r}")
        self.best = state["best"]
        self.best_step = int(state["best_step"])
        self.num_bad_steps = int(state["num_bad_steps"])
        self.num_steps = int(state["num_steps"])
        self.should_stop = bool(state["should_stop"])
