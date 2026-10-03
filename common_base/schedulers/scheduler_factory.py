# Purpose: Creates supported PyTorch learning-rate schedulers from an explicit name and config dict, with no hidden global defaults.
"""Explicit scheduler factory.

Supported names and REQUIRED config keys (in units of ``scheduler.step()`` calls):
    - "constant": none (LambdaLR returning 1.0).
    - "step": ``step_size`` (optional ``gamma``, default torch 0.1).
    - "multistep": ``milestones`` (optional ``gamma``).
    - "exponential": ``gamma``.
    - "cosine": ``T_max`` (optional ``eta_min``).
    - "linear": ``total_steps`` (optional ``start_factor`` 1.0, ``end_factor`` 0.0).
    - "warmup_cosine" / "warmup_linear": ``warmup_steps``, ``total_steps`` (optional ``min_lr_ratio``).
    - "plateau": ReduceLROnPlateau kwargs (``mode``, ``factor``, ``patience`` ...); remember
      ``scheduler.step(metric)`` must receive the monitored metric.

Unknown names or missing required keys raise ValueError. Self-contained (does not import the
sibling cosine/linear/warmup files).

Requires: torch.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Dict, Optional, Sequence

import torch
from torch.optim import lr_scheduler

SUPPORTED_SCHEDULERS = (
    "constant", "step", "multistep", "exponential", "cosine", "linear",
    "warmup_cosine", "warmup_linear", "plateau",
)


def _require(config: Dict[str, Any], keys: Sequence[str], name: str) -> None:
    missing = [k for k in keys if k not in config]
    if missing:
        raise ValueError(f"Scheduler {name!r} requires config keys {missing}")


def _warmup_lambda(warmup_steps: int, total_steps: int, decay: str, min_lr_ratio: float):
    if warmup_steps < 0 or total_steps <= warmup_steps:
        raise ValueError("Require 0 <= warmup_steps < total_steps")

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / (total_steps - warmup_steps))
        shape = 1.0 - progress if decay == "linear" else 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * shape

    return lr_lambda


def build_scheduler(
    name: str, optimizer: torch.optim.Optimizer, config: Optional[Mapping[str, Any]] = None
) -> Any:
    """Construct a scheduler by name; see module docstring for required keys."""
    key = name.lower().strip()
    cfg: Dict[str, Any] = dict(config or {})
    if key == "constant":
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    if key == "step":
        _require(cfg, ["step_size"], key)
        return lr_scheduler.StepLR(optimizer, **cfg)
    if key == "multistep":
        _require(cfg, ["milestones"], key)
        return lr_scheduler.MultiStepLR(optimizer, **cfg)
    if key == "exponential":
        _require(cfg, ["gamma"], key)
        return lr_scheduler.ExponentialLR(optimizer, **cfg)
    if key == "cosine":
        _require(cfg, ["T_max"], key)
        return lr_scheduler.CosineAnnealingLR(optimizer, **cfg)
    if key == "linear":
        _require(cfg, ["total_steps"], key)
        total = int(cfg["total_steps"])
        start = float(cfg.get("start_factor", 1.0))
        end = float(cfg.get("end_factor", 0.0))
        if total < 1:
            raise ValueError("total_steps must be >= 1")
        return lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda s: end if s >= total else start + (end - start) * s / total
        )
    if key in ("warmup_cosine", "warmup_linear"):
        _require(cfg, ["warmup_steps", "total_steps"], key)
        fn = _warmup_lambda(
            int(cfg["warmup_steps"]),
            int(cfg["total_steps"]),
            "linear" if key == "warmup_linear" else "cosine",
            float(cfg.get("min_lr_ratio", 0.0)),
        )
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=fn)
    if key == "plateau":
        return lr_scheduler.ReduceLROnPlateau(optimizer, **cfg)
    raise ValueError(f"Unknown scheduler {name!r}; supported: {', '.join(SUPPORTED_SCHEDULERS)}")
