# Purpose: Builds cosine-annealing learning-rate schedules (single cycle or warm restarts) for PyTorch optimizers with explicit step cadence.
"""Cosine learning-rate schedules.

Step cadence:
    ``T_max`` / ``T_0`` are counted in whatever unit you call ``scheduler.step()`` in.
    - Per-epoch: call ``scheduler.step()`` once after each epoch, ``T_max = num_epochs``.
    - Per-update: call after each ``optimizer.step()``, ``T_max = total optimizer updates``
      (``ceil(batches_per_epoch / accumulation_steps) * epochs``).
    Mixing cadences silently stretches or compresses the schedule.

Formula (single cycle): ``lr_t = eta_min + (base_lr - eta_min) * (1 + cos(pi * t / T_max)) / 2``.
After ``t > T_max`` torch's CosineAnnealingLR keeps following the cosine (lr rises again);
``build_cosine_lambda_schedule`` instead clamps at ``eta_min`` after ``total_steps``.

Requires: torch.
"""
from __future__ import annotations

import math

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, LambdaLR


def build_cosine_schedule(
    optimizer: torch.optim.Optimizer, T_max: int, eta_min: float = 0.0
) -> CosineAnnealingLR:
    """Single cosine decay from each group's initial lr to ``eta_min`` over ``T_max`` steps."""
    if T_max < 1:
        raise ValueError(f"T_max must be >= 1, got {T_max}")
    if eta_min < 0:
        raise ValueError(f"eta_min must be >= 0, got {eta_min}")
    return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)


def build_cosine_warm_restarts(
    optimizer: torch.optim.Optimizer, T_0: int, T_mult: int = 1, eta_min: float = 0.0
) -> CosineAnnealingWarmRestarts:
    """SGDR-style cosine with restarts every ``T_0 * T_mult**k`` steps."""
    if T_0 < 1:
        raise ValueError(f"T_0 must be >= 1, got {T_0}")
    if T_mult < 1:
        raise ValueError(f"T_mult must be >= 1, got {T_mult}")
    return CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)


def build_cosine_lambda_schedule(
    optimizer: torch.optim.Optimizer, total_steps: int, min_lr_ratio: float = 0.0
) -> LambdaLR:
    """Cosine decay as a multiplier of base lr, clamped at ``min_lr_ratio`` after ``total_steps``.

    Multiplier at step t: ``min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + cos(pi * min(t, total)/total))``.
    """
    if total_steps < 1:
        raise ValueError(f"total_steps must be >= 1, got {total_steps}")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError(f"min_lr_ratio must be in [0, 1], got {min_lr_ratio}")

    def lr_lambda(step: int) -> float:
        progress = min(step, total_steps) / total_steps
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
