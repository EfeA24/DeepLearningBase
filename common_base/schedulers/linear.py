# Purpose: Builds a linear learning-rate schedule with explicit start/end factors and total-step handling for PyTorch optimizers.
"""Linear learning-rate schedule.

Semantics:
    The multiplier applied to each param group's base lr moves linearly from ``start_factor``
    at step 0 to ``end_factor`` at step ``total_steps`` and stays at ``end_factor`` afterwards.
    ``total_steps`` is in the unit you call ``scheduler.step()`` in (epochs or optimizer updates).

    Example: linear decay to zero over all updates -> ``start_factor=1.0, end_factor=0.0``.

Unlike ``torch.optim.lr_scheduler.LinearLR`` (whose start_factor must be > 0), this LambdaLR-based
version allows ``start_factor == 0`` (useful for warmup from zero).

Requires: torch.
"""
from __future__ import annotations

import torch
from torch.optim.lr_scheduler import LambdaLR


def linear_factor(step: int, total_steps: int, start_factor: float, end_factor: float) -> float:
    """Multiplier at ``step``; clamped to ``end_factor`` for ``step >= total_steps``."""
    if step >= total_steps:
        return end_factor
    return start_factor + (end_factor - start_factor) * (step / total_steps)


def build_linear_schedule(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    start_factor: float = 1.0,
    end_factor: float = 0.0,
) -> LambdaLR:
    """Create a LambdaLR that interpolates the lr multiplier linearly.

    Args:
        optimizer: Optimizer whose groups' ``initial_lr`` is the base lr.
        total_steps: Number of scheduler steps to reach ``end_factor`` (>= 1).
        start_factor: Multiplier at step 0 (>= 0).
        end_factor: Multiplier at and after ``total_steps`` (>= 0).
    """
    if total_steps < 1:
        raise ValueError(f"total_steps must be >= 1, got {total_steps}")
    if start_factor < 0 or end_factor < 0:
        raise ValueError("start_factor and end_factor must be >= 0")
    return LambdaLR(optimizer, lr_lambda=lambda step: linear_factor(step, total_steps, start_factor, end_factor))
