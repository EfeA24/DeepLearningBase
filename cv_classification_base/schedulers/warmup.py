# Purpose: Builds linear-warmup learning-rate schedules (constant or cosine handoff, or SequentialLR chaining) for image-classification training.
"""Warmup LR schedules (all multiplicative on each param group's initial LR).

Step cadence: PER OPTIMIZER UPDATE is assumed (``warmup_steps``/``total_steps`` in updates); call
``scheduler.step()`` after every ``optimizer.step()``. Per-epoch use works too if every count is
expressed in epochs.

Warmup factor at scheduler step ``t < warmup_steps``:
    ``start_factor + (1 - start_factor) * t / warmup_steps``
With the default ``start_factor=0.0`` the very first update uses LR 0 (common convention);
pass e.g. ``start_factor=0.01`` to avoid that.

Handoffs:
    build_linear_warmup_scheduler  -> factor 1.0 after warmup (constant LR).
    build_warmup_cosine_scheduler  -> cosine from 1.0 to ``min_factor`` over
                                      ``total_steps - warmup_steps``, then held at ``min_factor``.
    chain_warmup_with              -> ``SequentialLR([warmup, main], milestones=[warmup_steps])``
                                      to hand off to any caller-built scheduler (main starts its
                                      own count at the milestone).

Requires: torch.
"""
from __future__ import annotations

import math

import torch
from torch.optim.lr_scheduler import LambdaLR, LRScheduler, SequentialLR


def _validate_warmup(warmup_steps: int, start_factor: float) -> None:
    if not isinstance(warmup_steps, int) or warmup_steps < 0:
        raise ValueError(f"warmup_steps must be an int >= 0, got {warmup_steps!r}")
    if not 0.0 <= start_factor <= 1.0:
        raise ValueError(f"start_factor must be in [0, 1], got {start_factor}")


def _warmup_factor(step: int, warmup_steps: int, start_factor: float) -> float:
    return start_factor + (1.0 - start_factor) * step / warmup_steps


def build_linear_warmup_scheduler(
    optimizer: torch.optim.Optimizer, warmup_steps: int, start_factor: float = 0.0
) -> LambdaLR:
    """Linear warmup to the initial LR, then constant."""
    _validate_warmup(warmup_steps, start_factor)

    def factor(step: int) -> float:
        if step < warmup_steps:
            return _warmup_factor(step, warmup_steps, start_factor)
        return 1.0

    return LambdaLR(optimizer, lr_lambda=factor)


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_factor: float = 0.0,
    start_factor: float = 0.0,
) -> LambdaLR:
    """Linear warmup then cosine decay to ``min_factor * initial_lr`` at ``total_steps``."""
    _validate_warmup(warmup_steps, start_factor)
    if not isinstance(total_steps, int) or total_steps <= warmup_steps:
        raise ValueError(f"total_steps must be an int > warmup_steps, got {total_steps!r}")
    if not 0.0 <= min_factor <= 1.0:
        raise ValueError(f"min_factor must be in [0, 1], got {min_factor}")
    decay_steps = total_steps - warmup_steps

    def factor(step: int) -> float:
        if step < warmup_steps:
            return _warmup_factor(step, warmup_steps, start_factor)
        progress = min(1.0, (step - warmup_steps) / decay_steps)
        return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=factor)


def chain_warmup_with(
    optimizer: torch.optim.Optimizer,
    main_scheduler: LRScheduler,
    warmup_steps: int,
    start_factor: float = 0.0,
) -> SequentialLR:
    """Warmup for ``warmup_steps`` scheduler steps, then hand off to ``main_scheduler``.

    ``main_scheduler`` must be built on the same optimizer and sized for the remaining steps.
    """
    _validate_warmup(warmup_steps, start_factor)
    if warmup_steps == 0:
        raise ValueError("warmup_steps must be >= 1 when chaining; use main_scheduler directly otherwise")
    warmup = build_linear_warmup_scheduler(optimizer, warmup_steps, start_factor)
    return SequentialLR(optimizer, schedulers=[warmup, main_scheduler], milestones=[warmup_steps])
