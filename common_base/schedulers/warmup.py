# Purpose: Builds learning-rate warmup schedules (linear warmup alone or handed off to a decay schedule) for PyTorch optimizers.
"""Warmup scheduling.

Two options:
    1. ``build_warmup_decay_schedule``: single LambdaLR with linear warmup then linear/cosine/
       constant decay. One object, one cadence, simple to checkpoint.
    2. ``build_sequential_warmup``: linear warmup LambdaLR followed by a caller-built scheduler via
       ``torch.optim.lr_scheduler.SequentialLR`` (handoff at ``warmup_steps``).

Warmup semantics:
    Multiplier at step t < warmup_steps is ``(t + 1) / warmup_steps`` so the first update uses a
    non-zero lr and step ``warmup_steps - 1`` reaches the full base lr. Steps are in the cadence you
    call ``scheduler.step()`` in (usually per optimizer update for warmup).
    With SequentialLR the post-warmup scheduler starts counting from 0 at the handoff, so its
    ``T_max``/``total_steps`` should be ``total_steps - warmup_steps``.

Requires: torch.
"""
from __future__ import annotations

import math

import torch
from torch.optim.lr_scheduler import LambdaLR, LRScheduler, SequentialLR


def warmup_factor(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, (step + 1) / warmup_steps)


def build_linear_warmup(optimizer: torch.optim.Optimizer, warmup_steps: int) -> LambdaLR:
    """Linear warmup to base lr, then constant."""
    if warmup_steps < 1:
        raise ValueError(f"warmup_steps must be >= 1, got {warmup_steps}")
    return LambdaLR(optimizer, lr_lambda=lambda step: warmup_factor(step, warmup_steps))


def build_warmup_decay_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    decay: str = "cosine",
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """Linear warmup for ``warmup_steps`` then ``decay`` ("cosine", "linear", "constant") to
    ``min_lr_ratio * base_lr`` at ``total_steps``; clamped afterwards.
    """
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
    if total_steps <= warmup_steps:
        raise ValueError(f"total_steps ({total_steps}) must exceed warmup_steps ({warmup_steps})")
    if decay not in ("cosine", "linear", "constant"):
        raise ValueError(f"decay must be 'cosine', 'linear' or 'constant', got {decay!r}")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError(f"min_lr_ratio must be in [0, 1], got {min_lr_ratio}")
    decay_steps = total_steps - warmup_steps

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return warmup_factor(step, warmup_steps)
        if decay == "constant":
            return 1.0
        progress = min(1.0, (step - warmup_steps) / decay_steps)
        if decay == "linear":
            shape = 1.0 - progress
        else:
            shape = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * shape

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_sequential_warmup(
    optimizer: torch.optim.Optimizer, warmup_steps: int, after_warmup: LRScheduler
) -> SequentialLR:
    """Compose linear warmup with an already constructed scheduler on the SAME optimizer.

    Note: constructing ``after_warmup`` already performed its initial step on the optimizer's lr;
    SequentialLR resets lr to the warmup schedule at construction, which is the intended handoff.
    """
    if warmup_steps < 1:
        raise ValueError(f"warmup_steps must be >= 1, got {warmup_steps}")
    if after_warmup.optimizer is not optimizer:
        raise ValueError("after_warmup must wrap the same optimizer instance")
    warmup = build_linear_warmup(optimizer, warmup_steps)
    return SequentialLR(optimizer, schedulers=[warmup, after_warmup], milestones=[warmup_steps])
