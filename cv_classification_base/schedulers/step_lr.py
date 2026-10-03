# Purpose: Builds step and multi-step learning-rate decay schedules with explicit epoch milestones for image-classification training.
"""Step-decay LR schedules.

``build_step_lr``: multiply LR by ``gamma`` every ``step_size`` scheduler steps.
``build_multistep_lr``: multiply LR by ``gamma`` at each milestone (e.g. ``[30, 60, 90]``,
the classic ImageNet ResNet recipe with ``gamma=0.1``).

Step cadence: these are normally stepped ONCE PER EPOCH (``scheduler.step()`` after each epoch),
so ``step_size``/``milestones`` are in epochs. If you step per update instead, express them in
optimizer updates.

Requires: torch.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch.optim.lr_scheduler import MultiStepLR, StepLR


def build_step_lr(optimizer: torch.optim.Optimizer, step_size: int, gamma: float = 0.1) -> StepLR:
    if not isinstance(step_size, int) or step_size < 1:
        raise ValueError(f"step_size must be an int >= 1, got {step_size!r}")
    if not 0.0 < gamma <= 1.0:
        raise ValueError(f"gamma must be in (0, 1], got {gamma}")
    return StepLR(optimizer, step_size=step_size, gamma=gamma)


def build_multistep_lr(
    optimizer: torch.optim.Optimizer, milestones: Sequence[int], gamma: float = 0.1
) -> MultiStepLR:
    milestones = list(milestones)
    if not milestones or any(not isinstance(m, int) or m < 1 for m in milestones):
        raise ValueError(f"milestones must be a non-empty list of ints >= 1, got {milestones}")
    if milestones != sorted(set(milestones)):
        raise ValueError(f"milestones must be strictly increasing, got {milestones}")
    if not 0.0 < gamma <= 1.0:
        raise ValueError(f"gamma must be in (0, 1], got {gamma}")
    return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
