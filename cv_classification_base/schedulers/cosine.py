# Purpose: Builds a cosine-annealing learning-rate schedule with explicit total steps and minimum LR for image-classification training.
"""Cosine annealing LR schedule.

``lr_t = min_lr + (base_lr - min_lr) * (1 + cos(pi * t / total_steps)) / 2`` for ``t`` in
``[0, total_steps]``, using ``torch.optim.lr_scheduler.CosineAnnealingLR`` (no restarts).

Step cadence (the caller decides and must be consistent):
    per-epoch: ``total_steps = num_epochs``; call ``scheduler.step()`` once after each epoch.
    per-update: ``total_steps = num_epochs * updates_per_epoch`` (updates = optimizer steps, i.e.
        ``ceil(len(loader) / accumulation_steps)``); call ``scheduler.step()`` after each
        ``optimizer.step()``.
Stepping past ``total_steps`` makes the LR rise again (cosine is periodic in torch's recursive
form); stop stepping or pick a larger ``total_steps``.

Requires: torch.
"""
from __future__ import annotations

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR


def build_cosine_scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int, min_lr: float = 0.0
) -> CosineAnnealingLR:
    """Create a cosine schedule decaying from each group's initial LR to ``min_lr``."""
    if not isinstance(total_steps, int) or total_steps < 1:
        raise ValueError(f"total_steps must be an int >= 1, got {total_steps!r}")
    if min_lr < 0:
        raise ValueError(f"min_lr must be >= 0, got {min_lr}")
    return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=min_lr)
