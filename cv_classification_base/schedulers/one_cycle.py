# Purpose: Builds a per-batch OneCycle learning-rate schedule with explicit total steps for image-classification training.
"""OneCycleLR construction (Smith & Topin, 2018).

Step cadence: PER OPTIMIZER UPDATE. Call ``scheduler.step()`` after every ``optimizer.step()``.
``total_steps`` must equal the number of optimizer updates in the whole run
(``num_epochs * ceil(len(train_loader) / accumulation_steps)``); stepping more than
``total_steps`` times raises a ValueError inside torch.

LR path: starts at ``max_lr / div_factor``, rises to ``max_lr`` over ``pct_start * total_steps``,
then anneals to ``max_lr / (div_factor * final_div_factor)``. OneCycle overrides the optimizer's
initial LR. With ``cycle_momentum=True`` the optimizer must have ``momentum`` (SGD) or ``betas``
(Adam/AdamW).

Requires: torch.
"""
from __future__ import annotations

from typing import Sequence, Union

import torch
from torch.optim.lr_scheduler import OneCycleLR


def build_one_cycle_scheduler(
    optimizer: torch.optim.Optimizer,
    max_lr: Union[float, Sequence[float]],
    total_steps: int,
    pct_start: float = 0.3,
    anneal_strategy: str = "cos",
    div_factor: float = 25.0,
    final_div_factor: float = 1e4,
    cycle_momentum: bool = True,
    three_phase: bool = False,
) -> OneCycleLR:
    """Create ``OneCycleLR`` with validated arguments (defaults match torch)."""
    if not isinstance(total_steps, int) or total_steps < 1:
        raise ValueError(f"total_steps must be an int >= 1, got {total_steps!r}")
    if not 0.0 < pct_start < 1.0:
        raise ValueError(f"pct_start must be in (0, 1), got {pct_start}")
    if anneal_strategy not in ("cos", "linear"):
        raise ValueError(f"anneal_strategy must be 'cos' or 'linear', got {anneal_strategy!r}")
    if div_factor <= 0 or final_div_factor <= 0:
        raise ValueError("div_factor and final_div_factor must be > 0")
    return OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy=anneal_strategy,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
        cycle_momentum=cycle_momentum,
        three_phase=three_phase,
    )
