# Purpose: Builds a torch.optim.SGD optimizer with explicit momentum, weight decay and Nesterov settings for image-classification training.
"""Explicit SGD construction.

Validation: Nesterov momentum requires ``momentum > 0`` and ``dampening == 0`` (torch rule).
``weight_decay`` is L2 added to the gradient (coupled). A typical CNN recipe is
``lr=0.1 * batch_size / 256, momentum=0.9, weight_decay=1e-4/5e-4`` but no recipe is applied here:
``lr`` is required.

Requires: torch.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Dict, Union

import torch

ParamsLike = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]


def build_sgd(
    params: ParamsLike,
    lr: float,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    nesterov: bool = False,
    dampening: float = 0.0,
    **kwargs: Any,
) -> torch.optim.SGD:
    """Create ``torch.optim.SGD`` with validated hyperparameters (defaults match torch)."""
    if lr <= 0:
        raise ValueError(f"lr must be > 0, got {lr}")
    if momentum < 0:
        raise ValueError(f"momentum must be >= 0, got {momentum}")
    if weight_decay < 0:
        raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
    if dampening < 0:
        raise ValueError(f"dampening must be >= 0, got {dampening}")
    if nesterov and (momentum <= 0 or dampening != 0):
        raise ValueError("nesterov=True requires momentum > 0 and dampening == 0")
    return torch.optim.SGD(
        params,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov,
        dampening=dampening,
        **kwargs,
    )
