# Purpose: Builds a torch.optim.SGD optimizer with explicit momentum, weight decay and Nesterov validation for any PyTorch project.
"""Explicit SGD construction.

``weight_decay`` in torch SGD is an L2 term added to the gradient. Nesterov momentum requires
``momentum > 0`` and ``dampening == 0`` (torch raises otherwise; validated here with clearer errors).

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
    """Create ``torch.optim.SGD``. ``lr`` has no default on purpose: choose it for your task.

    Args:
        params: ``model.parameters()`` or explicit param-group dicts.
        lr: Learning rate (> 0).
        momentum: Momentum factor (>= 0).
        weight_decay: L2 penalty (>= 0).
        nesterov: Enable Nesterov momentum; requires momentum > 0 and dampening == 0.
        dampening: Momentum dampening (>= 0).
        **kwargs: Forwarded to SGD (e.g. ``foreach``, ``maximize``).
    """
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
