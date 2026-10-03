# Purpose: Builds a torch.optim.Adam optimizer from caller-supplied parameters and explicit hyperparameters for any PyTorch project.
"""Explicit Adam construction.

Note: Adam's ``weight_decay`` is classic L2 regularization (added to the gradient), which
interacts with the adaptive learning rate. For decoupled weight decay use AdamW (adamw.py).

Requires: torch.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Dict, Tuple, Union

import torch

ParamsLike = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]


def build_adam(
    params: ParamsLike,
    lr: float = 1e-3,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    amsgrad: bool = False,
    **kwargs: Any,
) -> torch.optim.Adam:
    """Create ``torch.optim.Adam``.

    Args:
        params: ``model.parameters()`` or a list of parameter-group dicts chosen by the caller.
        lr: Learning rate (> 0).
        betas: Coefficients for running averages of gradient and its square, each in [0, 1).
        eps: Denominator term for numerical stability (> 0).
        weight_decay: L2 penalty (>= 0), coupled with the adaptive update.
        amsgrad: Use the AMSGrad variant.
        **kwargs: Forwarded to Adam (e.g. ``foreach``, ``fused``, ``capturable``).
    """
    if lr <= 0:
        raise ValueError(f"lr must be > 0, got {lr}")
    if len(betas) != 2 or not all(0.0 <= b < 1.0 for b in betas):
        raise ValueError(f"betas must be two values in [0, 1), got {betas}")
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}")
    if weight_decay < 0:
        raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
    return torch.optim.Adam(
        params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, **kwargs
    )
