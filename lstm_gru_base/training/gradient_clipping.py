# Purpose: Clips recurrent-network gradients by global norm or by value with a caller-chosen threshold for stable LSTM/GRU training.
"""Gradient clipping for recurrent models.

Usage order: ``loss.backward()`` -> (``scaler.unscale_(optimizer)`` when using ``torch.amp.GradScaler``)
-> ``clip_gradients(...)`` -> ``optimizer.step()``. Clipping scaled gradients without unscaling first
applies the wrong threshold.

Modes:
    - ``"norm"``: rescales all gradients so their joint ``norm_type`` norm is <= ``max_value``; returns the
      total norm BEFORE clipping (useful for logging/diverge detection).
    - ``"value"``: clamps each gradient element to ``[-max_value, max_value]``; returns None.

Requires: torch only.
"""
from __future__ import annotations

from typing import Iterable, Optional, Union

import torch

Params = Union[torch.Tensor, Iterable[torch.Tensor]]


def clip_gradients(
    parameters: Params,
    max_value: float,
    mode: str = "norm",
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
) -> Optional[torch.Tensor]:
    """Clip gradients in-place; see module docstring for mode semantics."""
    if max_value <= 0:
        raise ValueError(f"max_value must be > 0, got {max_value}.")
    params = [parameters] if isinstance(parameters, torch.Tensor) else list(parameters)
    if mode == "norm":
        return torch.nn.utils.clip_grad_norm_(params, max_value, norm_type=norm_type, error_if_nonfinite=error_if_nonfinite)
    if mode == "value":
        torch.nn.utils.clip_grad_value_(params, max_value)
        return None
    raise ValueError(f"mode must be 'norm' or 'value', got {mode!r}.")


def gradient_norm(parameters: Params, norm_type: float = 2.0) -> torch.Tensor:
    """Total gradient norm without modifying gradients (parameters without grads are skipped)."""
    params = [parameters] if isinstance(parameters, torch.Tensor) else list(parameters)
    grads = [p.grad.detach() for p in params if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    norms = torch.stack([torch.linalg.vector_norm(g, ord=norm_type) for g in grads])
    return torch.linalg.vector_norm(norms, ord=norm_type)
