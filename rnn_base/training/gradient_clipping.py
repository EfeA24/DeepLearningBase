# Purpose: Clips recurrent-model gradients by global norm or by value with explicit, caller-chosen thresholds.
"""Gradient clipping for recurrent networks (exploding-gradient mitigation).

Call order: ``loss.backward()`` -> clip -> ``optimizer.step()``.
With ``torch.amp.GradScaler`` call ``scaler.unscale_(optimizer)`` before clipping, otherwise
thresholds are compared against scaled gradients.

Modes:
- "norm": rescale all gradients so their combined ``norm_type`` norm is <= ``max_norm``;
  returns the total norm measured BEFORE clipping (useful for logging).
- "value": clamp every gradient element into [-clip_value, clip_value]; returns None.

Requires: torch only. No sibling files.
"""
from __future__ import annotations

from typing import Iterable, Optional, Union

import torch
from torch import nn

Parameters = Union[torch.Tensor, Iterable[torch.Tensor]]


def clip_gradients(
    parameters: Parameters,
    mode: str = "norm",
    max_norm: Optional[float] = None,
    clip_value: Optional[float] = None,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
) -> Optional[torch.Tensor]:
    """Clip gradients in place according to ``mode`` (see module docstring)."""
    if mode == "norm":
        if max_norm is None or max_norm <= 0:
            raise ValueError("mode='norm' requires a positive max_norm.")
        return nn.utils.clip_grad_norm_(
            parameters, max_norm=max_norm, norm_type=norm_type, error_if_nonfinite=error_if_nonfinite
        )
    if mode == "value":
        if clip_value is None or clip_value <= 0:
            raise ValueError("mode='value' requires a positive clip_value.")
        nn.utils.clip_grad_value_(parameters, clip_value=clip_value)
        return None
    raise ValueError(f"mode must be 'norm' or 'value', got {mode!r}.")


def gradient_norm(parameters: Parameters, norm_type: float = 2.0) -> torch.Tensor:
    """Total gradient norm without modifying gradients (0.0 if no parameter has a gradient)."""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    norms = torch.stack([torch.linalg.vector_norm(g, ord=norm_type) for g in grads])
    return torch.linalg.vector_norm(norms, ord=norm_type)
