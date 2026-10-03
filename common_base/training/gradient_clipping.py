# Purpose: Clips gradients by global norm or by value between backward and optimizer step for any PyTorch training loop.
"""Gradient clipping helpers.

Call order::

    loss.backward()                       # or scaler.scale(loss).backward()
    scaler.unscale_(optimizer)            # REQUIRED first when using torch.amp.GradScaler
    clip_gradients(model.parameters(), mode="norm", max_value=1.0)
    optimizer.step()                      # or scaler.step(optimizer)

Modes:
    - "norm": rescales all gradients so their combined p-norm is at most ``max_value``
      (``torch.nn.utils.clip_grad_norm_``). Returns the total norm BEFORE clipping.
    - "value": clamps each gradient element into ``[-max_value, max_value]``
      (``torch.nn.utils.clip_grad_value_``). Returns None.

Requires: torch.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Optional, Union

import torch

ParamsLike = Union[torch.Tensor, Iterable[torch.Tensor]]


def clip_gradients(
    parameters: ParamsLike,
    mode: str = "norm",
    max_value: float = 1.0,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
) -> Optional[torch.Tensor]:
    """Clip gradients in place.

    Args:
        parameters: Tensors (usually ``model.parameters()``) whose ``.grad`` will be clipped.
        mode: "norm" or "value".
        max_value: Max total norm ("norm") or max absolute element value ("value"); > 0.
        norm_type: p of the p-norm for "norm" mode (``float('inf')`` allowed).
        error_if_nonfinite: In "norm" mode raise if the total norm is NaN/Inf.

    Returns:
        Total gradient norm before clipping for "norm" mode, else None.
    """
    if max_value <= 0:
        raise ValueError(f"max_value must be > 0, got {max_value}")
    if mode == "norm":
        return torch.nn.utils.clip_grad_norm_(
            parameters, max_norm=max_value, norm_type=norm_type, error_if_nonfinite=error_if_nonfinite
        )
    if mode == "value":
        torch.nn.utils.clip_grad_value_(parameters, clip_value=max_value)
        return None
    raise ValueError(f"mode must be 'norm' or 'value', got {mode!r}")


def grad_norm(parameters: ParamsLike, norm_type: float = 2.0) -> torch.Tensor:
    """Compute the total gradient norm without modifying gradients (params without grad skipped)."""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    device = grads[0].device
    if math.isinf(norm_type):
        return torch.stack([g.abs().max().to(device) for g in grads]).max()
    return torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(g, norm_type).to(device) for g in grads]), norm_type
    )
