# Purpose: Provides explicit opt-in automatic mixed precision configuration, autocast context and scaled backward/step helpers for image-classification training.
"""Mixed precision (AMP) helpers with explicit device and dtype.

Policy (caller decides, nothing is auto-selected):
    float16  + cuda -> autocast + ``torch.amp.GradScaler("cuda")`` (required to avoid underflow)
    bfloat16 + cuda -> autocast, no scaler (needs Ampere+; check ``torch.cuda.is_bf16_supported()``)
    bfloat16 + cpu  -> autocast, no scaler
    float32         -> autocast disabled (plain full precision)
float16 on CPU is rejected here because CPU fp16 autocast support/perf is limited and version
dependent; use bfloat16 on CPU.

Typical step::

    cfg = AmpConfig(device_type="cuda", dtype=torch.float16)
    scaler = create_grad_scaler(cfg)
    with autocast_context(cfg):
        loss = loss_fn(model(x), y)
    backward_and_step(loss, optimizer, scaler, model.parameters(), max_grad_norm=1.0)

Requires: torch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import torch

_SUPPORTED = {
    ("cuda", torch.float16),
    ("cuda", torch.bfloat16),
    ("cpu", torch.bfloat16),
}


@dataclass(frozen=True)
class AmpConfig:
    device_type: str
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if self.device_type not in ("cuda", "cpu"):
            raise ValueError(f"device_type must be 'cuda' or 'cpu', got {self.device_type!r}")
        if self.dtype != torch.float32 and (self.device_type, self.dtype) not in _SUPPORTED:
            raise ValueError(f"unsupported AMP combination: device={self.device_type}, dtype={self.dtype}")

    @property
    def enabled(self) -> bool:
        return self.dtype != torch.float32

    @property
    def needs_scaler(self) -> bool:
        return self.dtype == torch.float16


def autocast_context(config: AmpConfig) -> Any:
    """Return ``torch.amp.autocast`` for the config (a disabled context when dtype is float32)."""
    return torch.amp.autocast(device_type=config.device_type, dtype=config.dtype, enabled=config.enabled)


def create_grad_scaler(config: AmpConfig) -> Optional["torch.amp.GradScaler"]:
    """Return a GradScaler for float16, otherwise None."""
    if not config.needs_scaler:
        return None
    return torch.amp.GradScaler(config.device_type)


def backward_and_step(
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: Optional["torch.amp.GradScaler"],
    parameters: Optional[Iterable[torch.Tensor]] = None,
    max_grad_norm: Optional[float] = None,
    zero_grad: bool = True,
) -> None:
    """Backward + optional clipping + optimizer step, unscaling before clipping when scaled."""
    if max_grad_norm is not None and parameters is None:
        raise ValueError("parameters are required when max_grad_norm is set")
    params = list(parameters) if parameters is not None else []
    if scaler is not None:
        scaler.scale(loss).backward()
        if max_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        optimizer.step()
    if zero_grad:
        optimizer.zero_grad(set_to_none=True)
