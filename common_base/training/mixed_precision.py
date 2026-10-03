# Purpose: Provides an opt-in automatic mixed precision (autocast + GradScaler) helper for PyTorch training loops on CUDA or CPU.
"""Opt-in mixed precision helper built on ``torch.amp``.

Usage::

    amp = MixedPrecision(device_type="cuda", dtype=torch.float16, enabled=True)
    with amp.autocast():
        loss = loss_fn(model(x), y)
    amp.backward(loss)
    amp.unscale_(optimizer)               # before gradient clipping
    clip_grad_norm_(model.parameters(), 1.0)
    amp.step(optimizer)                   # scaler.step + scaler.update when scaling
    optimizer.zero_grad(set_to_none=True)

Device / dtype constraints:
    - float16 on CUDA needs a GradScaler to avoid gradient underflow (enabled automatically).
    - bfloat16 has float32's exponent range, so no scaler is used; requires hardware support
      (CUDA Ampere+ or recent CPUs). This helper does not check capability; see precision/bf16.py.
    - CPU autocast supports bfloat16 (float16 support depends on the torch version/ops).
    - When ``enabled=False`` all methods degrade to the plain float32 path.
    - Compute the loss inside autocast; keep optimizer state and master weights in float32.

Requires: torch >= 2.1 (torch.amp.GradScaler with device argument needs >= 2.3; falls back to
torch.cuda.amp.GradScaler on older versions).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


def make_grad_scaler(device_type: str, enabled: bool = True) -> Any:
    """Create a GradScaler for ``device_type`` using the newest available API."""
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device_type, enabled=enabled)
    if device_type != "cuda":
        raise RuntimeError("GradScaler for non-CUDA devices requires torch >= 2.3")
    return torch.cuda.amp.GradScaler(enabled=enabled)


class MixedPrecision:
    """Bundles autocast and optional loss scaling behind explicit options.

    Args:
        device_type: "cuda", "cpu", or another autocast-capable device type.
        dtype: torch.float16 or torch.bfloat16.
        enabled: Master switch; False gives a pure float32 path.
        use_scaler: Override scaler use; default is True only for float16.
    """

    def __init__(
        self,
        device_type: str,
        dtype: torch.dtype = torch.float16,
        enabled: bool = True,
        use_scaler: Optional[bool] = None,
    ) -> None:
        if dtype not in _SUPPORTED_DTYPES:
            raise ValueError(f"dtype must be torch.float16 or torch.bfloat16, got {dtype}")
        self.device_type = device_type
        self.dtype = dtype
        self.enabled = enabled
        scaling = (dtype == torch.float16) if use_scaler is None else use_scaler
        self.scaler = make_grad_scaler(device_type, enabled=True) if (enabled and scaling) else None

    def autocast(self):
        """Return an autocast context (a disabled, no-op autocast when ``enabled=False``)."""
        return torch.amp.autocast(device_type=self.device_type, dtype=self.dtype, enabled=self.enabled)

    def backward(self, loss: torch.Tensor) -> None:
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        """Unscale gradients in place (call before clipping); no-op without a scaler."""
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """Optimizer step; with a scaler, skips steps with inf/NaN grads and updates the scale."""
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def state_dict(self) -> Dict[str, Any]:
        return {} if self.scaler is None else self.scaler.state_dict()

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if self.scaler is not None and state:
            self.scaler.load_state_dict(state)
