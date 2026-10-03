# Purpose: Provides explicit float16 autocast contexts and GradScaler creation for PyTorch mixed-precision training and inference, with caveats documented.
"""Float16 mixed precision.

Training requirements and caveats:
    - Use ``torch.amp.autocast(device_type, dtype=torch.float16)`` around forward + loss only;
      run ``backward`` outside autocast.
    - A GradScaler is REQUIRED for training: fp16's narrow exponent range makes small gradients
      underflow to zero. Unscale before gradient clipping (``scaler.unscale_(optimizer)``).
    - Keep master weights in float32 (do not ``model.half()`` for training).
    - The scaler may skip optimizer steps whose gradients contain inf/NaN; LR schedulers stepped
      unconditionally then run slightly ahead (torch warns about this on the first step).
    - Primarily a CUDA feature; CPU fp16 autocast support is version- and op-dependent.

Inference:
    ``fp16_inference_context`` combines ``torch.inference_mode`` and fp16 autocast; no scaler.

Requires: torch >= 2.1 (``torch.amp.GradScaler(device)`` needs >= 2.3; falls back to
``torch.cuda.amp.GradScaler`` on older versions).
"""
from __future__ import annotations

from typing import Any

import torch


def fp16_autocast(device_type: str = "cuda", enabled: bool = True):
    """Return a float16 autocast context for ``device_type``."""
    return torch.amp.autocast(device_type=device_type, dtype=torch.float16, enabled=enabled)


def make_fp16_scaler(device_type: str = "cuda", init_scale: float = 2.0**16, enabled: bool = True) -> Any:
    """Create a GradScaler for float16 training."""
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device_type, init_scale=init_scale, enabled=enabled)
    if device_type != "cuda":
        raise RuntimeError("Non-CUDA GradScaler requires torch >= 2.3")
    return torch.cuda.amp.GradScaler(init_scale=init_scale, enabled=enabled)


def fp16_inference_context(device_type: str = "cuda"):
    """Inference mode + float16 autocast (outputs may be float16; cast before metric math)."""
    from contextlib import contextmanager

    @contextmanager
    def _context():
        with torch.inference_mode(), torch.amp.autocast(device_type=device_type, dtype=torch.float16):
            yield

    return _context()


def fp16_train_step(
    model: torch.nn.Module,
    batch_inputs: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: Any,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device_type: str = "cuda",
    max_grad_norm: float | None = None,
) -> float:
    """Reference single fp16 training step; returns the (unscaled) loss as a float.

    Inputs/targets must already be on the model's device.
    """
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type=device_type, dtype=torch.float16):
        loss = loss_fn(model(batch_inputs), targets)
    scaler.scale(loss).backward()
    if max_grad_norm is not None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
    return float(loss.detach())
