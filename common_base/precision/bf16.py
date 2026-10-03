# Purpose: Provides explicit bfloat16 autocast contexts and capability checks for PyTorch training and inference, raising instead of silently falling back.
"""BFloat16 mixed precision.

bfloat16 keeps float32's exponent range (8 bits) with a shorter mantissa, so no GradScaler is
needed. It needs hardware support: CUDA compute capability >= 8.0 (Ampere+) for fast bf16, or a
CPU with bf16-capable kernels. The caller decides whether to use it; if requested and unsupported,
``require_bf16`` raises rather than silently falling back to fp16/fp32.

Notes:
    - Reduced mantissa precision can affect very small updates; keep master weights in float32.
    - Outputs produced under autocast may be bfloat16; cast to float32 for metrics/losses that
      need precision.

Requires: torch.
"""
from __future__ import annotations

from typing import Optional

import torch


def is_bf16_supported(device: torch.device | str) -> bool:
    """Return whether bf16 autocast is expected to work on ``device`` ("cpu", "cuda", "cuda:1", ...).

    CPU returns True (bf16 autocast exists on CPU; actual speed depends on the processor).
    """
    dev = torch.device(device)
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            return False
        index: Optional[int] = dev.index if dev.index is not None else torch.cuda.current_device()
        with torch.cuda.device(index):
            return bool(torch.cuda.is_bf16_supported())
    if dev.type == "cpu":
        return True
    return False


def require_bf16(device: torch.device | str) -> None:
    """Raise RuntimeError if bf16 is not supported on ``device``."""
    if not is_bf16_supported(device):
        raise RuntimeError(f"bfloat16 was requested but is not supported on device '{device}'")


def bf16_autocast(device_type: str = "cuda", enabled: bool = True, check_support: bool = True):
    """Return a bfloat16 autocast context; validates support first unless ``check_support=False``."""
    if enabled and check_support:
        require_bf16(device_type)
    return torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=enabled)


def bf16_inference_context(device_type: str = "cuda"):
    """Inference mode + bfloat16 autocast after a support check (performed immediately)."""
    from contextlib import contextmanager

    require_bf16(device_type)

    @contextmanager
    def _context():
        with torch.inference_mode(), torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
            yield

    return _context()
