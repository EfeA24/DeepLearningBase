# Purpose: Documents the plain float32 training/inference path and provides dtype assertion helpers for PyTorch models and tensors.
"""Full precision (float32) path.

Float32 is the default: no autocast, no GradScaler. Two optional knobs affect float32 matmuls
on recent hardware and are NOT changed here unless the caller asks:
    - ``torch.backends.cuda.matmul.allow_tf32`` / ``torch.backends.cudnn.allow_tf32`` (Ampere+ TF32).
    - ``torch.set_float32_matmul_precision("highest" | "high" | "medium")``.
``set_float32_matmul_precision`` below is an explicit wrapper; calling it changes global state.

Requires: torch.
"""
from __future__ import annotations

from typing import Iterable, List, Tuple

import torch
from torch import nn


def assert_dtype(tensor: torch.Tensor, dtype: torch.dtype = torch.float32, name: str = "tensor") -> None:
    """Raise TypeError if ``tensor.dtype != dtype``."""
    if tensor.dtype != dtype:
        raise TypeError(f"{name} has dtype {tensor.dtype}, expected {dtype}")


def find_parameters_not_in_dtype(model: nn.Module, dtype: torch.dtype = torch.float32) -> List[Tuple[str, torch.dtype]]:
    """Return ``(name, dtype)`` for floating-point parameters/buffers whose dtype differs from ``dtype``."""
    named: Iterable[Tuple[str, torch.Tensor]] = list(model.named_parameters()) + list(model.named_buffers())
    return [(name, t.dtype) for name, t in named if t.is_floating_point() and t.dtype != dtype]


def assert_model_dtype(model: nn.Module, dtype: torch.dtype = torch.float32) -> None:
    """Raise TypeError listing floating-point params/buffers not in ``dtype``."""
    wrong = find_parameters_not_in_dtype(model, dtype)
    if wrong:
        preview = ", ".join(f"{n}:{d}" for n, d in wrong[:10])
        raise TypeError(f"{len(wrong)} floating tensors are not {dtype}: {preview}")


def set_float32_matmul_precision(precision: str) -> None:
    """Explicitly set global float32 matmul precision ("highest", "high" or "medium")."""
    if precision not in ("highest", "high", "medium"):
        raise ValueError(f"precision must be 'highest', 'high' or 'medium', got {precision!r}")
    torch.set_float32_matmul_precision(precision)
