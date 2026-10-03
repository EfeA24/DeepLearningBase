# Purpose: Counts total, trainable and frozen parameters of a supplied PyTorch model without printing.
"""Parameter counting.

Shared (tied) parameters are counted once by default (``nn.Module.parameters()`` already
deduplicates). Lazy (uninitialized) parameters raise because their size is unknown.

Requires: torch.
"""
from __future__ import annotations

from typing import Dict

from torch import nn
from torch.nn.parameter import UninitializedParameter


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Return ``{"total", "trainable", "frozen"}`` parameter counts."""
    total = trainable = 0
    for param in model.parameters():
        if isinstance(param, UninitializedParameter):
            raise ValueError("Model has uninitialized (lazy) parameters; run a forward pass first")
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def count_parameters_by_module(model: nn.Module, depth: int = 1) -> Dict[str, int]:
    """Parameter counts grouped by module-name prefix of ``depth`` components (e.g. "encoder")."""
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")
    counts: Dict[str, int] = {}
    for name, param in model.named_parameters():
        key = ".".join(name.split(".")[:depth]) if "." in name else name
        counts[key] = counts.get(key, 0) + param.numel()
    return counts


def format_count(n: int) -> str:
    """Human-readable count, e.g. 1234567 -> '1.23M'."""
    for unit, scale in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= scale:
            return f"{n / scale:.2f}{unit}"
    return str(n)
