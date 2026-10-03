# Purpose: Resolves and validates a caller-selected torch device for image-classification training and inference without silent CUDA selection.
"""Device resolution (consistent with common_base/utils/device.py).

``resolve_device("cpu")``, ``resolve_device("cuda:1")``, ``resolve_device("mps")`` validate that the
requested backend exists and raise otherwise (no silent fallback). ``"auto"`` is accepted only when
the caller passes it explicitly and picks cuda -> mps -> cpu in that order.

Nothing runs at import; availability is queried only inside the function.

Requires: torch.
"""
from __future__ import annotations

from typing import Union

import torch


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def resolve_device(device: Union[str, torch.device]) -> torch.device:
    """Return a validated ``torch.device`` for the caller's explicit choice."""
    if isinstance(device, str) and device.strip().lower() == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"device {resolved} requested but CUDA is not available")
        if resolved.index is not None and resolved.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"device {resolved} requested but only {torch.cuda.device_count()} CUDA device(s) exist"
            )
    elif resolved.type == "mps":
        if not _mps_available():
            raise RuntimeError("device 'mps' requested but MPS is not available")
    elif resolved.type != "cpu":
        raise ValueError(f"unsupported device type {resolved.type!r}; use 'cpu', 'cuda[:i]', 'mps' or 'auto'")
    return resolved


def describe_device(device: torch.device) -> str:
    """Human-readable device description (no side effects beyond querying properties)."""
    device = torch.device(device)
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        return f"cuda:{index} ({torch.cuda.get_device_name(index)})"
    return device.type
