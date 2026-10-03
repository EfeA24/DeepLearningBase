# Purpose: Resolves and validates a caller-selected torch device string without silently choosing CUDA for any PyTorch project.
"""Device resolution.

``resolve_device(spec)`` accepts "cpu", "cuda", "cuda:N", "mps", a ``torch.device``, or the
explicit opt-in "auto" (CUDA if available, else MPS if available, else CPU). Any other device is
validated: requesting an unavailable device raises instead of falling back.

Requires: torch.
"""
from __future__ import annotations

from typing import Union

import torch

DeviceLike = Union[str, torch.device]


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def validate_device(device: torch.device) -> torch.device:
    """Raise RuntimeError if ``device`` is not usable in this process."""
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f"cuda:{device.index} requested but only {torch.cuda.device_count()} device(s) visible")
    elif device.type == "mps":
        if not _mps_available():
            raise RuntimeError("MPS device requested but it is not available")
    elif device.type != "cpu":
        raise ValueError(f"Unsupported device type '{device.type}' (expected cpu, cuda or mps)")
    return device


def resolve_device(spec: DeviceLike) -> torch.device:
    """Turn ``spec`` into a validated ``torch.device``; "auto" must be passed explicitly."""
    if isinstance(spec, str) and spec.strip().lower() == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    if not isinstance(spec, (str, torch.device)):
        raise TypeError(f"device must be a str or torch.device, got {type(spec).__name__}")
    return validate_device(torch.device(spec))


def module_device(module: torch.nn.Module) -> torch.device:
    """Device of a module's first parameter/buffer; raises if it has none or spans several devices."""
    devices = {t.device for t in list(module.parameters()) + list(module.buffers())}
    if not devices:
        raise ValueError("Module has no parameters or buffers")
    if len(devices) > 1:
        raise ValueError(f"Module spans multiple devices: {sorted(str(d) for d in devices)}")
    return devices.pop()
