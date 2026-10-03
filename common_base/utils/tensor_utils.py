# Purpose: Provides small task-agnostic tensor helpers (nested batch device moves, shape/dtype assertions, detaching) for PyTorch projects.
"""Tensor utilities with no task policy.

``move_to_device`` recursively moves tensors inside mappings (returned as plain dicts),
lists/tuples/namedtuples and dataclass instances; non-tensor leaves are returned unchanged. ``assert_shape`` supports ``None`` as a wildcard
dimension and string names that must be consistent across calls via a shared ``dims`` dict.

Requires: torch.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, Dict, Optional, Sequence, Union

import torch


def move_to_device(batch: Any, device: Union[str, torch.device], non_blocking: bool = False) -> Any:
    """Recursively move all tensors in ``batch`` to ``device``."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=non_blocking)
    if isinstance(batch, Mapping):
        return {k: move_to_device(v, device, non_blocking) for k, v in batch.items()}
    if isinstance(batch, tuple) and hasattr(batch, "_fields"):
        return type(batch)(*(move_to_device(v, device, non_blocking) for v in batch))
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(v, device, non_blocking) for v in batch)
    if dataclasses.is_dataclass(batch) and not isinstance(batch, type):
        changes = {f.name: move_to_device(getattr(batch, f.name), device, non_blocking) for f in dataclasses.fields(batch) if f.init}
        return dataclasses.replace(batch, **changes)
    return batch


def detach_all(batch: Any) -> Any:
    """Recursively detach tensors (e.g. outputs kept for logging) without moving them."""
    if isinstance(batch, torch.Tensor):
        return batch.detach()
    if isinstance(batch, Mapping):
        return {k: detach_all(v) for k, v in batch.items()}
    if isinstance(batch, tuple) and hasattr(batch, "_fields"):
        return type(batch)(*(detach_all(v) for v in batch))
    if isinstance(batch, (list, tuple)):
        return type(batch)(detach_all(v) for v in batch)
    return batch


def assert_shape(
    tensor: torch.Tensor,
    expected: Sequence[Union[int, str, None]],
    name: str = "tensor",
    dims: Optional[Dict[str, int]] = None,
) -> Dict[str, int]:
    """Check ``tensor.shape`` against ``expected``.

    ``expected`` entries: int (exact), None (any), or str (named dim bound on first use in ``dims``).
    Returns the (updated) ``dims`` dict so it can be reused across tensors, e.g.::

        dims = assert_shape(x, ("B", "T", 512))
        assert_shape(mask, ("B", "T"), dims=dims)
    """
    dims = {} if dims is None else dims
    if tensor.ndim != len(expected):
        raise ValueError(f"{name}: expected {len(expected)} dims {tuple(expected)}, got shape {tuple(tensor.shape)}")
    for axis, (actual, want) in enumerate(zip(tensor.shape, expected)):
        if want is None:
            continue
        if isinstance(want, str):
            bound = dims.setdefault(want, actual)
            if bound != actual:
                raise ValueError(f"{name}: dim {axis} ('{want}') is {actual}, expected {bound}")
        elif actual != want:
            raise ValueError(f"{name}: dim {axis} is {actual}, expected {want} (shape {tuple(tensor.shape)})")
    return dims


def assert_dtype(tensor: torch.Tensor, dtypes: Union[torch.dtype, Sequence[torch.dtype]], name: str = "tensor") -> None:
    allowed = (dtypes,) if isinstance(dtypes, torch.dtype) else tuple(dtypes)
    if tensor.dtype not in allowed:
        raise TypeError(f"{name}: dtype {tensor.dtype} not in {allowed}")


def assert_same_device(*tensors: torch.Tensor) -> torch.device:
    """Raise if tensors live on different devices; return the common device."""
    if not tensors:
        raise ValueError("At least one tensor required")
    devices = {t.device for t in tensors}
    if len(devices) > 1:
        raise ValueError(f"Tensors are on different devices: {sorted(str(d) for d in devices)}")
    return devices.pop()


def to_python_scalars(values: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert 0-d / single-element tensors in a flat dict to Python numbers."""
    out: Dict[str, Any] = {}
    for k, v in values.items():
        out[k] = v.detach().item() if isinstance(v, torch.Tensor) and v.numel() == 1 else v
    return out
