# Purpose: Loads image-classification checkpoint weights into a caller-supplied architecture with explicit map_location and strictness.
"""Classification checkpoint loading (counterpart of save_checkpoint.py).

The caller constructs the architecture (same class/config as at save time) and passes it in; this
module never builds models. ``map_location`` is required (e.g. "cpu" or ``torch.device("cuda:0")``).
``weights_only=True`` by default: only tensors and plain Python containers are unpickled. Set it to
False only for checkpoints from a trusted source.

Accepted files: dicts produced by save_checkpoint.py (``"model"`` key) or a bare model state dict.
Optional ``optimizer``/``scheduler``/``scaler`` are restored only if supplied AND present.

``strict=False`` loads matching keys and reports missing/unexpected ones; shape mismatches are
always reported as an error listing every offending key (torch would raise anyway).

Requires: torch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
from torch import nn


@dataclass
class LoadResult:
    missing_keys: List[str] = field(default_factory=list)
    unexpected_keys: List[str] = field(default_factory=list)
    restored: List[str] = field(default_factory=list)
    epoch: Optional[int] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    class_names: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def _strip_prefix(state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if prefix and all(k.startswith(prefix) for k in state):
        return {k[len(prefix):]: v for k, v in state.items()}
    return state


def load_classification_checkpoint(
    path: str,
    model: nn.Module,
    map_location: Union[str, torch.device],
    *,
    strict: bool = True,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    weights_only: bool = True,
    strip_prefix: str = "",
) -> LoadResult:
    """Load weights (and optionally training state) from ``path`` into the supplied objects.

    Args:
        strip_prefix: Remove this key prefix when every key has it (e.g. "module." from DDP or
            "_orig_mod." from ``torch.compile``).
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=weights_only)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint at {path!r} is not a dict")
    is_wrapped = "model" in checkpoint and isinstance(checkpoint["model"], dict)
    state = checkpoint["model"] if is_wrapped else checkpoint
    state = _strip_prefix(state, strip_prefix)

    own = model.state_dict()
    mismatched = [
        f"{k}: checkpoint {tuple(v.shape)} vs model {tuple(own[k].shape)}"
        for k, v in state.items()
        if k in own and isinstance(v, torch.Tensor) and v.shape != own[k].shape
    ]
    if mismatched:
        raise ValueError("shape mismatch for keys:\n  " + "\n  ".join(mismatched))

    incompatible = model.load_state_dict(state, strict=strict)
    result = LoadResult(
        missing_keys=list(incompatible.missing_keys),
        unexpected_keys=list(incompatible.unexpected_keys),
        restored=["model"],
    )
    if not is_wrapped:
        return result

    for name, obj in (("optimizer", optimizer), ("scheduler", scheduler), ("scaler", scaler)):
        if obj is None:
            continue
        if name not in checkpoint:
            raise KeyError(f"{name} state requested but not present in checkpoint {path!r}")
        obj.load_state_dict(checkpoint[name])
        result.restored.append(name)

    result.epoch = checkpoint.get("epoch")
    result.metrics = dict(checkpoint.get("metrics") or {})
    names = checkpoint.get("class_names")
    result.class_names = list(names) if names is not None else None
    result.metadata = dict(checkpoint.get("metadata") or {})
    return result
