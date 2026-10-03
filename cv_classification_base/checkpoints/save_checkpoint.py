# Purpose: Saves image-classification training state dicts plus plain caller metadata (epoch, metrics, class names) to an explicit checkpoint path.
"""Classification checkpoint saving.

Checkpoint layout (a plain dict saved with ``torch.save``)::

    {
        "format_version": 1,
        "model": model.state_dict(),
        "optimizer": ... | absent, "scheduler": ... | absent, "scaler": ... | absent,
        "epoch": int | None,
        "metrics": {str: float},
        "class_names": [str] | None,     # index i -> class name (defines the label order)
        "metadata": {str: plain values}  # caller extras: str/int/float/bool/None/list/dict
    }

Only state dicts and plain Python values are stored so the file loads with
``torch.load(..., weights_only=True)``. Arbitrary objects are rejected rather than pickled.
Writes go to ``<path>.tmp`` then ``os.replace`` (atomic on the same filesystem). Parent
directories are created only when ``create_dirs=True``.

Requires: torch.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import torch
from torch import nn

_PLAIN_TYPES = (str, int, float, bool, type(None))


def _check_plain(value: Any, where: str) -> None:
    if isinstance(value, _PLAIN_TYPES):
        return
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _check_plain(item, f"{where}[{i}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, (str, int)):
                raise TypeError(f"{where} has non-str/int key {key!r}")
            _check_plain(item, f"{where}[{key!r}]")
        return
    raise TypeError(
        f"{where} has unsupported type {type(value).__name__}; store only plain Python values"
    )


def save_classification_checkpoint(
    path: str,
    model: nn.Module,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Mapping[str, float]] = None,
    class_names: Optional[Sequence[str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    create_dirs: bool = False,
) -> str:
    """Save a checkpoint to ``path`` and return the path."""
    import os

    parent = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(parent):
        if not create_dirs:
            raise FileNotFoundError(f"directory {parent!r} does not exist (pass create_dirs=True)")
        os.makedirs(parent, exist_ok=True)

    metrics_dict = {str(k): float(v) for k, v in (metrics or {}).items()}
    names = list(class_names) if class_names is not None else None
    if names is not None and not all(isinstance(n, str) for n in names):
        raise TypeError("class_names must be strings")
    extra = dict(metadata or {})
    _check_plain(extra, "metadata")

    checkpoint: Dict[str, Any] = {
        "format_version": 1,
        "model": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics_dict,
        "class_names": names,
        "metadata": extra,
    }
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()

    tmp_path = f"{path}.tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)
    return path
