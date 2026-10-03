# Purpose: Saves model/optimizer/scheduler/scaler state dicts plus caller metadata to an explicit checkpoint path for any PyTorch project.
"""Checkpoint saving.

Checkpoint layout (a plain dict, loadable with ``torch.load(..., weights_only=True)`` as long as
``metadata``/``extra`` contain only tensors and primitive containers/scalars/strings)::

    {
        "format_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),        # if supplied
        "scheduler": scheduler.state_dict(),        # if supplied
        "scaler": scaler.state_dict(),              # if supplied (GradScaler)
        "metadata": {...},                          # caller-provided (epoch, metrics, config ...)
        "extra": {...},                             # caller-provided additional state dicts
    }

DDP/compiled models: pass ``unwrap_model=True`` to strip ``DistributedDataParallel``/
``torch.compile`` wrappers so keys have no ``module.``/``_orig_mod.`` prefixes.
Writes go to a temporary file in the same directory and are then atomically renamed, so an
interrupted save does not corrupt an existing checkpoint. In multi-process training call this
on the main process only.

Requires: torch.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Optional

import torch
from torch import nn

CHECKPOINT_FORMAT_VERSION = 1


def unwrap(model: nn.Module) -> nn.Module:
    """Strip DDP/FSDP-style ``.module`` and ``torch.compile`` ``._orig_mod`` wrappers."""
    while True:
        if hasattr(model, "_orig_mod") and isinstance(model._orig_mod, nn.Module):
            model = model._orig_mod
        elif isinstance(model, (nn.parallel.DistributedDataParallel, nn.DataParallel)):
            model = model.module
        else:
            return model


def build_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
    unwrap_model: bool = True,
) -> Dict[str, Any]:
    """Assemble the checkpoint dict without writing it."""
    module = unwrap(model) if unwrap_model else model
    checkpoint: Dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model": module.state_dict(),
        "metadata": dict(metadata or {}),
    }
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    if extra:
        checkpoint["extra"] = dict(extra)
    return checkpoint


def save_checkpoint(
    path: Any,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
    create_dirs: bool = False,
    unwrap_model: bool = True,
) -> str:
    """Save a checkpoint dict to ``path`` and return the resolved path as a string.

    Args:
        path: Destination file path (str or PathLike).
        create_dirs: Create missing parent directories; otherwise a missing parent raises.
        Other args: see ``build_checkpoint``.
    """
    import os
    from pathlib import Path

    target = Path(path)
    parent = target.parent
    if not parent.exists():
        if not create_dirs:
            raise FileNotFoundError(f"Parent directory does not exist: {parent} (pass create_dirs=True)")
        parent.mkdir(parents=True, exist_ok=True)
    checkpoint = build_checkpoint(model, optimizer, scheduler, scaler, metadata, extra, unwrap_model)
    tmp_path = target.with_name(target.name + ".tmp")
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, target)
    return str(target)
