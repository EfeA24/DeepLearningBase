# Purpose: Loads PyTorch checkpoints with explicit map_location and restores only the requested state objects while reporting key and shape mismatches.
"""Checkpoint loading.

Expects the dict layout written by save.py (``"model"``, ``"optimizer"``, ``"scheduler"``,
``"scaler"``, ``"metadata"``) but also accepts a raw ``state_dict`` for the model when
``model_key=None``.

Safety: ``weights_only=True`` by default (no arbitrary unpickling). Only set it to False for
checkpoints you trust completely.

Shape mismatches: ``nn.Module.load_state_dict`` raises on shape mismatch even with
``strict=False``. ``load_model_state`` first compares shapes; mismatched tensors are either
reported as an error (default) or skipped when ``skip_shape_mismatch=True`` (e.g. replacing a
classification head with a different number of classes).

Requires: torch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
from torch import nn


@dataclass
class LoadReport:
    """What happened when loading model weights."""

    missing_keys: List[str] = field(default_factory=list)
    unexpected_keys: List[str] = field(default_factory=list)
    shape_mismatches: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not (self.missing_keys or self.unexpected_keys or self.shape_mismatches)

    def summary(self) -> str:
        lines = []
        if self.missing_keys:
            lines.append(f"missing keys ({len(self.missing_keys)}): {self.missing_keys}")
        if self.unexpected_keys:
            lines.append(f"unexpected keys ({len(self.unexpected_keys)}): {self.unexpected_keys}")
        for name, ckpt_shape, model_shape in self.shape_mismatches:
            lines.append(f"shape mismatch '{name}': checkpoint {ckpt_shape} vs model {model_shape}")
        return "\n".join(lines) if lines else "all keys matched"


def read_checkpoint(path: Any, map_location: Any, weights_only: bool = True) -> Any:
    """``torch.load`` with mandatory ``map_location`` (e.g. "cpu", "cuda:0", a torch.device)."""
    return torch.load(path, map_location=map_location, weights_only=weights_only)


def strip_prefix(state_dict: Mapping[str, Any], prefix: str) -> Dict[str, Any]:
    """Remove ``prefix`` (e.g. "module." or "_orig_mod.") from keys that start with it."""
    return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state_dict.items()}


def load_model_state(
    model: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    strict: bool = True,
    skip_shape_mismatch: bool = False,
) -> LoadReport:
    """Load ``state_dict`` into ``model`` and return a ``LoadReport``.

    Raises:
        RuntimeError: shape mismatches (unless skipped) or, when ``strict``, missing/unexpected keys.
    """
    model_state = model.state_dict()
    report = LoadReport()
    filtered: Dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        if name in model_state and hasattr(tensor, "shape") and tensor.shape != model_state[name].shape:
            report.shape_mismatches.append((name, tuple(tensor.shape), tuple(model_state[name].shape)))
            continue
        filtered[name] = tensor
    if report.shape_mismatches and not skip_shape_mismatch:
        raise RuntimeError("Shape mismatches while loading checkpoint:\n" + report.summary())
    result = model.load_state_dict(filtered, strict=False)
    skipped = {name for name, _, _ in report.shape_mismatches}
    report.missing_keys = [k for k in result.missing_keys if k not in skipped]
    report.unexpected_keys = list(result.unexpected_keys)
    if strict and (report.missing_keys or report.unexpected_keys):
        raise RuntimeError("Checkpoint keys do not match model (strict=True):\n" + report.summary())
    return report


def load_checkpoint(
    path: Any,
    map_location: Any,
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    strict: bool = True,
    skip_shape_mismatch: bool = False,
    weights_only: bool = True,
    model_key: Optional[str] = "model",
) -> Tuple[Dict[str, Any], Optional[LoadReport]]:
    """Load a checkpoint and restore only the objects passed in.

    Args:
        path: Checkpoint file.
        map_location: Where to map tensors (required; no implicit device choice).
        model / optimizer / scheduler / scaler: Objects to restore; None = do not restore.
            The model must already be built with the matching architecture. For optimizers,
            move the model to its final device BEFORE constructing/loading the optimizer.
        strict, skip_shape_mismatch: See ``load_model_state``.
        weights_only: Safe unpickling (default True).
        model_key: Key of the model state dict, or None if the file is a raw state dict.

    Returns:
        ``(checkpoint_dict, load_report_or_None)``. Requested-but-absent states raise KeyError.
    """
    checkpoint = read_checkpoint(path, map_location=map_location, weights_only=weights_only)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint must be a dict, got {type(checkpoint).__name__}")
    report: Optional[LoadReport] = None
    if model is not None:
        state = checkpoint if model_key is None else _get(checkpoint, model_key)
        report = load_model_state(model, state, strict=strict, skip_shape_mismatch=skip_shape_mismatch)
    if optimizer is not None:
        optimizer.load_state_dict(_get(checkpoint, "optimizer"))
    if scheduler is not None:
        scheduler.load_state_dict(_get(checkpoint, "scheduler"))
    if scaler is not None:
        scaler.load_state_dict(_get(checkpoint, "scaler"))
    return dict(checkpoint), report


def _get(checkpoint: Mapping[str, Any], key: str) -> Any:
    if key not in checkpoint:
        raise KeyError(f"Checkpoint has no '{key}' entry; available: {sorted(checkpoint.keys())}")
    return checkpoint[key]
