# Purpose: Manages latest/best checkpoint paths and retention with constructor-supplied metric direction and file naming for PyTorch training runs.
"""Checkpoint manager with an explicit path policy.

Policy (all from constructor arguments):
    - Every ``save(...)`` writes ``{prefix}_epoch{epoch:04d}{suffix}`` in ``directory`` and updates
      ``{prefix}_latest{suffix}`` (a copy) when ``keep_latest_copy`` is True.
    - If ``metric`` improves according to ``mode`` ("min"/"max") and ``min_delta``,
      ``{prefix}_best{suffix}`` is overwritten.
    - Only the newest ``keep_last`` epoch files are retained (None = keep all). Latest/best
      copies are never deleted by retention.
    - Only files created by THIS manager instance are deleted; pre-existing files are untouched.

State dicts are written with ``torch.save`` (atomic rename). In multi-process training, call
``save`` from the main process only. Use ``state_dict``/``load_state_dict`` to persist the
manager's best-metric bookkeeping across resumes.

Requires: torch.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Dict, List, Optional

import torch
from torch import nn


def _unwrap(model: nn.Module) -> nn.Module:
    while True:
        if hasattr(model, "_orig_mod") and isinstance(model._orig_mod, nn.Module):
            model = model._orig_mod
        elif isinstance(model, (nn.parallel.DistributedDataParallel, nn.DataParallel)):
            model = model.module
        else:
            return model


class CheckpointManager:
    """Writes epoch/latest/best checkpoints under an explicit directory.

    Args:
        directory: Target directory (str or PathLike).
        mode: "min" or "max" for the monitored metric.
        prefix: File name prefix.
        suffix: File extension, e.g. ".pt".
        keep_last: Number of epoch checkpoints to keep (>= 1) or None for all.
        min_delta: Required improvement to count as a new best (>= 0).
        keep_latest_copy: Also maintain a ``*_latest`` file.
        create_dirs: Create ``directory`` on first save if missing.
    """

    def __init__(
        self,
        directory: Any,
        mode: str,
        prefix: str = "checkpoint",
        suffix: str = ".pt",
        keep_last: Optional[int] = 3,
        min_delta: float = 0.0,
        keep_latest_copy: bool = True,
        create_dirs: bool = False,
    ) -> None:
        from pathlib import Path

        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        if keep_last is not None and keep_last < 1:
            raise ValueError(f"keep_last must be >= 1 or None, got {keep_last}")
        if min_delta < 0:
            raise ValueError(f"min_delta must be >= 0, got {min_delta}")
        self.directory = Path(directory)
        self.mode = mode
        self.prefix = prefix
        self.suffix = suffix
        self.keep_last = keep_last
        self.min_delta = min_delta
        self.keep_latest_copy = keep_latest_copy
        self.create_dirs = create_dirs
        self.best_metric: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self._saved: List[str] = []

    @property
    def latest_path(self):
        return self.directory / f"{self.prefix}_latest{self.suffix}"

    @property
    def best_path(self):
        return self.directory / f"{self.prefix}_best{self.suffix}"

    def epoch_path(self, epoch: int):
        return self.directory / f"{self.prefix}_epoch{epoch:04d}{self.suffix}"

    def is_improvement(self, metric: float) -> bool:
        if math.isnan(metric):
            return False
        if self.best_metric is None:
            return True
        if self.mode == "min":
            return metric < self.best_metric - self.min_delta
        return metric > self.best_metric + self.min_delta

    def save(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        metric: Optional[float] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Save checkpoints for ``epoch``; returns ``{"path", "is_best", "deleted"}``."""
        self._ensure_directory()
        checkpoint: Dict[str, Any] = {
            "format_version": 1,
            "model": _unwrap(model).state_dict(),
            "metadata": {**dict(metadata or {}), "epoch": epoch, "metric": metric},
        }
        if optimizer is not None:
            checkpoint["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint["scheduler"] = scheduler.state_dict()
        if scaler is not None:
            checkpoint["scaler"] = scaler.state_dict()

        path = self.epoch_path(epoch)
        self._atomic_save(checkpoint, path)
        if str(path) in self._saved:
            self._saved.remove(str(path))
        self._saved.append(str(path))
        if self.keep_latest_copy:
            self._atomic_save(checkpoint, self.latest_path)

        is_best = metric is not None and self.is_improvement(float(metric))
        if is_best:
            self.best_metric = float(metric)
            self.best_epoch = epoch
            self._atomic_save(checkpoint, self.best_path)
        return {"path": str(path), "is_best": is_best, "deleted": self._apply_retention()}

    def state_dict(self) -> Dict[str, Any]:
        return {"best_metric": self.best_metric, "best_epoch": self.best_epoch, "saved": list(self._saved)}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.best_metric = state.get("best_metric")
        self.best_epoch = state.get("best_epoch")
        self._saved = list(state.get("saved", []))

    def _ensure_directory(self) -> None:
        if self.directory.exists():
            return
        if not self.create_dirs:
            raise FileNotFoundError(f"Checkpoint directory does not exist: {self.directory} (pass create_dirs=True)")
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _atomic_save(obj: Dict[str, Any], path) -> None:
        import os

        tmp = path.with_name(path.name + ".tmp")
        torch.save(obj, tmp)
        os.replace(tmp, path)

    def _apply_retention(self) -> List[str]:
        from pathlib import Path

        if self.keep_last is None:
            return []
        deleted: List[str] = []
        while len(self._saved) > self.keep_last:
            old = Path(self._saved.pop(0))
            if old.exists():
                old.unlink()
                deleted.append(str(old))
        return deleted
