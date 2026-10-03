# Purpose: Provides an optional epoch orchestration wrapper for object-detection training with injected train/validation, callbacks, checkpoint and stopping policies.
"""Minimal detection training orchestration.

Everything is injected; this loop owns only the epoch counter and history:
    train_fn(epoch) -> dict          e.g. lambda e: train_one_epoch(model, loader, ...)
    val_fn(epoch) -> dict | None     optional, run every ``val_every`` epochs
    callbacks: [fn(epoch, metrics)]  logging, per-epoch schedulers, ...
    checkpoint_fn(epoch, metrics)    caller decides what/where to save (or None)
    should_stop(epoch, metrics) -> bool   e.g. an early-stopping object's __call__

Metrics passed to callbacks are merged as {"train/<k>": ..., "val/<k>": ..., "epoch": e}.
No logging backend, checkpoint format or stopping rule is assumed.

Requires: standard library only (works with any torch-based train/val callables).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

EpochFn = Callable[[int], Mapping[str, Any]]
EpochCallback = Callable[[int, Dict[str, Any]], None]


def run_detection_training(
    num_epochs: int,
    train_fn: EpochFn,
    val_fn: Optional[EpochFn] = None,
    callbacks: Sequence[EpochCallback] = (),
    checkpoint_fn: Optional[EpochCallback] = None,
    should_stop: Optional[Callable[[int, Dict[str, Any]], bool]] = None,
    start_epoch: int = 0,
    val_every: int = 1,
) -> List[Dict[str, Any]]:
    """Run epochs ``start_epoch .. num_epochs - 1`` and return the per-epoch history.

    Order per epoch: train_fn -> (val_fn) -> callbacks -> checkpoint_fn -> should_stop.
    ``start_epoch`` supports resuming (the caller restores model/optimizer state).
    """
    if num_epochs <= 0 or not 0 <= start_epoch <= num_epochs:
        raise ValueError("require num_epochs > 0 and 0 <= start_epoch <= num_epochs")
    if val_every <= 0:
        raise ValueError("val_every must be > 0")
    history: List[Dict[str, Any]] = []
    for epoch in range(start_epoch, num_epochs):
        metrics: Dict[str, Any] = {"epoch": epoch}
        metrics.update({f"train/{k}": v for k, v in train_fn(epoch).items()})
        if val_fn is not None and ((epoch + 1) % val_every == 0 or epoch == num_epochs - 1):
            val_metrics = val_fn(epoch)
            if val_metrics:
                metrics.update({f"val/{k}": v for k, v in val_metrics.items()})
        for callback in callbacks:
            callback(epoch, metrics)
        if checkpoint_fn is not None:
            checkpoint_fn(epoch, metrics)
        history.append(metrics)
        if should_stop is not None and should_stop(epoch, metrics):
            break
    return history


class MetricEarlyStopping:
    """Optional ``should_stop`` helper: stop when a metric has not improved for ``patience`` epochs.

    Args:
        key: Metric key in the merged dict (e.g. "val/map").
        mode: "max" or "min".
        patience: Epochs without improvement before stopping.
        min_delta: Minimum change counted as improvement.
    """

    def __init__(self, key: str, mode: str = "max", patience: int = 10, min_delta: float = 0.0) -> None:
        if mode not in ("max", "min"):
            raise ValueError("mode must be 'max' or 'min'")
        if patience < 0 or min_delta < 0:
            raise ValueError("patience and min_delta must be >= 0")
        self.key, self.mode, self.patience, self.min_delta = key, mode, patience, min_delta
        self.best: Optional[float] = None
        self.bad_epochs = 0

    def __call__(self, epoch: int, metrics: Dict[str, Any]) -> bool:
        if self.key not in metrics:
            return False
        value = float(metrics[self.key])
        if value != value:
            self.bad_epochs += 1
        elif self.best is None or (
            value > self.best + self.min_delta if self.mode == "max" else value < self.best - self.min_delta
        ):
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs > self.patience
