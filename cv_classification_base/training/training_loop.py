# Purpose: Orchestrates image-classification training epochs around caller-supplied train/validation callables with injectable stopping and checkpoint policy.
"""Small epoch orchestrator (optional; a plain for-loop is equally valid).

The loop owns only the epoch counter and the history. Everything else is injected:
    train_fn(epoch) -> dict              e.g. ``lambda e: train_one_epoch(model, loader, ...)``
    val_fn(epoch) -> dict | None         optional validation
    epoch_end_callbacks: [(epoch, record) -> None]   e.g. per-epoch scheduler.step, logging,
                                                     checkpoint saving (BestModelTracker)
    should_stop(epoch, record) -> bool   e.g. ``lambda e, r: early_stopper.step(r["val_loss"])``

Record format: train metrics prefixed with ``train_`` and validation metrics with ``val_``, plus
``"epoch"``. Epochs are 0-based; ``start_epoch`` supports resuming.

Requires: Python standard library only; torch is used inside the supplied callables.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

EpochFn = Callable[[int], Mapping[str, Any]]
Callback = Callable[[int, Dict[str, Any]], None]
StopFn = Callable[[int, Dict[str, Any]], bool]


@dataclass
class FitResult:
    history: List[Dict[str, Any]] = field(default_factory=list)
    stopped_early: bool = False
    last_epoch: int = -1


def fit(
    num_epochs: int,
    train_fn: EpochFn,
    val_fn: Optional[EpochFn] = None,
    epoch_end_callbacks: Sequence[Callback] = (),
    should_stop: Optional[StopFn] = None,
    start_epoch: int = 0,
) -> FitResult:
    """Run epochs ``start_epoch .. num_epochs - 1``; callbacks run before the stop check."""
    if num_epochs < 0 or start_epoch < 0:
        raise ValueError("num_epochs and start_epoch must be >= 0")
    result = FitResult()
    for epoch in range(start_epoch, num_epochs):
        record: Dict[str, Any] = {"epoch": epoch}
        record.update({f"train_{k}": v for k, v in dict(train_fn(epoch)).items()})
        if val_fn is not None:
            record.update({f"val_{k}": v for k, v in dict(val_fn(epoch)).items()})
        result.history.append(record)
        result.last_epoch = epoch
        for callback in epoch_end_callbacks:
            callback(epoch, record)
        if should_stop is not None and should_stop(epoch, record):
            result.stopped_early = True
            break
    return result
