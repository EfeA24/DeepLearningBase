# Purpose: Provides gradient accumulation (exact final-window loss scaling) and an accumulating training epoch for image classification.
"""Gradient accumulation for image classification (semantics aligned with
common_base/training/gradient_accumulation.py).

Loss scaling: each micro-batch mean loss is divided by the number of micro-batches in its
accumulation window. Full windows divide by ``accumulation_steps``; if the total micro-batch count
is known, the final shorter window divides by its real size so every update has a consistent
magnitude. This equals the large-batch gradient only for equal-size micro-batches and per-sample
mean losses. BatchNorm statistics still see only micro-batch sizes (accumulation does not change
that).

Caller-owned loop pattern::

    accum = GradientAccumulator(4, num_micro_batches=len(loader))
    optimizer.zero_grad(set_to_none=True)
    for images, targets in loader:
        loss = loss_fn(model(images), targets)
        accum.scale_loss(loss).backward()
        if accum.step():
            optimizer.step(); optimizer.zero_grad(set_to_none=True)

Requires: torch.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional

import torch
from torch import nn


class GradientAccumulator:
    """Tracks micro-batches and decides when an optimizer update is due."""

    def __init__(self, accumulation_steps: int, num_micro_batches: Optional[int] = None) -> None:
        if not isinstance(accumulation_steps, int) or accumulation_steps < 1:
            raise ValueError(f"accumulation_steps must be an int >= 1, got {accumulation_steps!r}")
        if num_micro_batches is not None and num_micro_batches < 1:
            raise ValueError(f"num_micro_batches must be >= 1 when given, got {num_micro_batches}")
        self.accumulation_steps = accumulation_steps
        self.num_micro_batches = num_micro_batches
        self.reset()

    def reset(self) -> None:
        self._seen = 0
        self._in_window = 0

    def _current_window_size(self) -> int:
        if self.num_micro_batches is None:
            return self.accumulation_steps
        window_start = (self._seen // self.accumulation_steps) * self.accumulation_steps
        remaining = self.num_micro_batches - window_start
        return max(1, min(self.accumulation_steps, remaining))

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        return loss / self._current_window_size()

    def step(self) -> bool:
        """Register one finished backward; return True when ``optimizer.step()`` is due."""
        self._seen += 1
        self._in_window += 1
        window_full = self._in_window >= self.accumulation_steps
        epoch_done = self.num_micro_batches is not None and self._seen >= self.num_micro_batches
        if window_full or epoch_done:
            self._in_window = 0
            return True
        return False

    def should_flush(self) -> bool:
        """True if an unfinished window has pending gradients (only without num_micro_batches)."""
        return self._in_window > 0

    def flush(self) -> None:
        self._in_window = 0


def train_one_epoch_accumulated(
    model: nn.Module,
    loader: Iterable[Any],
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    accumulation_steps: int,
    *,
    max_grad_norm: Optional[float] = None,
    scheduler: Optional[Any] = None,
) -> Dict[str, float]:
    """FP32 training epoch with gradient accumulation over ``(images, int targets)`` batches.

    ``len(loader)`` is used when available for exact final-window scaling; otherwise a pending
    partial window is flushed with an extra update at the end. ``scheduler`` (optional) is stepped
    once per optimizer update. Returns ``{"loss", "accuracy", "num_samples", "num_updates"}``.
    """
    device = torch.device(device)
    try:
        num_micro = len(loader)  # type: ignore[arg-type]
    except TypeError:
        num_micro = None
    accum = GradientAccumulator(accumulation_steps, num_micro_batches=num_micro or None)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum, correct, num_samples, num_updates = 0.0, 0, 0, 0

    def apply_update() -> None:
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()

    for batch in loader:
        images, targets = batch[0].to(device), batch[1].to(device)
        logits = model(images)
        loss = loss_fn(logits, targets)
        accum.scale_loss(loss).backward()
        if accum.step():
            apply_update()
            num_updates += 1
        batch_size = images.shape[0]
        loss_sum += float(loss.detach()) * batch_size
        num_samples += batch_size
        if targets.dim() == 1 and not targets.is_floating_point():
            correct += int((logits.detach().argmax(dim=1) == targets).sum())

    if accum.should_flush():
        apply_update()
        accum.flush()
        num_updates += 1
    if num_samples == 0:
        raise ValueError("loader produced no batches")
    return {
        "loss": loss_sum / num_samples,
        "accuracy": correct / num_samples,
        "num_samples": num_samples,
        "num_updates": num_updates,
    }
