# Purpose: Provides a gradient-accumulation helper that scales micro-batch losses and handles final partial accumulation windows for any PyTorch training loop.
"""Gradient accumulation helper.

Pattern (caller-owned loop)::

    accum = GradientAccumulator(accumulation_steps=4, num_micro_batches=len(loader))
    optimizer.zero_grad(set_to_none=True)
    for batch in loader:
        loss = loss_fn(model(x), y)              # mean loss of this micro-batch
        (accum.scale_loss(loss)).backward()      # divided by the size of the current window
        if accum.step():                         # True when an optimizer update is due
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

Loss scaling:
    Each micro-batch loss is divided by the number of micro-batches in its accumulation
    window. Full windows use ``accumulation_steps``; if ``num_micro_batches`` is given and the
    final window is shorter, that window divides by its real size so the update magnitude
    stays consistent. Without ``num_micro_batches`` the last partial window is divided by
    ``accumulation_steps`` (slightly smaller gradient); call ``should_flush()`` after the loop
    to decide whether to apply a final update.

    This equals the large-batch gradient only when micro-batches have equal size and the
    loss is a per-sample mean. For token-weighted losses, sum losses and divide by the total
    token count of the window instead.

Requires: torch.
"""
from __future__ import annotations

from typing import Optional

import torch


class GradientAccumulator:
    """Tracks micro-batches and decides when to run ``optimizer.step()``.

    Args:
        accumulation_steps: Micro-batches per optimizer update (>= 1).
        num_micro_batches: Optional total micro-batches per epoch; enables exact scaling
            of the final partial window.
    """

    def __init__(self, accumulation_steps: int, num_micro_batches: Optional[int] = None) -> None:
        if not isinstance(accumulation_steps, int) or accumulation_steps < 1:
            raise ValueError(f"accumulation_steps must be an int >= 1, got {accumulation_steps!r}")
        if num_micro_batches is not None and num_micro_batches < 1:
            raise ValueError(f"num_micro_batches must be >= 1 when given, got {num_micro_batches}")
        self.accumulation_steps = accumulation_steps
        self.num_micro_batches = num_micro_batches
        self.reset()

    def reset(self) -> None:
        """Reset counters at the start of an epoch."""
        self._seen = 0
        self._in_window = 0

    def _current_window_size(self) -> int:
        if self.num_micro_batches is None:
            return self.accumulation_steps
        window_start = (self._seen // self.accumulation_steps) * self.accumulation_steps
        remaining = self.num_micro_batches - window_start
        return max(1, min(self.accumulation_steps, remaining))

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Divide a micro-batch loss by the size of its accumulation window."""
        return loss / self._current_window_size()

    def step(self) -> bool:
        """Register one finished backward pass; return True when an optimizer update is due."""
        self._seen += 1
        self._in_window += 1
        window_full = self._in_window >= self.accumulation_steps
        epoch_done = self.num_micro_batches is not None and self._seen >= self.num_micro_batches
        if window_full or epoch_done:
            self._in_window = 0
            return True
        return False

    def should_flush(self) -> bool:
        """True if gradients from an unfinished window are pending (call after the loop)."""
        return self._in_window > 0

    def flush(self) -> None:
        """Mark pending gradients as consumed after the caller applied a final update."""
        self._in_window = 0
