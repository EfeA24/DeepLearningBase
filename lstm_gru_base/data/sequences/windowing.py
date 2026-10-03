# Purpose: Generates input/target windows with explicit size, stride and horizon controls for LSTM/GRU sequence and forecasting projects.
"""Window generation for sequence models.

Contract:
    - ``sequence`` is a tensor with time on dim 0: ``(N,)`` or ``(N, F)``.
    - A window starting at ``s`` yields ``inputs = seq[s : s + window]`` and, when ``horizon > 0``,
      ``targets = seq[s + window + offset : s + window + offset + horizon]``.
      ``offset = 0`` means targets start right after the input window.
    - Starts are ``0, stride, 2*stride, ...`` while a full window (+ targets) fits. ``drop_last=False``
      additionally yields a final window aligned to the end of the sequence if the stride skipped it.
    - Windows are views (no copies) until stacked.

Requires: torch only.
"""
from __future__ import annotations

from typing import Iterator, List, Optional, Tuple

import torch


def window_starts(
    num_steps: int,
    window: int,
    stride: int = 1,
    horizon: int = 0,
    offset: int = 0,
    drop_last: bool = True,
) -> List[int]:
    """Return valid window start indices for a sequence of ``num_steps`` steps."""
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}.")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}.")
    if horizon < 0 or offset < 0:
        raise ValueError("horizon and offset must be >= 0.")
    span = window + (offset + horizon if horizon > 0 else 0)
    last_start = num_steps - span
    if last_start < 0:
        return []
    starts = list(range(0, last_start + 1, stride))
    if not drop_last and starts[-1] != last_start:
        starts.append(last_start)
    return starts


def iter_windows(
    sequence: torch.Tensor,
    window: int,
    stride: int = 1,
    horizon: int = 0,
    offset: int = 0,
    drop_last: bool = True,
) -> Iterator[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """Yield ``(inputs, targets)`` view pairs; ``targets`` is None when ``horizon == 0``."""
    for s in window_starts(sequence.shape[0], window, stride, horizon, offset, drop_last):
        inputs = sequence[s : s + window]
        if horizon > 0:
            t0 = s + window + offset
            yield inputs, sequence[t0 : t0 + horizon]
        else:
            yield inputs, None


def make_windows(
    sequence: torch.Tensor,
    window: int,
    stride: int = 1,
    horizon: int = 0,
    offset: int = 0,
    drop_last: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Stack all windows: inputs ``(W, window, ...)`` and targets ``(W, horizon, ...)`` or None.

    Raises ``ValueError`` if no window fits.
    """
    pairs = list(iter_windows(sequence, window, stride, horizon, offset, drop_last))
    if not pairs:
        raise ValueError(f"Sequence of length {sequence.shape[0]} is too short for window={window}, horizon={horizon}.")
    inputs = torch.stack([p[0] for p in pairs])
    targets = torch.stack([p[1] for p in pairs]) if horizon > 0 else None
    return inputs, targets


def next_step_windows(sequence: torch.Tensor, window: int, stride: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Language-model style windows: inputs ``seq[s:s+window]``, targets ``seq[s+1:s+window+1]``."""
    starts = window_starts(sequence.shape[0], window + 1, stride)
    if not starts:
        raise ValueError(f"Sequence of length {sequence.shape[0]} is too short for window={window} + 1 shift.")
    inputs = torch.stack([sequence[s : s + window] for s in starts])
    targets = torch.stack([sequence[s + 1 : s + window + 1] for s in starts])
    return inputs, targets
