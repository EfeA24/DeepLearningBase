# Purpose: Provides a generic windowed input/target sequence dataset with a configurable target contract for recurrent models.
"""Generic windowed sequence dataset.

Input: one long time-major series ``inputs`` of shape (T, *F) (e.g. (T,) token ids or
(T, F) features) and an optional time-aligned ``targets`` series (T, *G). Windows of
``window_size`` start at 0, stride, 2*stride, ... and are created lazily as views.

Target contract (``target_mode``), with ``src = targets if given else inputs``:
- "next_step":  x = inputs[s:s+W],  y = src[s+1:s+W+1]   (y has length W; next-step prediction)
- "next_value": x = inputs[s:s+W],  y = src[s+W]         (single value after the window)
- "aligned":    x = inputs[s:s+W],  y = targets[s:s+W]   (per-step labels; ``targets`` required)

Only complete windows are produced. Items are returned as ``(x, y)`` tuples (batch-first
after default collation: x (B, W, *F)). No padding, so no mask is needed.

Requires: torch only. No sibling files.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
from torch.utils.data import Dataset

_TARGET_MODES = ("next_step", "next_value", "aligned")


class SequenceDataset(Dataset):
    """Lazy windowed (input, target) pairs over a time-major series."""

    def __init__(
        self,
        inputs: torch.Tensor,
        window_size: int,
        stride: int = 1,
        target_mode: str = "next_step",
        targets: Optional[torch.Tensor] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        if not isinstance(inputs, torch.Tensor) or inputs.dim() == 0:
            raise TypeError("inputs must be a tensor of shape (T, *F).")
        if window_size <= 0 or stride <= 0:
            raise ValueError("window_size and stride must be positive.")
        if target_mode not in _TARGET_MODES:
            raise ValueError(f"target_mode must be one of {_TARGET_MODES}, got {target_mode!r}.")
        if target_mode == "aligned" and targets is None:
            raise ValueError("target_mode='aligned' requires a targets series.")
        if targets is not None and targets.shape[0] != inputs.shape[0]:
            raise ValueError("targets must have the same length (dim 0) as inputs.")

        self.inputs = inputs
        self.targets = targets
        self.window_size = window_size
        self.stride = stride
        self.target_mode = target_mode
        self.transform = transform
        self.target_transform = target_transform

        span = window_size if target_mode == "aligned" else window_size + 1
        length = inputs.shape[0]
        self._num_windows = 0 if length < span else (length - span) // stride + 1

    def __len__(self) -> int:
        return self._num_windows

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if index < 0:
            index += self._num_windows
        if not 0 <= index < self._num_windows:
            raise IndexError(f"index {index} out of range for {self._num_windows} windows.")
        start = index * self.stride
        end = start + self.window_size
        source = self.targets if self.targets is not None else self.inputs

        x = self.inputs[start:end]
        if self.target_mode == "next_step":
            y = source[start + 1 : end + 1]
        elif self.target_mode == "next_value":
            y = source[end]
        else:
            y = source[start:end]

        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y
