# Purpose: Exposes an index-based sliding-window view over a sequence without eagerly copying windows, for sequence-model data pipelines.
"""Sliding-window view dataset.

``data`` is any sliceable sequence (Python list/tuple or a tensor whose dim 0 is time).
Window ``i`` is ``data[i * stride : i * stride + window_size]``; it is computed on access,
so tensors yield views and nothing is copied up front.

- ``drop_last=True``: only full windows.
- ``drop_last=False``: one extra, shorter final window is added when full windows do not
  reach the end (no padding here; pad later in a collator, e.g. padded_collator.py).
- An optional ``transform`` is applied to each window on access.

Requires: torch only. No sibling files.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset


class SlidingWindowDataset(Dataset):
    """Lazy windows over ``data`` with explicit size/stride/drop_last semantics."""

    def __init__(
        self,
        data: Union[Sequence[Any], torch.Tensor],
        window_size: int,
        stride: int = 1,
        drop_last: bool = True,
        transform: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        if window_size <= 0 or stride <= 0:
            raise ValueError("window_size and stride must be positive.")
        self.data = data
        self.window_size = window_size
        self.stride = stride
        self.transform = transform

        length = data.shape[0] if isinstance(data, torch.Tensor) else len(data)
        full = 0 if length < window_size else (length - window_size) // stride + 1
        count = full
        if not drop_last:
            covered_end = (full - 1) * stride + window_size if full else 0
            next_start = full * stride
            if covered_end < length and next_start < length:
                count += 1
        self._length = length
        self._num_windows = count

    def __len__(self) -> int:
        return self._num_windows

    def window_bounds(self, index: int) -> Tuple[int, int]:
        """Return ``(start, end)`` (end exclusive, clipped to the data length) of window ``index``."""
        if index < 0:
            index += self._num_windows
        if not 0 <= index < self._num_windows:
            raise IndexError(f"index {index} out of range for {self._num_windows} windows.")
        start = index * self.stride
        return start, min(start + self.window_size, self._length)

    def __getitem__(self, index: int) -> Any:
        start, end = self.window_bounds(index)
        window = self.data[start:end]
        if self.transform is not None:
            window = self.transform(window)
        return window
