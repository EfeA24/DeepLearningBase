# Purpose: Builds forecasting windows (input history -> future horizon) from numeric time series for recurrent forecasting models.
"""Forecasting window dataset over a numeric series.

Series shape: (T,) univariate or (T, F) multivariate, time-major, float.
Window ``i`` (start ``s = i * stride``):
- ``x = series[s : s + input_length]``                                   -> (L,) or (L, F)
- ``y = series[s + L + gap : s + L + gap + horizon][:, target_columns]`` -> (H,) or (H, F_t)
  (``target_columns`` only applies to 2-D series; default keeps all columns)
- ``gap`` skips steps between history and forecast (0 = forecast starts right after history).

Only complete windows are produced; batch-first after default collation: x (B, L, F).
No scaling is applied: fit normalization statistics on the training split only and pass
already-normalized series (or apply them via ``transform``).

Requires: torch only. No sibling files.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


class TimeSeriesForecastDataset(Dataset):
    """Lazy (history, future) windows with explicit input length, horizon, gap and stride."""

    def __init__(
        self,
        series: torch.Tensor,
        input_length: int,
        horizon: int,
        stride: int = 1,
        gap: int = 0,
        target_columns: Optional[Sequence[int]] = None,
        transform: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> None:
        if not isinstance(series, torch.Tensor) or series.dim() not in (1, 2):
            raise ValueError("series must be a tensor of shape (T,) or (T, F).")
        if input_length <= 0 or horizon <= 0 or stride <= 0:
            raise ValueError("input_length, horizon and stride must be positive.")
        if gap < 0:
            raise ValueError(f"gap must be non-negative, got {gap}.")
        if target_columns is not None:
            if series.dim() != 2:
                raise ValueError("target_columns requires a 2-D (T, F) series.")
            num_features = series.shape[1]
            for column in target_columns:
                if not -num_features <= column < num_features:
                    raise ValueError(f"target column {column} out of range for {num_features} features.")

        self.series = series
        self.input_length = input_length
        self.horizon = horizon
        self.stride = stride
        self.gap = gap
        self.target_columns = None if target_columns is None else list(target_columns)
        self.transform = transform

        span = input_length + gap + horizon
        length = series.shape[0]
        self._num_windows = 0 if length < span else (length - span) // stride + 1

    def __len__(self) -> int:
        return self._num_windows

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if index < 0:
            index += self._num_windows
        if not 0 <= index < self._num_windows:
            raise IndexError(f"index {index} out of range for {self._num_windows} windows.")
        start = index * self.stride
        target_start = start + self.input_length + self.gap
        x = self.series[start : start + self.input_length]
        y = self.series[target_start : target_start + self.horizon]
        if self.target_columns is not None:
            y = y[:, self.target_columns]
        if self.transform is not None:
            x, y = self.transform(x, y)
        return x, y
