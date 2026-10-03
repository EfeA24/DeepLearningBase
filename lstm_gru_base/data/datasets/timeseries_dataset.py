# Purpose: Provides forecasting windows from a numeric multivariate series with explicit feature count, lookback and horizon for LSTM/GRU forecasters.
"""Time-series forecasting dataset.

Contract:
    - ``series``: float tensor ``(N, F)`` (time first, ``F`` input features) or ``(N,)`` treated as ``F = 1``.
    - Window ``i`` starts at ``s = i * stride``:
        ``inputs  = series[s : s + lookback, :]``                      -> ``(lookback, F)``
        ``targets = series[s + lookback + gap : ... + horizon, target_columns]`` -> ``(horizon, len(target_columns))``
      ``gap = 0`` forecasts the steps immediately after the input window.
    - Normalization statistics must be fit on the training span only and applied by the caller.
    - Windows are views into ``series`` (no eager copies).

Requires: torch only.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
from torch.utils.data import Dataset


class TimeSeriesWindowDataset(Dataset):
    """Sliding lookback/horizon windows over a numeric series."""

    def __init__(
        self,
        series: torch.Tensor,
        lookback: int,
        horizon: int,
        stride: int = 1,
        gap: int = 0,
        target_columns: Optional[Sequence[int]] = None,
    ) -> None:
        data = torch.as_tensor(series, dtype=torch.float32) if not isinstance(series, torch.Tensor) else series
        if data.dim() == 1:
            data = data.unsqueeze(-1)
        if data.dim() != 2:
            raise ValueError(f"series must be (N,) or (N, F), got {tuple(data.shape)}.")
        if lookback < 1 or horizon < 1 or stride < 1 or gap < 0:
            raise ValueError("lookback, horizon and stride must be >= 1; gap must be >= 0.")
        self.num_features = data.shape[1]
        cols = list(range(self.num_features)) if target_columns is None else list(target_columns)
        for c in cols:
            if not 0 <= c < self.num_features:
                raise ValueError(f"target column {c} outside [0, {self.num_features}).")
        span = lookback + gap + horizon
        if data.shape[0] < span:
            raise ValueError(f"Series of length {data.shape[0]} is shorter than lookback+gap+horizon={span}.")
        self.series = data
        self.lookback = lookback
        self.horizon = horizon
        self.stride = stride
        self.gap = gap
        self.target_columns = cols
        self._num_windows = (data.shape[0] - span) // stride + 1

    def __len__(self) -> int:
        return self._num_windows

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if index < 0:
            index += self._num_windows
        if not 0 <= index < self._num_windows:
            raise IndexError(f"Index {index} out of range for {self._num_windows} windows.")
        s = index * self.stride
        t0 = s + self.lookback + self.gap
        return {
            "inputs": self.series[s : s + self.lookback],
            "targets": self.series[t0 : t0 + self.horizon, self.target_columns],
        }
