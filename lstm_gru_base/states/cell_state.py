# Purpose: Validates and manipulates LSTM cell states and (h, c) state tuples in PyTorch's (layers*directions, B, H) layout.
"""LSTM cell-state and (h, c) tuple helpers (self-contained; mirrors states/hidden_state.py for ``c``).

Layout: ``h`` and ``c`` are each ``(num_layers * num_directions, B, H)``; batch is dim 1. nn.LSTM with
``proj_size > 0`` makes ``h`` ``(L*D, B, proj_size)`` while ``c`` stays ``(L*D, B, H)``; pass
``proj_size`` to ``validate_lstm_state`` in that case.

Requires: torch only.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch

LSTMState = Tuple[torch.Tensor, torch.Tensor]


def zeros_cell(
    num_layers: int,
    batch_size: int,
    hidden_size: int,
    num_directions: int = 1,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Zero cell state ``(L*D, B, H)``."""
    if num_directions not in (1, 2):
        raise ValueError("num_directions must be 1 or 2.")
    return torch.zeros(num_layers * num_directions, batch_size, hidden_size, device=device, dtype=dtype)


def zeros_lstm_state(
    num_layers: int,
    batch_size: int,
    hidden_size: int,
    num_directions: int = 1,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
    proj_size: int = 0,
) -> LSTMState:
    """Zero ``(h_0, c_0)`` tuple for nn.LSTM."""
    c = zeros_cell(num_layers, batch_size, hidden_size, num_directions, device, dtype)
    h_size = proj_size if proj_size > 0 else hidden_size
    h = torch.zeros(num_layers * num_directions, batch_size, h_size, device=device, dtype=dtype)
    return h, c


def validate_lstm_state(
    state: LSTMState,
    num_layers: int,
    batch_size: int,
    hidden_size: int,
    num_directions: int = 1,
    proj_size: int = 0,
) -> None:
    """Raise if ``state`` is not a 2-tuple of correctly shaped tensors."""
    if not (isinstance(state, (tuple, list)) and len(state) == 2):
        raise TypeError("LSTM state must be a (h, c) tuple.")
    h, c = state
    ld = num_layers * num_directions
    h_expected = (ld, batch_size, proj_size if proj_size > 0 else hidden_size)
    c_expected = (ld, batch_size, hidden_size)
    if tuple(h.shape) != h_expected:
        raise ValueError(f"h shape {tuple(h.shape)} != expected {h_expected}.")
    if tuple(c.shape) != c_expected:
        raise ValueError(f"c shape {tuple(c.shape)} != expected {c_expected}.")


def detach_lstm_state(state: LSTMState) -> LSTMState:
    """Detach both tensors (TBPTT boundary)."""
    h, c = state
    return h.detach(), c.detach()


def select_lstm_batch(state: LSTMState, indices: Union[torch.Tensor, Sequence[int]]) -> LSTMState:
    """Select/reorder batch rows (dim 1) of both ``h`` and ``c``."""
    h, c = state
    index = torch.as_tensor(indices, dtype=torch.long, device=h.device)
    return h.index_select(1, index), c.index_select(1, index)


def split_lstm_state(state: LSTMState) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(h, c)`` after validating tuple structure."""
    if not (isinstance(state, (tuple, list)) and len(state) == 2):
        raise TypeError("LSTM state must be a (h, c) tuple.")
    return state[0], state[1]


def merge_lstm_state(h: torch.Tensor, c: torch.Tensor) -> LSTMState:
    """Build an ``(h, c)`` tuple, checking layer/batch dims agree."""
    if h.shape[:2] != c.shape[:2]:
        raise ValueError(f"h {tuple(h.shape)} and c {tuple(c.shape)} disagree on (L*D, B).")
    return h, c
