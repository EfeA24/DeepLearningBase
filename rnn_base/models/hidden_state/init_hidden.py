# Purpose: Creates zero-initialized recurrent hidden states (and optional LSTM cell states) with explicit device and dtype.
"""Hidden-state initialization.

Layout (PyTorch convention, independent of batch_first):
- h: (num_layers * num_directions, B, hidden_size)
- LSTM: tuple (h, c) where c has the same shape as h (with ``proj_size`` > 0, h uses proj_size).
Device and dtype are explicit arguments; nothing is guessed from global state.

Requires: torch only. No sibling files.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from torch import nn

HiddenState = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


def init_hidden(
    batch_size: int,
    hidden_size: int,
    num_layers: int = 1,
    num_directions: int = 1,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
    with_cell_state: bool = False,
) -> HiddenState:
    """Zeros of shape (num_layers * num_directions, batch_size, hidden_size); (h, c) if ``with_cell_state``."""
    if batch_size <= 0 or hidden_size <= 0 or num_layers <= 0:
        raise ValueError("batch_size, hidden_size and num_layers must be positive.")
    if num_directions not in (1, 2):
        raise ValueError(f"num_directions must be 1 or 2, got {num_directions}.")
    shape = (num_layers * num_directions, batch_size, hidden_size)
    h = torch.zeros(shape, device=device, dtype=dtype)
    if with_cell_state:
        return h, torch.zeros(shape, device=device, dtype=dtype)
    return h


def init_hidden_for(
    rnn: nn.RNNBase,
    batch_size: int,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
) -> HiddenState:
    """Zero state matching a ``nn.RNN`` / ``nn.GRU`` / ``nn.LSTM`` configuration (tuple for LSTM)."""
    if not isinstance(rnn, nn.RNNBase):
        raise TypeError(f"rnn must be an nn.RNNBase instance, got {type(rnn).__name__}.")
    num_directions = 2 if rnn.bidirectional else 1
    if isinstance(rnn, nn.LSTM):
        proj_size = getattr(rnn, "proj_size", 0)
        h_size = proj_size if proj_size > 0 else rnn.hidden_size
        h = init_hidden(batch_size, h_size, rnn.num_layers, num_directions, device, dtype)
        c = init_hidden(batch_size, rnn.hidden_size, rnn.num_layers, num_directions, device, dtype)
        return h, c  # type: ignore[return-value]
    return init_hidden(batch_size, rnn.hidden_size, rnn.num_layers, num_directions, device, dtype)
