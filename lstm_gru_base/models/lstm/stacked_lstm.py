# Purpose: Builds stacked LSTMs with explicit per-layer sizes and between-layer-only dropout (optional residuals) for deep LSTM sequence models.
"""Stacked LSTM constructors.

Two options:
    - ``build_stacked_lstm``: returns a plain ``nn.LSTM(num_layers=L, dropout=p, batch_first=True)``.
      PyTorch applies dropout to the outputs of layers ``0..L-2`` only (never after the last layer).
    - ``StackedLSTM``: a ModuleList of single-layer nn.LSTMs, allowing different hidden sizes per layer
      and optional residual connections (only where input/output widths match). Same dropout semantics.

Shapes: ``x`` ``(B, T, I)`` -> outputs ``(B, T, H_last)``. ``StackedLSTM`` states are lists of per-layer
``(h, c)`` tuples, each ``(1, B, H_k)``, because widths may differ across layers.
``lengths`` ``(B,)`` (right padding) triggers packing per layer.

Requires: torch only.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

LSTMState = Tuple[torch.Tensor, torch.Tensor]


def build_stacked_lstm(
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float = 0.0,
    bidirectional: bool = False,
) -> nn.LSTM:
    """Construct a batch-first multi-layer nn.LSTM with validated dropout semantics."""
    if num_layers < 2:
        raise ValueError("A stacked LSTM needs num_layers >= 2.")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1).")
    return nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=dropout,
                   bidirectional=bidirectional, batch_first=True)


class StackedLSTM(nn.Module):
    """Explicit layer-by-layer LSTM stack with inter-layer dropout and optional residuals."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        dropout: float = 0.0,
        residual: bool = False,
    ) -> None:
        super().__init__()
        if len(hidden_sizes) < 1:
            raise ValueError("hidden_sizes must contain at least one layer size.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        sizes = [input_size, *hidden_sizes]
        self.layers = nn.ModuleList(nn.LSTM(sizes[k], sizes[k + 1], batch_first=True) for k in range(len(hidden_sizes)))
        self.residual_flags = [residual and sizes[k] == sizes[k + 1] for k in range(len(hidden_sizes))]
        self.dropout = nn.Dropout(dropout)
        self.output_size = sizes[-1]

    def forward(
        self,
        x: torch.Tensor,
        states: Optional[List[Optional[LSTMState]]] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[LSTMState]]:
        if states is not None and len(states) != len(self.layers):
            raise ValueError(f"Expected {len(self.layers)} per-layer states, got {len(states)}.")
        cpu_lengths = None if lengths is None else lengths.detach().to("cpu", torch.long)
        out = x
        new_states: List[LSTMState] = []
        for k, layer in enumerate(self.layers):
            layer_in = out
            init = None if states is None else states[k]
            if cpu_lengths is None:
                out, st = layer(layer_in, init)
            else:
                packed = pack_padded_sequence(layer_in, cpu_lengths, batch_first=True, enforce_sorted=False)
                packed_out, st = layer(packed, init)
                out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.shape[1])
            if self.residual_flags[k]:
                out = out + layer_in
            if k < len(self.layers) - 1:
                out = self.dropout(out)
            new_states.append(st)
        return out, new_states
