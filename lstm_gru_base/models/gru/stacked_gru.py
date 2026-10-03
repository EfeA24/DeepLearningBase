# Purpose: Builds stacked GRUs with explicit per-layer sizes and between-layer-only dropout (optional residuals) for deep GRU sequence models.
"""Stacked GRU constructors.

Two options:
    - ``build_stacked_gru``: plain ``nn.GRU(num_layers=L, dropout=p, batch_first=True)``; dropout is applied
      to outputs of layers ``0..L-2`` only.
    - ``StackedGRU``: ModuleList of single-layer nn.GRUs with per-layer hidden sizes and optional residual
      connections where widths match. Same dropout semantics.

Shapes: ``x`` ``(B, T, I)`` -> outputs ``(B, T, H_last)``; ``StackedGRU`` states are lists of per-layer
tensors ``(1, B, H_k)``. ``lengths`` ``(B,)`` (right padding) triggers packing per layer.

Requires: torch only.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def build_stacked_gru(
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float = 0.0,
    bidirectional: bool = False,
) -> nn.GRU:
    """Construct a batch-first multi-layer nn.GRU with validated dropout semantics."""
    if num_layers < 2:
        raise ValueError("A stacked GRU needs num_layers >= 2.")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1).")
    return nn.GRU(input_size, hidden_size, num_layers=num_layers, dropout=dropout,
                  bidirectional=bidirectional, batch_first=True)


class StackedGRU(nn.Module):
    """Explicit layer-by-layer GRU stack with inter-layer dropout and optional residuals."""

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
        self.layers = nn.ModuleList(nn.GRU(sizes[k], sizes[k + 1], batch_first=True) for k in range(len(hidden_sizes)))
        self.residual_flags = [residual and sizes[k] == sizes[k + 1] for k in range(len(hidden_sizes))]
        self.dropout = nn.Dropout(dropout)
        self.output_size = sizes[-1]

    def forward(
        self,
        x: torch.Tensor,
        states: Optional[List[Optional[torch.Tensor]]] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if states is not None and len(states) != len(self.layers):
            raise ValueError(f"Expected {len(self.layers)} per-layer states, got {len(states)}.")
        cpu_lengths = None if lengths is None else lengths.detach().to("cpu", torch.long)
        out = x
        new_states: List[torch.Tensor] = []
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
