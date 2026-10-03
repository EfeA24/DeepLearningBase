# Purpose: Wraps torch.nn.RNN with explicit batch_first, layers, dropout and bidirectional options plus length-aware packing.
"""Thin ``torch.nn.RNN`` wrapper (production path; see rnn_scratch.py for the educational version).

Shapes (batch_first=True default):
- x: (B, T, input_size); h0: (num_layers * D, B, hidden_size) or None (zeros), D = 2 if bidirectional
- output: (B, T, D * hidden_size); h_n: (num_layers * D, B, hidden_size)
Optional ``lengths`` (B,) for right-padded batches: inputs are packed with
``enforce_sorted=False`` so padded steps are skipped, ``h_n`` holds each sequence's last valid
state (in original batch order) and padded output positions are zeros (output length stays T).
``dropout`` applies between stacked layers only, so it requires num_layers > 1.

Requires: torch only. No sibling files.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class PyTorchRNN(nn.Module):
    """Explicitly configured ``nn.RNN`` with optional length-aware packing."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = "tanh",
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")
        if dropout > 0.0 and num_layers == 1:
            raise ValueError("dropout is applied between layers only; it needs num_layers > 1.")
        self.batch_first = batch_first
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    @property
    def num_directions(self) -> int:
        return 2 if self.rnn.bidirectional else 1

    @property
    def output_size(self) -> int:
        return self.rnn.hidden_size * self.num_directions

    def forward(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(f"x must be 3-D, got shape {tuple(x.shape)}.")
        if lengths is None:
            return self.rnn(x, h0)
        total_length = x.shape[1] if self.batch_first else x.shape[0]
        cpu_lengths = lengths.detach().to(device="cpu", dtype=torch.int64)
        if cpu_lengths.numel() and int(cpu_lengths.min()) < 1:
            raise ValueError("All lengths must be >= 1 when packing.")
        packed = pack_padded_sequence(x, cpu_lengths, batch_first=self.batch_first, enforce_sorted=False)
        packed_output, h_n = self.rnn(packed, h0)
        output, _ = pad_packed_sequence(packed_output, batch_first=self.batch_first, total_length=total_length)
        return output, h_n
