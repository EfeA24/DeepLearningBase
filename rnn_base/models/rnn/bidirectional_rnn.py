# Purpose: Provides a bidirectional vanilla RNN wrapper with an explicit concat/sum/mean merge policy for direction outputs and states.
"""Bidirectional ``nn.RNN`` wrapper with explicit direction merging.

Shapes (batch-first):
- x: (B, T, input_size); optional h0: (num_layers * 2, B, hidden_size)
- raw nn.RNN output (B, T, 2 * H) is laid out as [forward | backward] on the last dim.
- merged output: (B, T, 2 * H) for merge="concat", (B, T, H) for "sum" / "mean".
- h_n returned raw: (num_layers * 2, B, H), layer-major ([l0 fwd, l0 bwd, l1 fwd, ...]).
  ``merge_final_state(h_n)`` -> (num_layers, B, 2H or H) using the same merge policy.

Padding: without ``lengths`` the backward direction starts at the padded end and reads pad
steps first. Pass ``lengths`` (B,) for right-padded batches so inputs are packed and each
direction sees only valid steps (padded output positions are zeros).

Requires: torch only. No sibling files.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

_MERGES = ("concat", "sum", "mean")


class BidirectionalRNN(nn.Module):
    """Bidirectional vanilla RNN; outputs merged according to ``merge``."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        merge: str = "concat",
        nonlinearity: str = "tanh",
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if merge not in _MERGES:
            raise ValueError(f"merge must be one of {_MERGES}, got {merge!r}.")
        if dropout > 0.0 and num_layers == 1:
            raise ValueError("dropout is applied between layers only; it needs num_layers > 1.")
        self.merge = merge
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            bias=bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

    @property
    def output_size(self) -> int:
        return 2 * self.hidden_size if self.merge == "concat" else self.hidden_size

    def _merge_pair(self, forward: torch.Tensor, backward: torch.Tensor) -> torch.Tensor:
        if self.merge == "concat":
            return torch.cat([forward, backward], dim=-1)
        if self.merge == "sum":
            return forward + backward
        return (forward + backward) / 2.0

    def merge_final_state(self, h_n: torch.Tensor) -> torch.Tensor:
        """(num_layers * 2, B, H) -> (num_layers, B, 2H) for concat or (num_layers, B, H) otherwise."""
        batch_size = h_n.shape[1]
        per_direction = h_n.reshape(self.num_layers, 2, batch_size, self.hidden_size)
        return self._merge_pair(per_direction[:, 0], per_direction[:, 1])

    def forward(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(f"x must have shape (B, T, input_size), got {tuple(x.shape)}.")
        if lengths is None:
            output, h_n = self.rnn(x, h0)
        else:
            cpu_lengths = lengths.detach().to(device="cpu", dtype=torch.int64)
            if cpu_lengths.numel() and int(cpu_lengths.min()) < 1:
                raise ValueError("All lengths must be >= 1 when packing.")
            packed = pack_padded_sequence(x, cpu_lengths, batch_first=True, enforce_sorted=False)
            packed_output, h_n = self.rnn(packed, h0)
            output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=x.shape[1])
        forward, backward = output.split(self.hidden_size, dim=-1)
        return self._merge_pair(forward, backward), h_n
