# Purpose: Thin production adapter around torch.nn.GRU returning outputs and h_n with optional length-aware packing for GRU projects.
"""nn.GRU adapter (framework wrapper, not a scratch implementation).

Contract:
    - Always ``batch_first=True``: ``x`` ``(B, T, I)`` -> ``outputs`` ``(B, T, D*H)``.
    - ``h`` layout ``(L*D, B, H)`` ordered ``[layer0_fwd, layer0_bwd, layer1_fwd, ...]`` (D = 2 if bidirectional).
    - With ``lengths`` ``(B,)`` (RIGHT padding) the batch is packed with ``enforce_sorted=False``; outputs keep
      the original order, padded outputs are 0 and ``h_n`` is each sequence's last valid step.
    - ``dropout`` applies between stacked layers only.

Requires: torch only.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class PyTorchGRU(nn.Module):
    """Batch-first nn.GRU with explicit options and optional lengths."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")
        if dropout > 0 and num_layers == 1:
            raise ValueError("dropout > 0 has no effect with num_layers=1 (it applies between layers only).")
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    @property
    def num_directions(self) -> int:
        return 2 if self.gru.bidirectional else 1

    @property
    def output_size(self) -> int:
        return self.gru.hidden_size * self.num_directions

    def init_state(self, batch_size: int, device=None, dtype=None) -> torch.Tensor:
        shape = (self.gru.num_layers * self.num_directions, batch_size, self.gru.hidden_size)
        return torch.zeros(shape, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        h_0: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(f"x must be (B, T, I), got {tuple(x.shape)}.")
        if lengths is None:
            return self.gru(x, h_0)
        packed = pack_padded_sequence(x, lengths.detach().to("cpu", torch.long), batch_first=True, enforce_sorted=False)
        packed_out, h_n = self.gru(packed, h_0)
        outputs, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.shape[1])
        return outputs, h_n
