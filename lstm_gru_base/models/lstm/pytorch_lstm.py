# Purpose: Thin production adapter around torch.nn.LSTM returning outputs and (h_n, c_n) with optional length-aware packing for LSTM projects.
"""nn.LSTM adapter (framework wrapper, not a scratch implementation).

Contract:
    - Always ``batch_first=True``: ``x`` ``(B, T, I)`` -> ``outputs`` ``(B, T, D*H)``.
    - States follow PyTorch layout ``(L*D, B, H)`` for both ``h`` and ``c`` (D = 2 if bidirectional),
      ordered ``[layer0_fwd, layer0_bwd, layer1_fwd, ...]``.
    - If ``lengths`` ``(B,)`` is given the batch must be RIGHT-padded; it is packed with
      ``enforce_sorted=False`` so outputs/states keep the original batch order, padded outputs are 0,
      and ``h_n``/``c_n`` hold each sequence's last valid step.
    - ``dropout`` applies between stacked layers only (PyTorch semantics).

Requires: torch only.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

LSTMState = Tuple[torch.Tensor, torch.Tensor]


class PyTorchLSTM(nn.Module):
    """Batch-first nn.LSTM with explicit options and optional lengths."""

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
        self.lstm = nn.LSTM(
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
        return 2 if self.lstm.bidirectional else 1

    @property
    def output_size(self) -> int:
        return self.lstm.hidden_size * self.num_directions

    def init_state(self, batch_size: int, device=None, dtype=None) -> LSTMState:
        shape = (self.lstm.num_layers * self.num_directions, batch_size, self.lstm.hidden_size)
        h = torch.zeros(shape, device=device, dtype=dtype)
        return h, torch.zeros_like(h)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[LSTMState] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, LSTMState]:
        if x.dim() != 3:
            raise ValueError(f"x must be (B, T, I), got {tuple(x.shape)}.")
        if lengths is None:
            return self.lstm(x, state)
        packed = pack_padded_sequence(x, lengths.detach().to("cpu", torch.long), batch_first=True, enforce_sorted=False)
        packed_out, new_state = self.lstm(packed, state)
        outputs, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.shape[1])
        return outputs, new_state
