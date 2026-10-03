# Purpose: Educational/reference time-unrolled multi-layer LSTM built from explicit gate math, state-dict compatible with unidirectional nn.LSTM.
"""Scratch unrolled LSTM (EDUCATIONAL REFERENCE - a Python loop over time; use nn.LSTM for speed).

Convention: identical to ``torch.nn.LSTM`` (unidirectional, ``batch_first=True``, no ``proj_size``):
    - Gate order ``[i | f | g | o]``; per layer ``weight_ih_l{k}`` ``(4H, I_k)``, ``weight_hh_l{k}`` ``(4H, H)``,
      ``bias_ih_l{k}``/``bias_hh_l{k}`` ``(4H,)`` with ``I_0 = input_size`` and ``I_k = H`` for k > 0.
      Parameter names match nn.LSTM, so ``scratch.load_state_dict(torch_lstm.state_dict())`` works.
    - Dropout is applied to the outputs of every layer except the last (training mode only).

Shapes:
    ``x`` ``(B, T, I)``; optional ``state = (h_0, c_0)`` each ``(L, B, H)``;
    returns ``outputs`` ``(B, T, H)`` (last layer) and ``(h_n, c_n)`` each ``(L, B, H)``.
    Optional ``lengths`` ``(B,)`` for RIGHT-padded batches: state stops updating after each sequence's
    last valid step and padded outputs are zero, matching packed-sequence semantics of nn.LSTM.

Requires: torch only (gate math is self-contained; same equations as models/lstm/lstm_cell_scratch.py).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn

LSTMState = Tuple[torch.Tensor, torch.Tensor]


def _lstm_step(x, h, c, w_ih, w_hh, b_ih, b_hh) -> LSTMState:
    gates = x @ w_ih.t() + h @ w_hh.t()
    if b_ih is not None:
        gates = gates + b_ih + b_hh
    i, f, g, o = gates.chunk(4, dim=-1)
    c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
    h_next = torch.sigmoid(o) * torch.tanh(c_next)
    return h_next, c_next


class ScratchLSTM(nn.Module):
    """Multi-layer unidirectional LSTM unrolled step by step."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        for k in range(num_layers):
            in_k = input_size if k == 0 else hidden_size
            self.register_parameter(f"weight_ih_l{k}", nn.Parameter(torch.empty(4 * hidden_size, in_k)))
            self.register_parameter(f"weight_hh_l{k}", nn.Parameter(torch.empty(4 * hidden_size, hidden_size)))
            if bias:
                self.register_parameter(f"bias_ih_l{k}", nn.Parameter(torch.empty(4 * hidden_size)))
                self.register_parameter(f"bias_hh_l{k}", nn.Parameter(torch.empty(4 * hidden_size)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            nn.init.uniform_(p, -bound, bound)

    def _layer_params(self, k: int):
        return tuple(getattr(self, f"{name}_l{k}", None) for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"))

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[LSTMState] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, LSTMState]:
        if x.dim() != 3 or x.shape[2] != self.input_size:
            raise ValueError(f"x must be (B, T, {self.input_size}), got {tuple(x.shape)}.")
        bsz, steps, _ = x.shape
        if state is None:
            h0 = x.new_zeros(self.num_layers, bsz, self.hidden_size)
            state = (h0, h0.clone())
        h0, c0 = state
        expected = (self.num_layers, bsz, self.hidden_size)
        if tuple(h0.shape) != expected or tuple(c0.shape) != expected:
            raise ValueError(f"h_0 and c_0 must be {expected}, got {tuple(h0.shape)} and {tuple(c0.shape)}.")
        valid = None
        if lengths is not None:
            lengths = lengths.to(x.device)
            valid = torch.arange(steps, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)

        layer_input = x
        h_n, c_n = [], []
        for k in range(self.num_layers):
            w_ih, w_hh, b_ih, b_hh = self._layer_params(k)
            h, c = h0[k], c0[k]
            outputs = []
            for t in range(steps):
                h_new, c_new = _lstm_step(layer_input[:, t], h, c, w_ih, w_hh, b_ih, b_hh)
                if valid is not None:
                    m = valid[:, t].unsqueeze(1)
                    h = torch.where(m, h_new, h)
                    c = torch.where(m, c_new, c)
                    outputs.append(h_new * m)
                else:
                    h, c = h_new, c_new
                    outputs.append(h_new)
            layer_out = torch.stack(outputs, dim=1)
            h_n.append(h)
            c_n.append(c)
            if self.dropout > 0 and k < self.num_layers - 1:
                layer_out = nn.functional.dropout(layer_out, p=self.dropout, training=self.training)
            layer_input = layer_out
        return layer_input, (torch.stack(h_n), torch.stack(c_n))
