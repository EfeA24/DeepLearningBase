# Purpose: Educational/reference time-unrolled multi-layer GRU built from explicit gate math, state-dict compatible with unidirectional nn.GRU.
"""Scratch unrolled GRU (EDUCATIONAL REFERENCE - a Python loop over time; use nn.GRU for speed).

Convention: identical to ``torch.nn.GRU`` (unidirectional, ``batch_first=True``):
    - Gate order ``[r | z | n]``; ``n = tanh(W_in x + b_in + r * (W_hn h + b_hn))``; ``h' = (1 - z) * n + z * h``.
    - Per layer ``weight_ih_l{k}`` ``(3H, I_k)``, ``weight_hh_l{k}`` ``(3H, H)``, ``bias_ih_l{k}``/``bias_hh_l{k}``
      ``(3H,)``; names match nn.GRU so ``scratch.load_state_dict(torch_gru.state_dict())`` works.
    - Dropout on outputs of every layer except the last (training mode only).

Shapes:
    ``x`` ``(B, T, I)``; optional ``h_0`` ``(L, B, H)``; returns ``outputs`` ``(B, T, H)`` and ``h_n`` ``(L, B, H)``.
    Optional ``lengths`` ``(B,)`` for RIGHT-padded batches freezes state after the last valid step and
    zeroes padded outputs (packed-sequence semantics).

Requires: torch only (gate math is self-contained; same equations as models/gru/gru_cell_scratch.py).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn


def _gru_step(x, h, w_ih, w_hh, b_ih, b_hh) -> torch.Tensor:
    gi = x @ w_ih.t()
    gh = h @ w_hh.t()
    if b_ih is not None:
        gi = gi + b_ih
        gh = gh + b_hh
    i_r, i_z, i_n = gi.chunk(3, dim=-1)
    h_r, h_z, h_n = gh.chunk(3, dim=-1)
    r = torch.sigmoid(i_r + h_r)
    z = torch.sigmoid(i_z + h_z)
    n = torch.tanh(i_n + r * h_n)
    return (1.0 - z) * n + z * h


class ScratchGRU(nn.Module):
    """Multi-layer unidirectional GRU unrolled step by step."""

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
            self.register_parameter(f"weight_ih_l{k}", nn.Parameter(torch.empty(3 * hidden_size, in_k)))
            self.register_parameter(f"weight_hh_l{k}", nn.Parameter(torch.empty(3 * hidden_size, hidden_size)))
            if bias:
                self.register_parameter(f"bias_ih_l{k}", nn.Parameter(torch.empty(3 * hidden_size)))
                self.register_parameter(f"bias_hh_l{k}", nn.Parameter(torch.empty(3 * hidden_size)))
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
        h_0: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 3 or x.shape[2] != self.input_size:
            raise ValueError(f"x must be (B, T, {self.input_size}), got {tuple(x.shape)}.")
        bsz, steps, _ = x.shape
        if h_0 is None:
            h_0 = x.new_zeros(self.num_layers, bsz, self.hidden_size)
        expected = (self.num_layers, bsz, self.hidden_size)
        if tuple(h_0.shape) != expected:
            raise ValueError(f"h_0 must be {expected}, got {tuple(h_0.shape)}.")
        valid = None
        if lengths is not None:
            lengths = lengths.to(x.device)
            valid = torch.arange(steps, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)

        layer_input = x
        h_n = []
        for k in range(self.num_layers):
            w_ih, w_hh, b_ih, b_hh = self._layer_params(k)
            h = h_0[k]
            outputs = []
            for t in range(steps):
                h_new = _gru_step(layer_input[:, t], h, w_ih, w_hh, b_ih, b_hh)
                if valid is not None:
                    m = valid[:, t].unsqueeze(1)
                    h = torch.where(m, h_new, h)
                    outputs.append(h_new * m)
                else:
                    h = h_new
                    outputs.append(h_new)
            layer_out = torch.stack(outputs, dim=1)
            h_n.append(h)
            if self.dropout > 0 and k < self.num_layers - 1:
                layer_out = nn.functional.dropout(layer_out, p=self.dropout, training=self.training)
            layer_input = layer_out
        return layer_input, torch.stack(h_n)
