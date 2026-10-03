# Purpose: Educational/reference LSTM cell implementing the (i, f, g, o) gate equations explicitly for one time step, weight-compatible with nn.LSTMCell.
"""Scratch LSTM cell (EDUCATIONAL REFERENCE - use nn.LSTM / nn.LSTMCell for production speed).

Gate equations (PyTorch convention, identical to ``torch.nn.LSTMCell``)::

    i  = sigmoid(W_ii x + b_ii + W_hi h + b_hi)      input gate
    f  = sigmoid(W_if x + b_if + W_hf h + b_hf)      forget gate
    g  = tanh   (W_ig x + b_ig + W_hg h + b_hg)      candidate cell
    o  = sigmoid(W_io x + b_io + W_ho h + b_ho)      output gate
    c' = f * c + i * g
    h' = o * tanh(c')

Parameter layout (same names/shapes as nn.LSTMCell, so ``load_state_dict`` works directly):
    ``weight_ih`` ``(4H, I)``, ``weight_hh`` ``(4H, H)``, ``bias_ih`` ``(4H,)``, ``bias_hh`` ``(4H,)``,
    with rows stacked in gate order ``[i | f | g | o]``.

Shapes: ``x`` ``(B, I)``; ``h``, ``c`` ``(B, H)``; returns ``(h', c')`` each ``(B, H)``.
Initialization matches PyTorch: ``U(-1/sqrt(H), 1/sqrt(H))`` for all parameters.

Requires: torch only.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn


def lstm_cell_step(
    x: torch.Tensor,
    state: Tuple[torch.Tensor, torch.Tensor],
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: Optional[torch.Tensor] = None,
    bias_hh: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Functional single LSTM step with explicit gates (see module docstring for equations)."""
    h, c = state
    gates = x @ weight_ih.t() + h @ weight_hh.t()
    if bias_ih is not None:
        gates = gates + bias_ih
    if bias_hh is not None:
        gates = gates + bias_hh
    i_lin, f_lin, g_lin, o_lin = gates.chunk(4, dim=-1)
    i = torch.sigmoid(i_lin)
    f = torch.sigmoid(f_lin)
    g = torch.tanh(g_lin)
    o = torch.sigmoid(o_lin)
    c_next = f * c + i * g
    h_next = o * torch.tanh(c_next)
    return h_next, c_next


class ScratchLSTMCell(nn.Module):
    """One-step LSTM cell built from tensor ops; drop-in weight-compatible with nn.LSTMCell."""

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super().__init__()
        if input_size < 1 or hidden_size < 1:
            raise ValueError("input_size and hidden_size must be >= 1.")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.empty(4 * hidden_size))
            self.bias_hh = nn.Parameter(torch.empty(4 * hidden_size))
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            nn.init.uniform_(p, -bound, bound)

    def init_state(self, batch_size: int, device=None, dtype=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Zero ``(h, c)`` each ``(B, H)``."""
        h = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        return h, torch.zeros_like(h)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 2 or x.shape[1] != self.input_size:
            raise ValueError(f"x must be (B, {self.input_size}), got {tuple(x.shape)}.")
        if state is None:
            state = self.init_state(x.shape[0], device=x.device, dtype=x.dtype)
        return lstm_cell_step(x, state, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)
