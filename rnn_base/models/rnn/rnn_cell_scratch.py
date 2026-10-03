# Purpose: Implements an educational from-scratch vanilla RNN cell (one time step) for learning and verifying recurrent math.
"""Scratch vanilla RNN cell (EDUCATIONAL reference, not an optimized implementation).

One step:  h' = act(x @ W_ih^T + b_ih + h @ W_hh^T + b_hh),  act in {tanh, relu}.

Shapes:
- x: (B, input_size), h: (B, hidden_size) -> h': (B, hidden_size)
- weight_ih: (hidden_size, input_size), weight_hh: (hidden_size, hidden_size),
  bias_ih / bias_hh: (hidden_size,)

Parameter names and initialization (U(-1/sqrt(H), 1/sqrt(H))) match ``torch.nn.RNNCell``,
so ``cell.load_state_dict(nn.RNNCell(...).state_dict())`` works for equivalence tests.
For real training use ``torch.nn.RNN`` (see pytorch_rnn.py).

Requires: torch only. No sibling files.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


def rnn_cell_step(
    x: torch.Tensor,
    h: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: Optional[torch.Tensor] = None,
    bias_hh: Optional[torch.Tensor] = None,
    nonlinearity: str = "tanh",
) -> torch.Tensor:
    """Functional single vanilla RNN step from plain tensor ops (shapes in module docstring)."""
    pre_activation = x @ weight_ih.T + h @ weight_hh.T
    if bias_ih is not None:
        pre_activation = pre_activation + bias_ih
    if bias_hh is not None:
        pre_activation = pre_activation + bias_hh
    if nonlinearity == "tanh":
        return torch.tanh(pre_activation)
    if nonlinearity == "relu":
        return torch.relu(pre_activation)
    raise ValueError(f"nonlinearity must be 'tanh' or 'relu', got {nonlinearity!r}.")


class ScratchRNNCell(nn.Module):
    """Educational vanilla RNN cell with nn.RNNCell-compatible parameters."""

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, nonlinearity: str = "tanh") -> None:
        super().__init__()
        if input_size <= 0 or hidden_size <= 0:
            raise ValueError("input_size and hidden_size must be positive.")
        if nonlinearity not in ("tanh", "relu"):
            raise ValueError(f"nonlinearity must be 'tanh' or 'relu', got {nonlinearity!r}.")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        self.weight_ih = nn.Parameter(torch.empty(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.empty(hidden_size))
            self.bias_hh = nn.Parameter(torch.empty(hidden_size))
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.hidden_size)
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -bound, bound)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.input_size:
            raise ValueError(f"x must have shape (B, {self.input_size}), got {tuple(x.shape)}.")
        if h is None:
            h = x.new_zeros(x.shape[0], self.hidden_size)
        elif h.shape != (x.shape[0], self.hidden_size):
            raise ValueError(f"h must have shape ({x.shape[0]}, {self.hidden_size}), got {tuple(h.shape)}.")
        return rnn_cell_step(x, h, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh, self.nonlinearity)
