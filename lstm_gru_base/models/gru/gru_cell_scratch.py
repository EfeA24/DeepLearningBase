# Purpose: Educational/reference GRU cell implementing the (r, z, n) gate equations explicitly for one time step, weight-compatible with nn.GRUCell.
"""Scratch GRU cell (EDUCATIONAL REFERENCE - use nn.GRU / nn.GRUCell for production speed).

Gate equations (PyTorch convention, identical to ``torch.nn.GRUCell``)::

    r  = sigmoid(W_ir x + b_ir + W_hr h + b_hr)          reset gate
    z  = sigmoid(W_iz x + b_iz + W_hz h + b_hz)          update gate
    n  = tanh   (W_in x + b_in + r * (W_hn h + b_hn))    candidate (reset applied AFTER the hidden matmul + bias)
    h' = (1 - z) * n + z * h

Note the PyTorch/cuDNN variant: ``r`` multiplies ``(W_hn h + b_hn)``, not ``h`` before the matmul as in
the original Cho et al. (2014) formulation; and ``z`` weights the OLD state. Both matter for weight copying.

Parameter layout (same names/shapes as nn.GRUCell): ``weight_ih`` ``(3H, I)``, ``weight_hh`` ``(3H, H)``,
``bias_ih``/``bias_hh`` ``(3H,)``, rows stacked in gate order ``[r | z | n]``.

Shapes: ``x`` ``(B, I)``; ``h`` ``(B, H)``; returns ``h'`` ``(B, H)``.

Requires: torch only.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


def gru_cell_step(
    x: torch.Tensor,
    h: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: Optional[torch.Tensor] = None,
    bias_hh: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Functional single GRU step with explicit gates (see module docstring)."""
    gi = x @ weight_ih.t()
    gh = h @ weight_hh.t()
    if bias_ih is not None:
        gi = gi + bias_ih
    if bias_hh is not None:
        gh = gh + bias_hh
    i_r, i_z, i_n = gi.chunk(3, dim=-1)
    h_r, h_z, h_n = gh.chunk(3, dim=-1)
    r = torch.sigmoid(i_r + h_r)
    z = torch.sigmoid(i_z + h_z)
    n = torch.tanh(i_n + r * h_n)
    return (1.0 - z) * n + z * h


class ScratchGRUCell(nn.Module):
    """One-step GRU cell built from tensor ops; drop-in weight-compatible with nn.GRUCell."""

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super().__init__()
        if input_size < 1 or hidden_size < 1:
            raise ValueError("input_size and hidden_size must be >= 1.")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.empty(3 * hidden_size))
            self.bias_hh = nn.Parameter(torch.empty(3 * hidden_size))
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            nn.init.uniform_(p, -bound, bound)

    def init_state(self, batch_size: int, device=None, dtype=None) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.input_size:
            raise ValueError(f"x must be (B, {self.input_size}), got {tuple(x.shape)}.")
        if h is None:
            h = self.init_state(x.shape[0], device=x.device, dtype=x.dtype)
        return gru_cell_step(x, h, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)
