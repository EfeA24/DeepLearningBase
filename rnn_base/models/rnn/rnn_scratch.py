# Purpose: Unrolls an educational from-scratch vanilla RNN over time (optionally multi-layer) to show how recurrent layers work.
"""Scratch multi-layer vanilla RNN (EDUCATIONAL reference; slow Python time loop).

Shapes (batch_first=True, the default):
- x: (B, T, input_size); optional h0: (num_layers, B, hidden_size), zeros if None
- returns ``output`` (B, T, hidden_size) from the last layer and ``h_n`` (num_layers, B, hidden_size)
With ``batch_first=False`` x/output are (T, B, ...). Unidirectional only.

Optional ``lengths`` (B,) handles right-padded batches: for t >= length the hidden state is
frozen (so ``h_n`` is the state at each sequence's last valid step) and outputs are zeros.
This matches nn.RNN with packed sequences.

Dropout (``dropout`` > 0) is applied to every layer's output except the last, like nn.RNN.
The cell is defined locally (self-contained); a custom cell can be injected via
``cell_factory(input_size, hidden_size) -> nn.Module`` whose forward is ``(x (B,I), h (B,H)) -> h'``.
``copy_from_torch_rnn`` copies weights from a matching ``nn.RNN`` for equivalence checks
(default cells only).

Requires: torch only. Self-contained (rnn_cell_scratch.py is not imported).
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class _VanillaRNNCell(nn.Module):
    """Local copy of the scratch cell: h' = act(x W_ih^T + b_ih + h W_hh^T + b_hh)."""

    def __init__(self, input_size: int, hidden_size: int, bias: bool, nonlinearity: str) -> None:
        super().__init__()
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
        bound = 1.0 / math.sqrt(hidden_size)
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -bound, bound)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        pre_activation = x @ self.weight_ih.T + h @ self.weight_hh.T
        if self.bias_ih is not None:
            pre_activation = pre_activation + self.bias_ih + self.bias_hh
        return torch.tanh(pre_activation) if self.nonlinearity == "tanh" else torch.relu(pre_activation)


class ScratchRNN(nn.Module):
    """Educational time-unrolled multi-layer vanilla RNN."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = "tanh",
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0.0,
        cell_factory: Optional[Callable[[int, int], nn.Module]] = None,
    ) -> None:
        super().__init__()
        if input_size <= 0 or hidden_size <= 0 or num_layers <= 0:
            raise ValueError("input_size, hidden_size and num_layers must be positive.")
        if nonlinearity not in ("tanh", "relu"):
            raise ValueError(f"nonlinearity must be 'tanh' or 'relu', got {nonlinearity!r}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout

        cells = []
        for layer in range(num_layers):
            layer_input = input_size if layer == 0 else hidden_size
            if cell_factory is None:
                cells.append(_VanillaRNNCell(layer_input, hidden_size, bias, nonlinearity))
            else:
                cells.append(cell_factory(layer_input, hidden_size))
        self.cells = nn.ModuleList(cells)

    @torch.no_grad()
    def copy_from_torch_rnn(self, rnn: nn.RNN) -> None:
        """Copy weights from a unidirectional ``nn.RNN`` with the same sizes (default cells only)."""
        if rnn.bidirectional or rnn.num_layers != self.num_layers or rnn.hidden_size != self.hidden_size:
            raise ValueError("nn.RNN must be unidirectional with matching num_layers and hidden_size.")
        for layer, cell in enumerate(self.cells):
            cell.weight_ih.copy_(getattr(rnn, f"weight_ih_l{layer}"))
            cell.weight_hh.copy_(getattr(rnn, f"weight_hh_l{layer}"))
            if cell.bias_ih is not None:
                cell.bias_ih.copy_(getattr(rnn, f"bias_ih_l{layer}"))
                cell.bias_hh.copy_(getattr(rnn, f"bias_hh_l{layer}"))

    def forward(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(f"x must be 3-D, got shape {tuple(x.shape)}.")
        if not self.batch_first:
            x = x.transpose(0, 1)
        batch_size, steps, features = x.shape
        if features != self.input_size:
            raise ValueError(f"Expected input_size {self.input_size}, got {features}.")
        if steps == 0:
            raise ValueError("Sequence length must be at least 1.")

        expected = (self.num_layers, batch_size, self.hidden_size)
        if h0 is None:
            h0 = x.new_zeros(expected)
        elif tuple(h0.shape) != expected:
            raise ValueError(f"h0 must have shape {expected}, got {tuple(h0.shape)}.")

        mask = None
        if lengths is not None:
            if lengths.shape != (batch_size,):
                raise ValueError(f"lengths must have shape ({batch_size},).")
            positions = torch.arange(steps, device=x.device)
            mask = positions.unsqueeze(0) < lengths.to(x.device).unsqueeze(1)

        layer_input = x
        final_states = []
        for layer, cell in enumerate(self.cells):
            h = h0[layer]
            outputs = []
            for t in range(steps):
                h_new = cell(layer_input[:, t], h)
                if mask is None:
                    h = h_new
                    outputs.append(h_new)
                else:
                    valid = mask[:, t].unsqueeze(-1)
                    h = torch.where(valid, h_new, h)
                    outputs.append(torch.where(valid, h_new, torch.zeros_like(h_new)))
            layer_output = torch.stack(outputs, dim=1)
            final_states.append(h)
            if self.dropout > 0.0 and layer < self.num_layers - 1:
                layer_output = F.dropout(layer_output, p=self.dropout, training=self.training)
            layer_input = layer_output

        output = layer_input if self.batch_first else layer_input.transpose(0, 1)
        return output, torch.stack(final_states, dim=0)
