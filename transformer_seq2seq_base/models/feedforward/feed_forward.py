# Purpose: Implements the position-wise feed-forward network (ReLU or GELU selectable) used inside seq2seq encoder and decoder blocks.
"""Position-wise feed-forward network.

    x (B, T, d_model) -> Linear(d_model, d_ff) -> activation -> dropout -> Linear(d_ff, d_model)
Residual connection and residual dropout belong to the enclosing block, not to this module.

Requires: torch only.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

_ACTIVATIONS = {"relu": F.relu, "gelu": F.gelu}


class PositionwiseFeedForward(nn.Module):
    """Two-layer FFN applied independently at every position.

    Args:
        d_model: input/output width.
        d_ff: hidden width (commonly 4 * d_model).
        activation: "relu" (original Transformer) or "gelu".
        dropout: dropout applied after the activation.
        bias: use biases in both linear layers.
    """

    def __init__(self, d_model: int, d_ff: int, activation: str = "relu", dropout: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        if d_model < 1 or d_ff < 1:
            raise ValueError("d_model and d_ff must be positive")
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(_ACTIVATIONS)}, got {activation!r}")
        self.activation_name = activation
        self.activation = _ACTIVATIONS[activation]
        self.linear1 = nn.Linear(d_model, d_ff, bias=bias)
        self.linear2 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
