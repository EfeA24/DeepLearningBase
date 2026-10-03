# Purpose: Provides the standard two-layer position-wise MLP (Linear-activation-Linear-dropout) for causal decoder blocks.
"""Decoder MLP: ``dropout(fc_out(act(fc_in(x))))``.

Shapes: x (..., d_model) -> (..., d_model); hidden width defaults to ``4 * d_model`` (GPT-2).
Activation is selected explicitly by name ("gelu", "gelu_tanh", "relu", "silu") or passed as
an ``nn.Module``. Dropout is applied once, on the output (GPT-2 "resid_pdrop" position).

Requires: torch.
"""
from __future__ import annotations

from typing import Optional, Union

import torch
from torch import nn


def build_activation(activation: Union[str, nn.Module]) -> nn.Module:
    """Map an activation name to a module; modules are returned unchanged."""
    if isinstance(activation, nn.Module):
        return activation
    table = {
        "gelu": lambda: nn.GELU(),
        "gelu_tanh": lambda: nn.GELU(approximate="tanh"),
        "relu": lambda: nn.ReLU(),
        "silu": lambda: nn.SiLU(),
    }
    if activation not in table:
        raise ValueError(f"Unknown activation {activation!r}; choose from {sorted(table)} or pass an nn.Module")
    return table[activation]()


class DecoderMLP(nn.Module):
    """Two-layer feed-forward network.

    Args:
        d_model: Input/output width.
        hidden_dim: Intermediate width (default ``4 * d_model``).
        activation: Activation name or module.
        dropout: Output dropout probability.
        bias: Biases in both linear layers.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: Optional[int] = None,
        activation: Union[str, nn.Module] = "gelu",
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        hidden_dim = 4 * d_model if hidden_dim is None else hidden_dim
        if d_model <= 0 or hidden_dim <= 0:
            raise ValueError("d_model and hidden_dim must be positive")
        self.fc_in = nn.Linear(d_model, hidden_dim, bias=bias)
        self.act = build_activation(activation)
        self.fc_out = nn.Linear(hidden_dim, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc_out(self.act(self.fc_in(x))))
