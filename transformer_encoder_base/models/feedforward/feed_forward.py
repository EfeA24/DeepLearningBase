# Purpose: Two-layer position-wise feed-forward network (Linear-activation-dropout-Linear-dropout) for transformer encoder blocks.
"""Position-wise feed-forward network (FFN).

Shape: (B, T, d_model) -> (B, T, d_model); applied independently at every position.
Order: linear1 (d_model -> d_ff) -> activation -> dropout(activation_dropout)
       -> linear2 (d_ff -> d_model) -> dropout(dropout).
The output dropout lives here, so encoder blocks add the FFN output to the residual directly.

Activation: "relu", "gelu", "gelu_tanh", "silu", or any nn.Module / callable.

Requires: torch only.
"""
from __future__ import annotations

from typing import Callable, Union

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["FeedForward", "resolve_activation"]

ActivationSpec = Union[str, nn.Module, Callable[[torch.Tensor], torch.Tensor]]


def resolve_activation(activation: ActivationSpec) -> Callable[[torch.Tensor], torch.Tensor]:
    """Map an activation name to a callable; modules/callables are returned unchanged."""
    if not isinstance(activation, str):
        if not callable(activation):
            raise TypeError(f"activation must be a name or callable, got {type(activation).__name__}")
        return activation
    table = {
        "relu": F.relu,
        "gelu": F.gelu,
        "gelu_tanh": lambda x: F.gelu(x, approximate="tanh"),
        "silu": F.silu,
    }
    if activation not in table:
        raise ValueError(f"unknown activation {activation!r}; choose from {sorted(table)}")
    return table[activation]


class FeedForward(nn.Module):
    """Linear -> activation -> dropout -> Linear -> dropout."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: ActivationSpec = "gelu",
        dropout: float = 0.1,
        activation_dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if d_model <= 0 or d_ff <= 0:
            raise ValueError("d_model and d_ff must be positive")
        self.linear1 = nn.Linear(d_model, d_ff, bias=bias)
        self.linear2 = nn.Linear(d_ff, d_model, bias=bias)
        self.activation = resolve_activation(activation)
        self.activation_dropout = nn.Dropout(activation_dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.activation_dropout(self.activation(self.linear1(x)))
        return self.dropout(self.linear2(hidden))
