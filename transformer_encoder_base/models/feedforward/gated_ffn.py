# Purpose: Gated feed-forward network (GLU family: GEGLU, SwiGLU, ReGLU, GLU) with explicit gate/up/down dims for transformer encoders.
"""Gated linear unit FFN.

out = dropout( down( dropout_act( act(gate(x)) * up(x) ) ) )
    gate, up: Linear(d_model -> d_ff); down: Linear(d_ff -> d_model).
Shape: (B, T, d_model) -> (B, T, d_model).

Variants by ``activation``: "gelu" -> GEGLU, "silu" -> SwiGLU, "relu" -> ReGLU, "sigmoid" -> GLU.
Parameter count is 3 * d_model * d_ff; to match a standard 4*d FFN budget, a common choice is
d_ff = round_to_multiple(2/3 * 4 * d_model). ``gated_hidden_size`` computes that.

Requires: torch only.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["GatedFeedForward", "gated_hidden_size"]

_ACTIVATIONS = {
    "gelu": F.gelu,
    "silu": F.silu,
    "relu": F.relu,
    "sigmoid": torch.sigmoid,
}


def gated_hidden_size(d_model: int, expansion: float = 4.0, multiple_of: int = 64) -> int:
    """Hidden size giving roughly the same parameter count as a ``expansion * d_model`` plain FFN."""
    if d_model <= 0 or multiple_of <= 0 or expansion <= 0:
        raise ValueError("d_model, expansion and multiple_of must be positive")
    hidden = int(2 * expansion * d_model / 3)
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)


class GatedFeedForward(nn.Module):
    """GLU-family feed-forward block."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: str = "gelu",
        dropout: float = 0.1,
        activation_dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}; choose from {sorted(_ACTIVATIONS)}")
        if d_model <= 0 or d_ff <= 0:
            raise ValueError("d_model and d_ff must be positive")
        self.activation_name = activation
        self.gate_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.up_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.down_proj = nn.Linear(d_ff, d_model, bias=bias)
        self.activation_dropout = nn.Dropout(activation_dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = _ACTIVATIONS[self.activation_name](self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(self.activation_dropout(gated)))
