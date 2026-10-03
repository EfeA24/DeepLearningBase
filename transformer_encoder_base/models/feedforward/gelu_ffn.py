# Purpose: GELU position-wise feed-forward network (BERT-style intermediate/output layers) for transformer encoders.
"""GELU feed-forward network.

Shape: (B, T, d_model) -> (B, T, d_model).
Order: Linear(d_model -> d_ff) -> GELU(approximate) -> dropout(activation_dropout)
       -> Linear(d_ff -> d_model) -> dropout(dropout).
``approximate="none"`` is the exact erf GELU used by original BERT; "tanh" is the
approximation used by GPT-2 style models. BERT uses d_ff = 4 * d_model.

Requires: torch only.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["GELUFeedForward"]


class GELUFeedForward(nn.Module):
    """Two-layer FFN with GELU activation."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation_dropout: float = 0.0,
        approximate: str = "none",
        bias: bool = True,
    ) -> None:
        super().__init__()
        if approximate not in ("none", "tanh"):
            raise ValueError(f"approximate must be 'none' or 'tanh', got {approximate!r}")
        if d_model <= 0 or d_ff <= 0:
            raise ValueError("d_model and d_ff must be positive")
        self.approximate = approximate
        self.intermediate = nn.Linear(d_model, d_ff, bias=bias)
        self.output = nn.Linear(d_ff, d_model, bias=bias)
        self.activation_dropout = nn.Dropout(activation_dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.intermediate(x), approximate=self.approximate)
        return self.dropout(self.output(self.activation_dropout(hidden)))
