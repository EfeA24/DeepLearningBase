# Purpose: Provides a GELU feed-forward MLP (exact or tanh-approximated) for GPT-style causal decoder blocks.
"""GELU MLP: ``dropout(fc_out(gelu(fc_in(x))))``.

Shapes: x (..., d_model) -> (..., d_model); hidden width defaults to ``4 * d_model``.
``approximate="tanh"`` reproduces the original GPT-2 "gelu_new" formula
0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))); ``"none"`` is the exact erf GELU.
Checkpoints trained with one variant should be run with the same variant.

Requires: torch.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class GELUMLP(nn.Module):
    """GELU feed-forward block.

    Args:
        d_model: Input/output width.
        hidden_dim: Intermediate width (default ``4 * d_model``).
        approximate: "none" (exact) or "tanh".
        dropout: Output dropout probability.
        bias: Biases in both linear layers.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: Optional[int] = None,
        approximate: str = "none",
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if approximate not in ("none", "tanh"):
            raise ValueError(f"approximate must be 'none' or 'tanh', got {approximate!r}")
        hidden_dim = 4 * d_model if hidden_dim is None else hidden_dim
        if d_model <= 0 or hidden_dim <= 0:
            raise ValueError("d_model and hidden_dim must be positive")
        self.approximate = approximate
        self.fc_in = nn.Linear(d_model, hidden_dim, bias=bias)
        self.fc_out = nn.Linear(hidden_dim, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc_out(F.gelu(self.fc_in(x), approximate=self.approximate)))
