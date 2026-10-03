# Purpose: Implements the SwiGLU feed-forward network used in Llama-style causal decoder blocks.
"""SwiGLU MLP (Shazeer, 2020 "GLU Variants Improve Transformer").

    out = down( silu(gate(x)) * up(x) )

Shapes: x (..., d_model) -> (..., d_model); gate/up: d_model -> hidden_dim, down: hidden_dim -> d_model.

Intermediate-size convention (Llama): start from the classic 4 * d_model FFN and take 2/3
of it so the three-matrix SwiGLU has roughly the same parameter count as a two-matrix MLP,
optionally scale by ``ffn_dim_multiplier``, then round UP to a multiple of ``multiple_of``:

    hidden = int(2 * 4 * d_model / 3)
    hidden = int(ffn_dim_multiplier * hidden)          (if given)
    hidden = multiple_of * ceil(hidden / multiple_of)

e.g. d_model=4096, multiple_of=256 -> 11008. Pass ``hidden_dim`` to override.

Requires: torch.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


def swiglu_hidden_dim(
    d_model: int, multiple_of: int = 256, ffn_dim_multiplier: Optional[float] = None
) -> int:
    """Return the Llama-convention SwiGLU intermediate size."""
    if d_model <= 0 or multiple_of <= 0:
        raise ValueError("d_model and multiple_of must be positive")
    hidden = int(2 * 4 * d_model / 3)
    if ffn_dim_multiplier is not None:
        if ffn_dim_multiplier <= 0:
            raise ValueError(f"ffn_dim_multiplier must be positive, got {ffn_dim_multiplier}")
        hidden = int(ffn_dim_multiplier * hidden)
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward block.

    Args:
        d_model: Input/output width.
        hidden_dim: Intermediate width; default from :func:`swiglu_hidden_dim`.
        multiple_of: Rounding granularity for the default hidden size.
        ffn_dim_multiplier: Optional scale for the default hidden size.
        bias: Biases in the three projections (Llama uses False).
        dropout: Dropout applied to the output (training only).
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: Optional[int] = None,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        bias: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = swiglu_hidden_dim(d_model, multiple_of, ffn_dim_multiplier)
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))
