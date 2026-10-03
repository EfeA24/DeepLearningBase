# Purpose: Learned absolute positional embeddings (BERT-style) with max_len/offset validation, plus a clearly named fixed sinusoidal variant.
"""Absolute positional embeddings.

Both modules share one call contract:
    forward(position_ids=None, *, seq_len=None, offset=0)
        position_ids: (B, T) or (T,) long; if None, positions ``offset .. offset+seq_len-1`` are used.
    returns (B, T, D) when position_ids is (B, T), else (1, T, D) (broadcast over the batch).
Passing a (B, T) id tensor positionally makes them nn.Embedding-compatible, which is what
models/embeddings/transformer_embedding.py expects from an injected position module.

Positions must lie in [0, max_len); out-of-range positions raise ValueError (an encoder cannot
extrapolate learned positions). Left-padded batches should pass explicit position_ids.

Requires: torch only.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

__all__ = ["LearnedPositionalEmbedding", "SinusoidalPositionalEmbedding"]


def _resolve_positions(
    position_ids: Optional[torch.Tensor],
    seq_len: Optional[int],
    offset: int,
    max_len: int,
    device: torch.device,
) -> torch.Tensor:
    if position_ids is None:
        if seq_len is None:
            raise ValueError("pass either position_ids or seq_len")
        if offset < 0 or seq_len < 0:
            raise ValueError("offset and seq_len must be non-negative")
        if offset + seq_len > max_len:
            raise ValueError(f"offset + seq_len = {offset + seq_len} exceeds max_len={max_len}")
        return torch.arange(offset, offset + seq_len, device=device)
    if position_ids.dtype.is_floating_point or position_ids.dtype == torch.bool:
        raise TypeError(f"position_ids must be an integer tensor, got {position_ids.dtype}")
    if position_ids.dim() not in (1, 2):
        raise ValueError(f"position_ids must be (T,) or (B, T), got shape {tuple(position_ids.shape)}")
    if position_ids.numel() > 0 and (int(position_ids.min()) < 0 or int(position_ids.max()) >= max_len):
        raise ValueError(f"position_ids must lie in [0, {max_len})")
    return position_ids


class LearnedPositionalEmbedding(nn.Module):
    """Trainable (max_len, d_model) position table."""

    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__()
        if max_len <= 0 or d_model <= 0:
            raise ValueError("max_len and d_model must be positive")
        self.max_len = max_len
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(
        self,
        position_ids: Optional[torch.Tensor] = None,
        *,
        seq_len: Optional[int] = None,
        offset: int = 0,
    ) -> torch.Tensor:
        positions = _resolve_positions(position_ids, seq_len, offset, self.max_len, self.embedding.weight.device)
        out = self.embedding(positions)
        return out.unsqueeze(0) if out.dim() == 2 else out


class SinusoidalPositionalEmbedding(nn.Module):
    """Fixed sin/cos table (Vaswani et al.); stored as a non-persistent buffer, no parameters.

    PE[pos, 2i] = sin(pos / 10000^(2i/D)), PE[pos, 2i+1] = cos(pos / 10000^(2i/D)).
    """

    def __init__(self, max_len: int, d_model: int, base: float = 10000.0) -> None:
        super().__init__()
        if max_len <= 0 or d_model <= 0:
            raise ValueError("max_len and d_model must be positive")
        self.max_len = max_len
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(base) / d_model))
        table = torch.zeros(max_len, d_model)
        table[:, 0::2] = torch.sin(position * div_term)
        table[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        self.register_buffer("table", table, persistent=False)

    def forward(
        self,
        position_ids: Optional[torch.Tensor] = None,
        *,
        seq_len: Optional[int] = None,
        offset: int = 0,
    ) -> torch.Tensor:
        positions = _resolve_positions(position_ids, seq_len, offset, self.max_len, self.table.device)
        out = self.table[positions]
        return out.unsqueeze(0) if out.dim() == 2 else out
