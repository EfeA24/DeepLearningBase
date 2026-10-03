# Purpose: Provides sinusoidal (buffer) and learned absolute positional encodings for seq2seq transformers with max_len and decoding-offset validation.
"""Absolute positional encodings.

Input/output: x (B, T, d_model) -> x + PE[offset : offset + T] (then dropout), same shape.
``offset`` is the number of already-decoded positions when using an incremental KV cache.
Positions must satisfy offset + T <= max_len, otherwise ValueError.

    SinusoidalPositionalEncoding: fixed table stored as a non-persistent buffer (not saved
        in state_dict, rebuilt from max_len/d_model; odd d_model supported).
    LearnedPositionalEncoding:    trainable nn.Embedding(max_len, d_model).

Requires: torch only.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def sinusoidal_table(max_len: int, d_model: int) -> torch.Tensor:
    """(max_len, d_model) float32 table: sin on even dims, cos on odd dims."""
    if max_len < 1 or d_model < 1:
        raise ValueError("max_len and d_model must be positive")
    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    table = torch.zeros(max_len, d_model)
    table[:, 0::2] = torch.sin(position * div_term)
    table[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
    return table


def _check_positions(x: torch.Tensor, offset: int, max_len: int, d_model: int) -> None:
    if x.dim() != 3 or x.size(-1) != d_model:
        raise ValueError(f"expected (B, T, {d_model}), got {tuple(x.shape)}")
    if offset < 0:
        raise ValueError(f"offset must be >= 0, got {offset}")
    if offset + x.size(1) > max_len:
        raise ValueError(f"positions up to {offset + x.size(1)} exceed max_len={max_len}")


class SinusoidalPositionalEncoding(nn.Module):
    """Adds fixed sinusoidal encodings, then dropout."""

    def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("table", sinusoidal_table(max_len, d_model), persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        _check_positions(x, offset, self.max_len, self.d_model)
        pe = self.table[offset : offset + x.size(1)].to(dtype=x.dtype)
        return self.dropout(x + pe.unsqueeze(0))


class LearnedPositionalEncoding(nn.Module):
    """Adds trainable absolute position embeddings, then dropout."""

    def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.0) -> None:
        super().__init__()
        if max_len < 1 or d_model < 1:
            raise ValueError("max_len and d_model must be positive")
        self.d_model = d_model
        self.max_len = max_len
        self.embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        _check_positions(x, offset, self.max_len, self.d_model)
        positions = torch.arange(offset, offset + x.size(1), device=x.device)
        return self.dropout(x + self.embedding(positions).unsqueeze(0).to(dtype=x.dtype))
