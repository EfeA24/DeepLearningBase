# Purpose: Provides learned absolute position embeddings with context-length validation and KV-cache offsets for GPT-style decoders.
"""Learned absolute positional embedding (GPT-2 ``wpe``).

Contract:
    forward(seq_len=T, offset=past_len)        -> (1, T, d_model) for positions offset..offset+T-1
    forward(position_ids=LongTensor (B, T))    -> (B, T, d_model)  (e.g. left padding / packing)

During incremental decoding pass ``offset = number of cached tokens`` so new tokens get the
correct absolute positions. Every position must be < ``max_positions`` (the trained context
length); exceeding it raises instead of silently wrapping.

Requires: torch.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class LearnedPositionalEmbedding(nn.Module):
    """Trainable position table of shape (max_positions, d_model).

    Args:
        max_positions: Maximum context length supported.
        d_model: Embedding width.
        init_std: Std of the normal initialisation.
    """

    def __init__(self, max_positions: int, d_model: int, init_std: float = 0.02) -> None:
        super().__init__()
        if max_positions <= 0 or d_model <= 0:
            raise ValueError("max_positions and d_model must be positive")
        self.max_positions = max_positions
        self.embedding = nn.Embedding(max_positions, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=init_std)

    def forward(
        self,
        seq_len: Optional[int] = None,
        offset: int = 0,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if position_ids is None:
            if seq_len is None:
                raise ValueError("Pass either seq_len (with optional offset) or position_ids")
            if offset < 0:
                raise ValueError(f"offset must be >= 0, got {offset}")
            if offset + seq_len > self.max_positions:
                raise ValueError(
                    f"positions {offset}..{offset + seq_len - 1} exceed max_positions={self.max_positions}"
                )
            position_ids = torch.arange(
                offset, offset + seq_len, device=self.embedding.weight.device
            ).unsqueeze(0)
        else:
            if position_ids.dtype not in (torch.int32, torch.int64):
                raise TypeError(f"position_ids must be integer, got {position_ids.dtype}")
            if position_ids.numel():
                lo, hi = int(position_ids.min()), int(position_ids.max())
                if lo < 0 or hi >= self.max_positions:
                    raise ValueError(
                        f"position_ids range [{lo}, {hi}] outside [0, {self.max_positions})"
                    )
        return self.embedding(position_ids)
