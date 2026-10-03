# Purpose: Provides a token embedding layer with optional sqrt(d) scaling and dropout (no positional encoding) for recurrent models.
"""Embedding layer: ids -> embedding * (sqrt(d) if scale) -> dropout.

- Input: integer ids (B, T). Output: (B, T, embedding_dim) float.
- No positional information is added; recurrent models get order from recurrence.
- ``scale=True`` multiplies by sqrt(embedding_dim) (mainly useful when tying weights with an
  output projection); default False.
- ``padding_idx`` follows nn.Embedding semantics (zero init, no gradient).

Requires: torch only. Self-contained (does not import token_embedding.py).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


class EmbeddingLayer(nn.Module):
    """Token embedding with optional scaling and dropout."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        scale: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if vocab_size <= 0 or embedding_dim <= 0:
            raise ValueError("vocab_size and embedding_dim must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.scale_factor = math.sqrt(embedding_dim) if scale else 1.0
        self.dropout = nn.Dropout(dropout)

    @property
    def weight(self) -> torch.Tensor:
        return self.embedding.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)
        if self.scale_factor != 1.0:
            embedded = embedded * self.scale_factor
        return self.dropout(embedded)
