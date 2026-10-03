# Purpose: Maps token ids to dense embeddings with explicit vocab_size and padding_idx for recurrent sequence models.
"""Token embedding.

- Input: integer ids (B, T) (any leading shape works), values in [0, vocab_size).
- Output: float embeddings (B, T, embedding_dim).
- ``padding_idx`` row is initialized to zeros and receives no gradient (nn.Embedding semantics);
  set it to the pad id used by the collator.

Requires: torch only. No sibling files.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    """Thin, validated wrapper around ``nn.Embedding``."""

    def __init__(self, vocab_size: int, embedding_dim: int, padding_idx: Optional[int] = None) -> None:
        super().__init__()
        if vocab_size <= 0 or embedding_dim <= 0:
            raise ValueError("vocab_size and embedding_dim must be positive.")
        if padding_idx is not None and not 0 <= padding_idx < vocab_size:
            raise ValueError(f"padding_idx {padding_idx} must be in [0, {vocab_size}).")
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

    @property
    def weight(self) -> torch.Tensor:
        return self.embedding.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.dtype not in (torch.long, torch.int):
            raise TypeError(f"token_ids must be int64/int32, got {token_ids.dtype}.")
        return self.embedding(token_ids)
