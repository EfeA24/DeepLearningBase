# Purpose: Projects recurrent hidden states to vocabulary logits for language modelling, with optional explicit weight tying.
"""Language-model head.

- Input: hidden states (B, T, H) (or (B, H) for a single generation step).
- Output: raw vocabulary logits (B, T, vocab_size) / (B, vocab_size); no softmax.
- Weight tying is opt-in via ``tie_weights(embedding)``: the projection then shares the
  embedding matrix (vocab_size, H), which requires embedding_dim == hidden_size.
- Loss pairing: next-token targets with -100 at padded positions (see losses/).

Requires: torch only. No sibling files.
"""
from __future__ import annotations

import torch
from torch import nn


class LanguageModelHead(nn.Module):
    """Dropout + linear projection to vocabulary logits."""

    def __init__(self, hidden_size: int, vocab_size: int, bias: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_size <= 0 or vocab_size <= 0:
            raise ValueError("hidden_size and vocab_size must be positive.")
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(hidden_size, vocab_size, bias=bias)

    def tie_weights(self, embedding: nn.Embedding) -> None:
        """Share ``embedding.weight`` (vocab_size, hidden_size) with the output projection."""
        expected = (self.vocab_size, self.hidden_size)
        if tuple(embedding.weight.shape) != expected:
            raise ValueError(
                f"Cannot tie: embedding weight has shape {tuple(embedding.weight.shape)}, expected {expected}."
            )
        self.projection.weight = embedding.weight

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.shape[-1] != self.hidden_size:
            raise ValueError(f"Expected last dim {self.hidden_size}, got {hidden.shape[-1]}.")
        return self.projection(self.dropout(hidden))
