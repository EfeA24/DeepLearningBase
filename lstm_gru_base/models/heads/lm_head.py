# Purpose: Projects recurrent hidden states to vocabulary logits with optional explicit embedding weight tying for LSTM/GRU language models.
"""Language-model head.

Contract:
    - Input ``(B, T, H)`` (or ``(B, H)`` for one step); output raw logits ``(B, T, V)`` / ``(B, V)``.
    - LM loss uses inputs ``x[:, :-1]`` and targets ``x[:, 1:]`` (or pre-shifted labels) with padded labels =
      -100; no softmax here.
    - Weight tying is opt-in via ``tie_weights(embedding)``; it requires ``H == embedding_dim`` and shares the
      same Parameter object (Press & Wolf, 2017). Add a projection layer before this head if ``H`` differs.

Requires: torch only.
"""
from __future__ import annotations

import torch
from torch import nn


class LanguageModelHead(nn.Module):
    """Hidden -> vocab logits."""

    def __init__(self, hidden_size: int, vocab_size: int, bias: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=bias)

    def tie_weights(self, embedding: nn.Embedding) -> None:
        """Share ``embedding.weight`` ``(V, H)`` with the output projection."""
        if not isinstance(embedding, nn.Embedding):
            raise TypeError("tie_weights expects an nn.Embedding.")
        if embedding.weight.shape != self.decoder.weight.shape:
            raise ValueError(
                f"Cannot tie: embedding weight {tuple(embedding.weight.shape)} != projection weight "
                f"{tuple(self.decoder.weight.shape)} (need vocab_size and hidden_size == embedding_dim)."
            )
        self.decoder.weight = embedding.weight

    def is_tied(self, embedding: nn.Embedding) -> bool:
        """True if the projection shares its Parameter with ``embedding.weight``."""
        return self.decoder.weight is embedding.weight

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.shape[-1] != self.hidden_size:
            raise ValueError(f"Last dim must be {self.hidden_size}, got {hidden.shape[-1]}.")
        return self.decoder(self.dropout(hidden))
