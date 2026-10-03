# Purpose: Token-id to vector embedding with explicit vocabulary size and pad id for transformer encoders.
"""Token embedding.

Input: input_ids (B, T) long in [0, vocab_size). Output: (B, T, d_model) float.
``pad_id`` is passed to nn.Embedding(padding_idx=...), so the pad row is initialized to zero and
receives no gradient. ``scale=True`` multiplies by sqrt(d_model) (Transformer-paper convention;
BERT does not scale). ``validate_ids=True`` checks the id range (forces a device sync; debug only).

Requires: torch only.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

__all__ = ["TokenEmbedding"]


class TokenEmbedding(nn.Module):
    """Vocabulary embedding table."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        pad_id: Optional[int] = None,
        scale: bool = False,
        validate_ids: bool = False,
    ) -> None:
        super().__init__()
        if vocab_size <= 0 or d_model <= 0:
            raise ValueError("vocab_size and d_model must be positive")
        if pad_id is not None and not 0 <= pad_id < vocab_size:
            raise ValueError(f"pad_id={pad_id} outside [0, {vocab_size})")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_id = pad_id
        self.scale = math.sqrt(d_model) if scale else 1.0
        self.validate_ids = validate_ids
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)

    @property
    def weight(self) -> nn.Parameter:
        """Embedding matrix (vocab_size, d_model); pass to models/heads/mlm_head.py for weight tying."""
        return self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dtype.is_floating_point or input_ids.dtype == torch.bool:
            raise TypeError(f"input_ids must be an integer tensor, got {input_ids.dtype}")
        if self.validate_ids and input_ids.numel() > 0:
            low, high = int(input_ids.min()), int(input_ids.max())
            if low < 0 or high >= self.vocab_size:
                raise ValueError(f"token ids span [{low}, {high}] outside [0, {self.vocab_size})")
        out = self.embedding(input_ids)
        return out * self.scale if self.scale != 1.0 else out
