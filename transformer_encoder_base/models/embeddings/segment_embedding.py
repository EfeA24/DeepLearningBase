# Purpose: Segment (token-type) embeddings distinguishing sentence A/B in sentence-pair encoder inputs.
"""Segment / token-type embedding.

Input: token_type_ids (B, T) long in [0, num_segments); BERT uses 0 for [CLS] + sentence A + first
[SEP] and 1 for sentence B + final [SEP]. Padding positions conventionally use 0.
Output: (B, T, d_model).

Requires: torch only.
"""
from __future__ import annotations

import torch
from torch import nn

__all__ = ["SegmentEmbedding"]


class SegmentEmbedding(nn.Module):
    """Embedding table over segment ids."""

    def __init__(self, num_segments: int, d_model: int, validate_ids: bool = False) -> None:
        super().__init__()
        if num_segments <= 0 or d_model <= 0:
            raise ValueError("num_segments and d_model must be positive")
        self.num_segments = num_segments
        self.validate_ids = validate_ids
        self.embedding = nn.Embedding(num_segments, d_model)

    def forward(self, token_type_ids: torch.Tensor) -> torch.Tensor:
        if token_type_ids.dtype.is_floating_point or token_type_ids.dtype == torch.bool:
            raise TypeError(f"token_type_ids must be an integer tensor, got {token_type_ids.dtype}")
        if self.validate_ids and token_type_ids.numel() > 0:
            if int(token_type_ids.min()) < 0 or int(token_type_ids.max()) >= self.num_segments:
                raise ValueError(f"token_type_ids must lie in [0, {self.num_segments})")
        return self.embedding(token_type_ids)
