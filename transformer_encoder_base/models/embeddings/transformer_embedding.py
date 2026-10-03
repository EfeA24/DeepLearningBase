# Purpose: BERT-style input embedding that sums token, position and optional segment embeddings, then applies LayerNorm and dropout.
"""Transformer encoder input embedding.

Order (BERT): out = dropout(LayerNorm(token(ids) + position(position_ids) + segment(token_type_ids)))

Inputs:
    input_ids (B, T) long; token_type_ids (B, T) long or None (-> all zeros when segments enabled);
    position_ids (B, T) / (T,) long or None (-> 0..T-1, valid for right padding; pass explicit
    ids for left padding).
Output: (B, T, d_model).

Components may be injected; each must be "ids -> embeddings" callable like nn.Embedding:
    token_embedding(input_ids) -> (B, T, D)
    position_embedding(position_ids) -> (B|1, T, D)   (models/embeddings/positional_embedding.py fits)
    segment_embedding(token_type_ids) -> (B, T, D)
When not injected, plain nn.Embedding tables are created locally (no sibling imports).
``num_segments=0`` disables segment embeddings (e.g. RoBERTa-style single-segment models).

Requires: torch only.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

__all__ = ["TransformerEmbedding"]


class TransformerEmbedding(nn.Module):
    """Token + position (+ segment) -> LayerNorm -> dropout."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int,
        pad_id: Optional[int] = None,
        num_segments: int = 2,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
        token_embedding: Optional[nn.Module] = None,
        position_embedding: Optional[nn.Module] = None,
        segment_embedding: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if vocab_size <= 0 or d_model <= 0 or max_len <= 0:
            raise ValueError("vocab_size, d_model and max_len must be positive")
        if num_segments < 0:
            raise ValueError(f"num_segments must be >= 0, got {num_segments}")
        self.max_len = max_len
        self.token_embedding = token_embedding or nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position_embedding = position_embedding or nn.Embedding(max_len, d_model)
        if segment_embedding is not None:
            self.segment_embedding: Optional[nn.Module] = segment_embedding
        elif num_segments > 0:
            self.segment_embedding = nn.Embedding(num_segments, d_model)
        else:
            self.segment_embedding = None
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    @property
    def word_embedding_weight(self) -> torch.Tensor:
        """Token embedding matrix (V, D) for optional MLM-head weight tying."""
        weight = getattr(self.token_embedding, "weight", None)
        if weight is None:
            weight = getattr(getattr(self.token_embedding, "embedding", None), "weight", None)
        if weight is None:
            raise AttributeError("injected token_embedding exposes no .weight or .embedding.weight")
        return weight

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be (B, T), got shape {tuple(input_ids.shape)}")
        seq_len = input_ids.shape[1]
        if seq_len > self.max_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_len={self.max_len}")
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        elif position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)

        embeddings = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        if self.segment_embedding is not None:
            if token_type_ids is None:
                token_type_ids = torch.zeros_like(input_ids)
            elif token_type_ids.shape != input_ids.shape:
                raise ValueError("token_type_ids must have the same shape as input_ids")
            embeddings = embeddings + self.segment_embedding(token_type_ids)
        elif token_type_ids is not None:
            raise ValueError("token_type_ids given but segment embeddings are disabled (num_segments=0)")
        return self.dropout(self.layer_norm(embeddings))
