# Purpose: Provides source/target token embeddings for seq2seq transformers with optional sqrt(d_model) scaling and explicit shared-vs-separate weight handling.
"""Token embeddings for encoder-decoder models.

Shapes: ids (B, T) int64 -> embeddings (B, T, d_model) float.
Scaling: when ``scale=True`` outputs are multiplied by sqrt(d_model) (Vaswani et al.),
which pairs with the N(0, d_model**-0.5) init used here so scaled embeddings have unit
variance before positional encodings are added.

Shared vs separate:
    separate vocabularies -> one TokenEmbedding per side.
    shared vocabulary     -> build one TokenEmbedding and pass the SAME module to encoder
                             and decoder (see ``shared_embedding_pair``); optionally tie the
                             output projection with ``tie_output_projection``.

Requires: torch only.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    """Vocabulary embedding with optional sqrt(d_model) scaling.

    Args:
        vocab_size: number of token ids.
        d_model: embedding width.
        padding_idx: pad id whose row is zero and receives no gradient (None = none).
        scale: multiply outputs by sqrt(d_model).
    """

    def __init__(self, vocab_size: int, d_model: int, padding_idx: Optional[int] = None, scale: bool = True) -> None:
        super().__init__()
        if vocab_size < 1 or d_model < 1:
            raise ValueError("vocab_size and d_model must be positive")
        if padding_idx is not None and not 0 <= padding_idx < vocab_size:
            raise ValueError(f"padding_idx {padding_idx} outside [0, {vocab_size})")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.scale = math.sqrt(d_model) if scale else 1.0
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.embedding.weight, mean=0.0, std=self.d_model ** -0.5)
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].zero_()

    @property
    def weight(self) -> nn.Parameter:
        return self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dtype not in (torch.long, torch.int32):
            raise TypeError(f"input_ids must be int64/int32, got {input_ids.dtype}")
        return self.embedding(input_ids) * self.scale


def shared_embedding_pair(
    vocab_size: int, d_model: int, padding_idx: Optional[int] = None, scale: bool = True
) -> Tuple[TokenEmbedding, TokenEmbedding]:
    """Return (source_embedding, target_embedding) as the SAME module (shared vocab only)."""
    shared = TokenEmbedding(vocab_size, d_model, padding_idx=padding_idx, scale=scale)
    return shared, shared


def tie_output_projection(output_projection: nn.Linear, embedding: TokenEmbedding) -> None:
    """Share ``output_projection.weight`` with the target embedding weight (V, d_model)."""
    if output_projection.weight.shape != embedding.weight.shape:
        raise ValueError(
            f"cannot tie: projection weight {tuple(output_projection.weight.shape)} "
            f"!= embedding weight {tuple(embedding.weight.shape)}"
        )
    output_projection.weight = embedding.weight
