# Purpose: Implements Bahdanau-style additive attention over recurrent encoder outputs with explicit key mask support for LSTM/GRU encoder-decoder and pooling models.
"""Additive (Bahdanau) attention.

score(q, k_j) = v^T tanh(W_q q + W_k k_j)   (+ optional bias inside the tanh via W_k's bias)

Shapes (batch-first):
    - ``query``: ``(B, Dq)`` (single decoder step) or ``(B, Tq, Dq)``.
    - ``keys``: ``(B, Tk, Dk)``; ``values``: ``(B, Tk, Dv)`` (defaults to ``keys``).
    - ``mask``: bool, True = valid (attend), ``(B, Tk)`` or ``(B, Tq, Tk)``.
Returns ``(context, weights)``: ``(B, Dv)`` / ``(B, Tk)`` for 2-D queries, else ``(B, Tq, Dv)`` / ``(B, Tq, Tk)``.
Rows whose keys are all masked get all-zero weights and a zero context (no NaNs).

Requires: torch only.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn


class AdditiveAttention(nn.Module):
    """Bahdanau additive attention with mask (True = valid)."""

    def __init__(self, query_dim: int, key_dim: int, attention_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.query_proj = nn.Linear(query_dim, attention_dim, bias=False)
        self.key_proj = nn.Linear(key_dim, attention_dim, bias=True)
        self.score_proj = nn.Linear(attention_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if keys.dim() != 3:
            raise ValueError(f"keys must be (B, Tk, Dk), got {tuple(keys.shape)}.")
        values = keys if values is None else values
        if values.shape[:2] != keys.shape[:2]:
            raise ValueError("values must share (B, Tk) with keys.")
        single_step = query.dim() == 2
        q = query.unsqueeze(1) if single_step else query
        if q.dim() != 3 or q.shape[0] != keys.shape[0]:
            raise ValueError(f"query must be (B, Dq) or (B, Tq, Dq) with matching B, got {tuple(query.shape)}.")

        # (B, Tq, 1, A) + (B, 1, Tk, A) -> (B, Tq, Tk)
        scores = self.score_proj(torch.tanh(self.query_proj(q).unsqueeze(2) + self.key_proj(keys).unsqueeze(1))).squeeze(-1)
        valid = None
        if mask is not None:
            valid = mask.bool()
            if valid.dim() == 2:
                valid = valid.unsqueeze(1)
            if valid.shape[-1] != keys.shape[1]:
                raise ValueError(f"mask last dim must be Tk={keys.shape[1]}, got {tuple(mask.shape)}.")
            scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        if valid is not None:
            weights = weights * valid
        weights = self.dropout(weights)
        context = torch.bmm(weights, values)
        if single_step:
            return context.squeeze(1), weights.squeeze(1)
        return context, weights
