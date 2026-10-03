# Purpose: Masked-language-modeling head (dense + activation + LayerNorm, then vocabulary projection) with opt-in weight tying to a caller-supplied embedding matrix.
"""MLM prediction head (BERT "cls.predictions").

    logits = decoder(LayerNorm(act(dense(h)))) + bias

Input: hidden_states (B, T, D).
Output: logits (B, T, V), or (N, V) when ``selection_mask`` (B, T) bool is given (only the N selected
positions are projected - saves memory since only ~15% of tokens carry MLM labels; select with
``labels != -100``). No softmax.

Weight tying: only when the caller passes ``tied_weight`` (the (V, D) token embedding Parameter, e.g.
``TokenEmbedding.weight`` or ``TransformerEmbedding.word_embedding_weight``). The decoder then shares
that Parameter; the output bias stays separate.

Requires: torch only.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["MLMHead"]


class MLMHead(nn.Module):
    """Transform + vocabulary projection."""

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-12,
        tied_weight: Optional[nn.Parameter] = None,
    ) -> None:
        super().__init__()
        if activation not in ("gelu", "relu", "gelu_tanh"):
            raise ValueError(f"activation must be 'gelu', 'gelu_tanh' or 'relu', got {activation!r}")
        self.activation = activation
        self.dense = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.decoder = nn.Linear(d_model, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        if tied_weight is not None:
            self.tie_weights(tied_weight)

    def tie_weights(self, weight: nn.Parameter) -> None:
        """Share the decoder weight with an embedding matrix of shape (vocab_size, d_model)."""
        if not isinstance(weight, nn.Parameter):
            raise TypeError("tied_weight must be an nn.Parameter (the embedding's weight)")
        if weight.shape != self.decoder.weight.shape:
            raise ValueError(f"tied weight shape {tuple(weight.shape)} != expected {tuple(self.decoder.weight.shape)}")
        self.decoder.weight = weight

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            return F.relu(x)
        return F.gelu(x, approximate="tanh" if self.activation == "gelu_tanh" else "none")

    def forward(self, hidden_states: torch.Tensor, selection_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if selection_mask is not None:
            if selection_mask.dtype != torch.bool or selection_mask.shape != hidden_states.shape[:-1]:
                raise ValueError("selection_mask must be a bool tensor shaped like hidden_states[..., 0]")
            hidden_states = hidden_states[selection_mask]
        x = self.layer_norm(self._activate(self.dense(hidden_states)))
        return self.decoder(x) + self.bias
