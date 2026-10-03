# Purpose: Projects decoder hidden states to vocabulary logits with explicit, opt-in weight tying for causal LMs.
"""Language-model head.

Contract:
    hidden_states: (B, T, d_model)
    logits:        (B, T, vocab_size), raw (no softmax / log-softmax)

Weight tying is never implicit: call ``tie_weights(embedding)`` with an ``nn.Embedding``,
a module exposing ``.weight`` (vocab_size, d_model), or that Parameter itself. After tying,
the head and the embedding share ONE Parameter (state_dict contains both keys pointing to
the same storage; optimizers see it once via ``model.parameters()``).

Requires: torch.
"""
from __future__ import annotations

from typing import Union

import torch
from torch import nn


class LMHead(nn.Module):
    """Linear projection d_model -> vocab_size.

    Args:
        d_model: Hidden width.
        vocab_size: Number of output logits.
        bias: Add a per-token bias (GPT-2/Llama use False).
        init_std: Std of the normal weight initialisation.
    """

    def __init__(self, d_model: int, vocab_size: int, bias: bool = False, init_std: float = 0.02) -> None:
        super().__init__()
        if d_model <= 0 or vocab_size <= 0:
            raise ValueError("d_model and vocab_size must be positive")
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.proj = nn.Linear(d_model, vocab_size, bias=bias)
        nn.init.normal_(self.proj.weight, mean=0.0, std=init_std)
        if bias:
            nn.init.zeros_(self.proj.bias)

    def is_tied_to(self, embedding: Union[nn.Module, nn.Parameter]) -> bool:
        """Return True if the projection shares its weight Parameter with ``embedding``."""
        weight = embedding if isinstance(embedding, nn.Parameter) else getattr(embedding, "weight", None)
        return weight is self.proj.weight

    def tie_weights(self, embedding: Union[nn.Module, nn.Parameter]) -> None:
        """Share the projection weight with an embedding of shape (vocab_size, d_model)."""
        weight = embedding if isinstance(embedding, nn.Parameter) else getattr(embedding, "weight", None)
        if not isinstance(weight, nn.Parameter):
            raise TypeError("embedding must be an nn.Parameter or a module with a .weight Parameter")
        if tuple(weight.shape) != (self.vocab_size, self.d_model):
            raise ValueError(
                f"Embedding weight shape {tuple(weight.shape)} != (vocab_size, d_model) = "
                f"{(self.vocab_size, self.d_model)}"
            )
        self.proj.weight = weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[-1] != self.d_model:
            raise ValueError(f"Expected last dim {self.d_model}, got {tuple(hidden_states.shape)}")
        return self.proj(hidden_states)
