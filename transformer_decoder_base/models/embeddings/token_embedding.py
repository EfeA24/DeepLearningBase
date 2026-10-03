# Purpose: Maps token ids to embeddings for causal decoder language models with explicit vocab and padding settings.
"""Token embedding for decoder LMs.

Contract:
    input_ids: LongTensor (B, T) with values in [0, vocab_size)
    output:    (B, T, d_model)

``padding_idx`` (optional) keeps that row at zero and excludes it from gradients; GPT-style
decoders usually leave it None because pad positions are masked anyway (and pad may equal
EOS). ``scale=True`` multiplies by sqrt(d_model) (original Transformer); GPT-2/Llama do not.
Weights are initialised N(0, init_std^2). ``check_ids=True`` validates the id range (costs a
device sync; useful while debugging tokenizers).

Requires: torch.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    """Vocabulary embedding table.

    Args:
        vocab_size: Number of token ids.
        d_model: Embedding width.
        padding_idx: Optional pad id whose row stays zero.
        scale: Multiply outputs by sqrt(d_model).
        init_std: Std of the normal initialisation.
        check_ids: Validate id range on every forward.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx: Optional[int] = None,
        scale: bool = False,
        init_std: float = 0.02,
        check_ids: bool = False,
    ) -> None:
        super().__init__()
        if vocab_size <= 0 or d_model <= 0:
            raise ValueError("vocab_size and d_model must be positive")
        if padding_idx is not None and not 0 <= padding_idx < vocab_size:
            raise ValueError(f"padding_idx {padding_idx} outside [0, {vocab_size})")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.scale = math.sqrt(d_model) if scale else 1.0
        self.check_ids = check_ids
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=init_std)
        if padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[padding_idx].zero_()

    @property
    def weight(self) -> torch.Tensor:
        return self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"input_ids must be an integer tensor, got {input_ids.dtype}")
        if self.check_ids and input_ids.numel():
            lo, hi = int(input_ids.min()), int(input_ids.max())
            if lo < 0 or hi >= self.vocab_size:
                raise ValueError(f"token ids must be in [0, {self.vocab_size}), got range [{lo}, {hi}]")
        out = self.embedding(input_ids)
        return out * self.scale if self.scale != 1.0 else out
