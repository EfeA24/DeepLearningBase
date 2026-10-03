# Purpose: Accumulates token-weighted mean negative log-likelihood and perplexity for language models without averaging batch means.
"""Perplexity accumulator.

Perplexity = exp(sum of token NLL / number of counted tokens), aggregated over the whole
dataset. Batch means are never averaged (batches with different token counts would be mis-weighted).

Two ways to update:
    - ``update(logits, labels)``: logits (B, T, V) and ALREADY ALIGNED labels (B, T); positions
      with ``ignore_index`` (default -100) are skipped. For causal LM shift first:
      ``logits[:, :-1]`` vs ``input_ids[:, 1:]``. NLL computed in float32 with natural log.
    - ``update_from_sum(nll_sum, token_count)``: when the caller already has summed token losses
      (e.g. ``F.cross_entropy(..., reduction="sum")``) and the number of counted tokens.

Requires: torch.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class Perplexity:
    """Token-weighted NLL / perplexity accumulator."""

    def __init__(self, ignore_index: int = -100) -> None:
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        self.nll_sum = 0.0
        self.token_count = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        if logits.ndim != labels.ndim + 1 or logits.shape[:-1] != labels.shape:
            raise ValueError(f"logits {tuple(logits.shape)} must be labels shape {tuple(labels.shape)} + (V,)")
        vocab = logits.shape[-1]
        nll = F.cross_entropy(
            logits.reshape(-1, vocab).float(),
            labels.reshape(-1),
            ignore_index=self.ignore_index,
            reduction="sum",
        )
        count = int((labels != self.ignore_index).sum().item())
        self.update_from_sum(float(nll.item()), count)

    def update_from_sum(self, nll_sum: float, token_count: int) -> None:
        if token_count < 0:
            raise ValueError(f"token_count must be >= 0, got {token_count}")
        if token_count == 0 and nll_sum != 0:
            raise ValueError("nll_sum must be 0 when token_count is 0")
        self.nll_sum += float(nll_sum)
        self.token_count += int(token_count)

    def mean_nll(self) -> float:
        if self.token_count == 0:
            raise RuntimeError("No tokens accumulated")
        return self.nll_sum / self.token_count

    def compute(self) -> float:
        """Perplexity; may overflow to inf for very large mean NLL."""
        try:
            return math.exp(self.mean_nll())
        except OverflowError:
            return float("inf")
