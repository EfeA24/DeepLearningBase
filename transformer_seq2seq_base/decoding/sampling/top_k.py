# Purpose: Provides top-k logit filtering and top-k sampling for one seq2seq decoding step.
"""Top-k filtering / sampling.

    logits (..., V) raw scores -> filtered logits (..., V) with everything below the k-th
    largest value set to ``filter_value`` (default -inf). Ties at the k-th value are kept,
    so slightly more than k tokens may survive.
    sample_top_k(logits (B, V)) -> (B,) int64 sampled ids.
Temperature must be > 0 (use greedy/argmax for deterministic decoding).

Requires: torch only.
"""
from __future__ import annotations

from typing import Optional

import torch


def top_k_filter(logits: torch.Tensor, k: int, filter_value: float = float("-inf")) -> torch.Tensor:
    """Mask all but the top-``k`` logits along the last dimension (returns a new tensor)."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    k = min(k, logits.size(-1))
    kth_value = torch.topk(logits, k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < kth_value, filter_value)


def sample_top_k(
    logits: torch.Tensor,
    k: int,
    temperature: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample one token per row from temperature-scaled, top-k filtered logits (B, V)."""
    if logits.dim() != 2:
        raise ValueError(f"logits must be (B, V), got {tuple(logits.shape)}")
    if temperature <= 0:
        raise ValueError("temperature must be > 0; use argmax for greedy decoding")
    filtered = top_k_filter(logits.float() / temperature, k)
    probs = torch.softmax(filtered, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
