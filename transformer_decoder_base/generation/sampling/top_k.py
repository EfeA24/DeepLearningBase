# Purpose: Filters next-token logits to the top-k candidates (others set to -inf) before sampling in decoder LMs.
"""Top-k filtering.

Contract: logits (B, V) (or (..., V)) -> same shape; every logit strictly below the k-th
largest value of its row is replaced with ``filter_value`` (default -inf).

Edge cases:
    k >= V      -> no-op
    k < 1       -> ValueError (use no filter instead of k=0)
    ties        -> tokens tied with the k-th value are all kept, so slightly more than k tokens
                   may survive (same behavior as Hugging Face ``TopKLogitsWarper``)
At least one token always survives (k >= 1). Already ``-inf`` logits stay ``-inf``.

Processor contract: ``processor(input_ids, logits) -> logits``.

Requires: torch.
"""
from __future__ import annotations

from typing import Optional

import torch


def top_k_filter(logits: torch.Tensor, k: int, filter_value: float = float("-inf")) -> torch.Tensor:
    """Keep the ``k`` highest logits per row (ties at the threshold kept); returns a new tensor."""
    if not isinstance(k, int) or k < 1:
        raise ValueError(f"k must be an integer >= 1, got {k!r}")
    vocab = logits.shape[-1]
    if k >= vocab:
        return logits
    kth_value = torch.topk(logits, k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < kth_value, filter_value)


def sample_top_k(
    logits: torch.Tensor, k: int, generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """Sample one token per row from the top-k distribution; logits (B, V) -> ids (B,)."""
    if logits.dim() != 2:
        raise ValueError(f"logits must be (B, V), got {tuple(logits.shape)}")
    probs = torch.softmax(top_k_filter(logits, k).float(), dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


class TopKLogitsProcessor:
    """Logits processor wrapper around :func:`top_k_filter`."""

    def __init__(self, k: int, filter_value: float = float("-inf")) -> None:
        if not isinstance(k, int) or k < 1:
            raise ValueError(f"k must be an integer >= 1, got {k!r}")
        self.k = k
        self.filter_value = filter_value

    def __call__(self, input_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        return top_k_filter(logits, self.k, self.filter_value)
