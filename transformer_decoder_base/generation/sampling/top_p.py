# Purpose: Applies nucleus (top-p) filtering to next-token logits, always keeping at least one token, for decoder LM sampling.
"""Nucleus (top-p) filtering (Holtzman et al., 2020).

Keep the smallest set of highest-probability tokens whose cumulative probability reaches
``p``; set the rest to ``filter_value`` (default -inf).

Algorithm (per row): sort logits descending, probs = softmax (float32), cum = cumsum(probs).
Token at sorted rank r is REMOVED iff the probability mass strictly before it already
reaches p, i.e. ``cum[r] - probs[r] >= p``. Hence the token that crosses the threshold is
kept, and rank 0 is always kept (its preceding mass is 0 < p). ``min_tokens_to_keep``
(default 1) additionally protects the top ranks. The mask is scattered back to vocabulary
order.

Edge cases:
    p == 1.0           -> no-op
    p <= 0 or p > 1    -> ValueError
    rows with -inf     -> those tokens have prob 0 and are dropped once mass reaches p
Ties at the boundary are resolved by sort order (deterministic for a given torch build).

Processor contract: ``processor(input_ids, logits) -> logits``.

Requires: torch.
"""
from __future__ import annotations

from typing import Optional

import torch


def top_p_filter(
    logits: torch.Tensor,
    p: float,
    min_tokens_to_keep: int = 1,
    filter_value: float = float("-inf"),
) -> torch.Tensor:
    """Return logits with tokens outside the nucleus set to ``filter_value``."""
    if not 0.0 < p <= 1.0:
        raise ValueError(f"p must be in (0, 1], got {p}")
    if min_tokens_to_keep < 1:
        raise ValueError("min_tokens_to_keep must be >= 1")
    if p == 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    sorted_probs = torch.softmax(sorted_logits.float(), dim=-1)
    mass_before = torch.cumsum(sorted_probs, dim=-1) - sorted_probs
    sorted_remove = mass_before >= p
    sorted_remove[..., :min_tokens_to_keep] = False
    remove = torch.zeros_like(sorted_remove).scatter(-1, sorted_idx, sorted_remove)
    return logits.masked_fill(remove, filter_value)


def sample_top_p(
    logits: torch.Tensor, p: float, generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """Sample one token per row from the nucleus; logits (B, V) -> ids (B,)."""
    if logits.dim() != 2:
        raise ValueError(f"logits must be (B, V), got {tuple(logits.shape)}")
    probs = torch.softmax(top_p_filter(logits, p).float(), dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


class TopPLogitsProcessor:
    """Logits processor wrapper around :func:`top_p_filter`."""

    def __init__(self, p: float, min_tokens_to_keep: int = 1, filter_value: float = float("-inf")) -> None:
        if not 0.0 < p <= 1.0:
            raise ValueError(f"p must be in (0, 1], got {p}")
        self.p = p
        self.min_tokens_to_keep = min_tokens_to_keep
        self.filter_value = filter_value

    def __call__(self, input_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        return top_p_filter(logits, self.p, self.min_tokens_to_keep, self.filter_value)
