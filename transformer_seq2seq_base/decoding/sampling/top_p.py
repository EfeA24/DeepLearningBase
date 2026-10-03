# Purpose: Provides nucleus (top-p) logit filtering and sampling for one seq2seq decoding step, always keeping at least one token.
"""Top-p (nucleus) filtering / sampling.

Tokens are sorted by probability; the smallest prefix whose cumulative probability reaches
``top_p`` is kept (the token that crosses the threshold is included), plus at least
``min_tokens_to_keep`` tokens. Others are set to ``filter_value`` (default -inf).

    logits (..., V) -> filtered logits (..., V);  sample_top_p(logits (B, V)) -> (B,) ids.

Requires: torch only.
"""
from __future__ import annotations

from typing import Optional

import torch


def top_p_filter(
    logits: torch.Tensor,
    top_p: float,
    min_tokens_to_keep: int = 1,
    filter_value: float = float("-inf"),
) -> torch.Tensor:
    """Mask logits outside the nucleus along the last dimension (returns a new tensor)."""
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    if min_tokens_to_keep < 1:
        raise ValueError("min_tokens_to_keep must be >= 1")
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits.float(), dim=-1)
    probs_before = torch.cumsum(sorted_probs, dim=-1) - sorted_probs
    remove_sorted = probs_before >= top_p
    remove_sorted[..., :min_tokens_to_keep] = False
    remove = torch.zeros_like(remove_sorted).scatter(-1, sorted_indices, remove_sorted)
    return logits.masked_fill(remove, filter_value)


def sample_top_p(
    logits: torch.Tensor,
    top_p: float,
    temperature: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample one token per row from temperature-scaled nucleus-filtered logits (B, V)."""
    if logits.dim() != 2:
        raise ValueError(f"logits must be (B, V), got {tuple(logits.shape)}")
    if temperature <= 0:
        raise ValueError("temperature must be > 0; use argmax for greedy decoding")
    filtered = top_p_filter(logits.float() / temperature, top_p)
    probs = torch.softmax(filtered, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
