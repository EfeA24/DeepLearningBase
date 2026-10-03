# Purpose: Creates overlapping causal-LM windows with a stride and optional masking of already-scored targets (no double counting).
"""Sliding windows over one token stream for causal-LM training/evaluation.

Windows start at 0, stride, 2*stride, ...; each window has at most ``window_size`` tokens.
Labels follow the template's aligned convention (labels[t] = input_ids[t]; the loss compares
logits[:, :-1] with labels[:, 1:]), so within a window token ``t`` is scored iff ``t >= 1`` and
``labels[t] != -100``.

Target coverage with ``mask_overlap=True`` (perplexity-style evaluation): a token that was
already scored by an earlier window has its label set to -100, so every stream token
1..N-1 is scored exactly once, and tokens in later windows are scored with at least
``window_size - stride`` tokens of context. Token 0 of the stream is never scored (no
context). With ``stride == window_size`` (no overlap) the first token of each window is also
never scored — use ``stride < window_size`` if that matters.

``mask_overlap=False`` keeps every label (overlapping tokens are trained on multiple times).
A final partial window is emitted only if it contains at least one new token to score
(or always skipped with ``drop_last=True``).

Requires: torch.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import torch


def sliding_windows(
    token_ids: Sequence[int],
    window_size: int,
    stride: int,
    mask_overlap: bool = True,
    drop_last: bool = False,
    ignore_index: int = -100,
) -> List[Dict[str, torch.Tensor]]:
    """Return windows as dicts with ``input_ids``, ``labels`` (LongTensor (L,)) and ``start`` offset."""
    if window_size < 2:
        raise ValueError(f"window_size must be >= 2, got {window_size}")
    if not 0 < stride <= window_size:
        raise ValueError(f"stride must be in [1, window_size], got {stride}")
    ids = list(token_ids)
    n = len(ids)
    windows: List[Dict[str, torch.Tensor]] = []
    scored_until = 0
    start = 0
    while start < n:
        end = min(start + window_size, n)
        if end - start < window_size and drop_last:
            break
        window = torch.tensor(ids[start:end], dtype=torch.long)
        labels = window.clone()
        if mask_overlap:
            already = max(0, min(scored_until - start + 1, end - start))
            labels[:already] = ignore_index
        if (labels[1:] != ignore_index).any():
            windows.append({"input_ids": window, "labels": labels, "start": torch.tensor(start)})
        scored_until = max(scored_until, end - 1)
        if end == n:
            break
        start += stride
    return windows
