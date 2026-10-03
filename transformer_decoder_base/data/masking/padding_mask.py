# Purpose: Builds padding masks and combines them with causal masks into (B, 1, Tq, Tk) attention masks for decoder LMs.
"""Padding masks for causal decoders.

Polarity: True = real (valid) token / may attend. (Int masks such as a tokenizer's
``attention_mask`` with 1 = real token have the same meaning and are accepted.)

Helpers:
    padding_mask_from_lengths(lengths (B,), max_len, padding_side) -> (B, T) bool
    padding_mask_from_ids(input_ids (B, T), pad_id)                -> (B, T) bool
    combine_causal_and_padding(padding_mask (B, Tk), q_len)        -> (B, 1, Tq, Tk) bool

``combine_causal_and_padding`` aligns queries to the END of the keys (KV-cache offset
``Tk - Tq``). Rows with no allowed key (e.g. a left-padding query position) are, by default,
re-opened to attend all keys so SDPA does not return NaN; outputs at such positions are
meaningless and must be excluded from the loss (labels -100).

Caveat: ``padding_mask_from_ids`` is wrong when pad_id == eos_id and EOS appears inside real
text; prefer masks derived from lengths or returned by the collator.

Requires: torch.
"""
from __future__ import annotations

from typing import Optional

import torch


def padding_mask_from_lengths(
    lengths: torch.Tensor, max_len: Optional[int] = None, padding_side: str = "right"
) -> torch.Tensor:
    """Return (B, max_len) bool mask with True on the first (right padding) or last (left) ``length`` slots."""
    if lengths.dim() != 1:
        raise ValueError(f"lengths must be 1D (B,), got {tuple(lengths.shape)}")
    if padding_side not in ("right", "left"):
        raise ValueError("padding_side must be 'right' or 'left'")
    if lengths.numel() and int(lengths.min()) < 0:
        raise ValueError("lengths must be non-negative")
    longest = int(lengths.max()) if lengths.numel() else 0
    max_len = longest if max_len is None else max_len
    if longest > max_len:
        raise ValueError(f"max_len ({max_len}) is smaller than the longest length ({longest})")
    idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    lengths = lengths.unsqueeze(1)
    if padding_side == "right":
        return idx < lengths
    return idx >= (max_len - lengths)


def padding_mask_from_ids(input_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Return (B, T) bool mask, True where ``input_ids != pad_id``."""
    return input_ids != pad_id


def combine_causal_and_padding(
    padding_mask: torch.Tensor,
    q_len: Optional[int] = None,
    reopen_empty_rows: bool = True,
) -> torch.Tensor:
    """Combine a key padding mask with the causal rule.

    Args:
        padding_mask: (B, Tk) bool/int, True/1 = real key (covers cached + new tokens).
        q_len: Number of query positions (the last ``q_len`` keys). Default ``Tk``.
        reopen_empty_rows: Let fully-masked query rows attend everywhere (NaN guard).

    Returns:
        (B, 1, Tq, Tk) bool, True = allowed.
    """
    if padding_mask.dim() != 2:
        raise ValueError(f"padding_mask must be (B, Tk), got {tuple(padding_mask.shape)}")
    kv_len = padding_mask.shape[1]
    q_len = kv_len if q_len is None else q_len
    if not 0 < q_len <= kv_len:
        raise ValueError(f"q_len must be in [1, {kv_len}], got {q_len}")
    causal = torch.ones(q_len, kv_len, dtype=torch.bool, device=padding_mask.device).tril(kv_len - q_len)
    mask = causal[None, None] & padding_mask.to(torch.bool)[:, None, None, :]
    if reopen_empty_rows:
        mask = mask | ~mask.any(dim=-1, keepdim=True)
    return mask
