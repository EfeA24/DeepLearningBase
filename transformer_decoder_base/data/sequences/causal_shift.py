# Purpose: Builds causal-LM decoder inputs and next-token labels with explicit BOS/EOS/pad handling (pad labels -> -100).
"""Causal-LM input/label shift helpers.

Two equivalent conventions exist; this template uses (A) everywhere:

(A) Aligned labels, shift inside the loss (Hugging Face convention, TEMPLATE DEFAULT)
    labels = input_ids.clone(); labels[pad] = -100            (same length T)
    loss   = CE(logits[:, :-1].reshape(-1, V), labels[:, 1:].reshape(-1), ignore_index=-100)
    i.e. the logit at position t predicts token t+1; labels[:, 0] is never used.
    -> :func:`labels_from_input_ids`, :func:`shift_logits_and_labels`

(B) Pre-shifted pair (data-side shift)
    ids (T+1)  ->  inputs = ids[:-1], targets = ids[1:]       (both length T)
    loss   = CE(logits.reshape(-1, V), targets.reshape(-1), ignore_index=-100)
    -> :func:`make_shifted_pair`

BOS/EOS: :func:`add_special_tokens` prepends BOS (so the first real token is predicted from
BOS) and appends EOS (so the model learns to stop). Pad positions are always -100 in labels.

Pad masking prefers an explicit ``attention_mask`` (1 = real). Masking by ``pad_id`` is wrong
when pad_id == eos_id (the real EOS label would be dropped); a ValueError is raised if you
pass ``pad_id`` equal to ``eos_id``.

Requires: torch.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch


def add_special_tokens(
    token_ids: Sequence[int], bos_id: Optional[int] = None, eos_id: Optional[int] = None
) -> List[int]:
    """Return ``[bos] + token_ids + [eos]`` (each special token only if its id is given)."""
    out = list(token_ids)
    if bos_id is not None:
        out = [bos_id] + out
    if eos_id is not None:
        out = out + [eos_id]
    return out


def labels_from_input_ids(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    pad_id: Optional[int] = None,
    eos_id: Optional[int] = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Convention (A): labels aligned with ``input_ids``; padding replaced by ``ignore_index``.

    Args:
        input_ids: (B, T) or (T,) token ids.
        attention_mask: Same shape, 1/True = real token (preferred source of padding info).
        pad_id: Used only when ``attention_mask`` is None.
        eos_id: Optional; used to reject ``pad_id == eos_id`` without an attention mask.
    """
    labels = input_ids.clone()
    if attention_mask is not None:
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")
        labels = labels.masked_fill(~attention_mask.to(torch.bool), ignore_index)
    elif pad_id is not None:
        if eos_id is not None and pad_id == eos_id:
            raise ValueError("pad_id == eos_id: pass attention_mask so real EOS labels are kept")
        labels = labels.masked_fill(input_ids == pad_id, ignore_index)
    return labels


def shift_logits_and_labels(
    logits: torch.Tensor, labels: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convention (A) shift: returns ``logits[:, :-1]`` (B, T-1, V) and ``labels[:, 1:]`` (B, T-1)."""
    if logits.dim() != 3 or labels.dim() != 2 or logits.shape[:2] != labels.shape:
        raise ValueError(
            f"Expected logits (B, T, V) and labels (B, T); got {tuple(logits.shape)} and {tuple(labels.shape)}"
        )
    return logits[:, :-1, :], labels[:, 1:]


def make_shifted_pair(
    token_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    pad_id: Optional[int] = None,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convention (B): (…, T+1) ids -> inputs ``ids[..., :-1]`` and targets ``ids[..., 1:]``.

    Targets at padding positions (per ``attention_mask[..., 1:]`` or ``pad_id``) become
    ``ignore_index``. Inputs keep the pad id (mask them with the attention mask).
    """
    if token_ids.shape[-1] < 2:
        raise ValueError("Need at least 2 tokens to build an input/target pair")
    inputs = token_ids[..., :-1].clone()
    targets = token_ids[..., 1:].clone()
    if attention_mask is not None:
        if attention_mask.shape != token_ids.shape:
            raise ValueError("attention_mask must have the same shape as token_ids")
        targets = targets.masked_fill(~attention_mask[..., 1:].to(torch.bool), ignore_index)
    elif pad_id is not None:
        targets = targets.masked_fill(targets == pad_id, ignore_index)
    return inputs, targets


def count_target_tokens(labels: torch.Tensor, ignore_index: int = -100, shifted: bool = False) -> int:
    """Number of tokens contributing to the loss (``shifted=False`` -> convention (A), skips column 0)."""
    targets = labels if shifted else labels[..., 1:]
    return int((targets != ignore_index).sum())
