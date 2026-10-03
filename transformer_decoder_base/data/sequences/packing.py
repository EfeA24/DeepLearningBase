# Purpose: Packs multiple tokenized samples into fixed causal-LM context windows with position ids, segment ids and boundary-aware masks.
"""Sequence packing for causal-LM training.

Policy (explicit, order-preserving greedy): samples are appended to the current window in the
given order; when the next sample does not fit in the remaining space a new window is started
(the current window is padded). No reordering / bin-packing, so results are deterministic.
Samples longer than ``context_length`` follow ``overflow``:
    "truncate" keep the first context_length tokens, "split" cut into pieces of context_length
    (each piece is a separate segment), "error" raise.

Per packed window (all LongTensor/BoolTensor of shape (L,), L = context_length):
    input_ids      tokens, pad_id in the padded tail
    labels         aligned labels (template convention: loss uses logits[:, :-1] vs labels[:, 1:]);
                   -100 at padding, at positions excluded by the optional per-sample loss mask, and
                   at the FIRST token of every segment so that the last token of one sample never
                   gets a loss for predicting the first token of the next sample.
    position_ids   restart at 0 for every segment (0 in padding)
    segment_ids    1, 2, 3, ... per segment; 0 = padding
    attention_mask True = real token (padding mask only; NOT sufficient to stop cross-sample
                   attention)

Cross-sample attention: to isolate samples, pass ``packed_attention_mask(segment_ids)``
(B, 1, L, L) bool (True = allowed; same segment AND causal) as a 4D attention mask together
with ``position_ids``. Without it, samples attend to earlier samples in the window (a common,
cheaper, but leaky choice) — the label masking above still applies.

Requires: torch.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch


def _pieces(sample: List[int], context_length: int, overflow: str) -> List[List[int]]:
    if len(sample) <= context_length:
        return [sample]
    if overflow == "truncate":
        return [sample[:context_length]]
    if overflow == "split":
        return [sample[i : i + context_length] for i in range(0, len(sample), context_length)]
    raise ValueError(f"Sample of length {len(sample)} exceeds context_length={context_length}")


def pack_sequences(
    samples: Sequence[Sequence[int]],
    context_length: int,
    pad_id: int,
    loss_masks: Optional[Sequence[Sequence[int]]] = None,
    eos_id: Optional[int] = None,
    overflow: str = "truncate",
    ignore_index: int = -100,
) -> List[Dict[str, torch.Tensor]]:
    """Pack token-id samples into windows of ``context_length``.

    Args:
        samples: Token-id lists (already tokenized; BOS included if desired).
        context_length: Window length L.
        pad_id: Id used to fill the tail of each window.
        loss_masks: Optional per-sample 0/1 lists (same lengths as samples); 0 -> label -100
            (e.g. prompt tokens of instruction data).
        eos_id: If given, appended to every sample before packing (loss-mask value 1).
        overflow: "truncate", "split" or "error" for samples longer than L.
        ignore_index: Label value for ignored positions.
    """
    if context_length < 2:
        raise ValueError(f"context_length must be >= 2, got {context_length}")
    if overflow not in ("truncate", "split", "error"):
        raise ValueError("overflow must be 'truncate', 'split' or 'error'")
    if loss_masks is not None and len(loss_masks) != len(samples):
        raise ValueError("loss_masks must have one entry per sample")

    pieces: List[tuple] = []
    for idx, sample in enumerate(samples):
        ids = list(sample)
        mask = [1] * len(ids) if loss_masks is None else [int(bool(m)) for m in loss_masks[idx]]
        if len(mask) != len(ids):
            raise ValueError(f"loss_mask length {len(mask)} != sample length {len(ids)} (sample {idx})")
        if eos_id is not None:
            ids.append(eos_id)
            mask.append(1)
        if not ids:
            continue
        id_pieces = _pieces(ids, context_length, overflow)
        mask_pieces = _pieces(mask, context_length, "split" if overflow == "split" else "truncate")
        pieces.extend(zip(id_pieces, mask_pieces))

    windows: List[Dict[str, torch.Tensor]] = []
    current: List[tuple] = []
    used = 0
    for ids, mask in pieces:
        if used + len(ids) > context_length:
            windows.append(_build_window(current, context_length, pad_id, ignore_index))
            current, used = [], 0
        current.append((ids, mask))
        used += len(ids)
    if current:
        windows.append(_build_window(current, context_length, pad_id, ignore_index))
    return windows


def _build_window(
    segments: List[tuple], context_length: int, pad_id: int, ignore_index: int
) -> Dict[str, torch.Tensor]:
    input_ids = torch.full((context_length,), pad_id, dtype=torch.long)
    labels = torch.full((context_length,), ignore_index, dtype=torch.long)
    position_ids = torch.zeros(context_length, dtype=torch.long)
    segment_ids = torch.zeros(context_length, dtype=torch.long)
    cursor = 0
    for seg_idx, (ids, mask) in enumerate(segments, start=1):
        n = len(ids)
        seg = torch.tensor(ids, dtype=torch.long)
        seg_labels = seg.masked_fill(torch.tensor(mask, dtype=torch.long) == 0, ignore_index)
        seg_labels[0] = ignore_index
        input_ids[cursor : cursor + n] = seg
        labels[cursor : cursor + n] = seg_labels
        position_ids[cursor : cursor + n] = torch.arange(n)
        segment_ids[cursor : cursor + n] = seg_idx
        cursor += n
    return {
        "input_ids": input_ids,
        "labels": labels,
        "position_ids": position_ids,
        "segment_ids": segment_ids,
        "attention_mask": segment_ids > 0,
    }


def packed_attention_mask(segment_ids: torch.Tensor) -> torch.Tensor:
    """Block-diagonal causal mask from segment ids.

    Args:
        segment_ids: (B, L) or (L,), 0 = padding.

    Returns:
        (B, 1, L, L) bool, True = query may attend key (same non-zero segment and key <= query).
        Padding rows are allowed to attend only themselves (avoids NaN; outputs ignored).
    """
    if segment_ids.dim() == 1:
        segment_ids = segment_ids.unsqueeze(0)
    if segment_ids.dim() != 2:
        raise ValueError(f"segment_ids must be (B, L) or (L,), got {tuple(segment_ids.shape)}")
    length = segment_ids.shape[1]
    same = segment_ids[:, :, None] == segment_ids[:, None, :]
    real = (segment_ids > 0)[:, :, None]
    causal = torch.ones(length, length, dtype=torch.bool, device=segment_ids.device).tril()
    mask = same & real & causal[None]
    eye = torch.eye(length, dtype=torch.bool, device=segment_ids.device)[None]
    mask = mask | (eye & ~real)
    return mask.unsqueeze(1)
