# Purpose: Builds per-example and padded-batch teacher-forcing pairs (decoder inputs and labels) for seq2seq text-to-text training data.
"""Teacher-forcing pair construction at the data (token-list) level.

Convention for one example with target tokens ``y1..yn``:
    labels            = [y1, ..., yn, EOS]
    decoder_input_ids = [START, y1, ..., yn]        (same length as labels)
so the decoder at step t sees tokens < t and predicts labels[t].

Batch output (right padding):
    decoder_input_ids      (B, T) int64 = shift_right(padded labels); padding = ``pad_token_id``
                           except the first pad slot, which holds the shifted EOS (its label is -100)
    labels                 (B, T) int64, pad = ``ignore_index`` (-100)
    decoder_attention_mask (B, T) int64, 1 = real position, 0 = pad

For already-padded tensors use training/teacher_forcing.py or data/sequences/target_shift.py.

Requires: torch only.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


@dataclass(frozen=True)
class TeacherForcingExample:
    """Decoder inputs and labels for one target sequence (equal lengths)."""

    decoder_input_ids: List[int]
    labels: List[int]


def make_teacher_forcing_example(
    target_ids: Sequence[int],
    *,
    decoder_start_token_id: int,
    eos_token_id: Optional[int] = None,
    append_eos: bool = True,
) -> TeacherForcingExample:
    """Create one teacher-forcing pair.

    Args:
        target_ids: target token ids WITHOUT the decoder start token.
        decoder_start_token_id: id prepended to the decoder inputs.
        eos_token_id: EOS id appended to labels when ``append_eos`` and not already last.
        append_eos: whether labels must end with EOS (requires ``eos_token_id``).
    """
    labels = [int(t) for t in target_ids]
    if append_eos:
        if eos_token_id is None:
            raise ValueError("append_eos=True requires eos_token_id")
        if not labels or labels[-1] != eos_token_id:
            labels.append(int(eos_token_id))
    if not labels:
        raise ValueError("target_ids is empty and no EOS was appended; cannot build a pair")
    decoder_input_ids = [int(decoder_start_token_id)] + labels[:-1]
    return TeacherForcingExample(decoder_input_ids=decoder_input_ids, labels=labels)


def make_teacher_forcing_batch(
    targets: Sequence[Sequence[int]],
    *,
    decoder_start_token_id: int,
    pad_token_id: int,
    eos_token_id: Optional[int] = None,
    append_eos: bool = True,
    ignore_index: int = -100,
) -> Dict[str, torch.Tensor]:
    """Create a right-padded teacher-forcing batch from variable-length targets."""
    if len(targets) == 0:
        raise ValueError("targets must contain at least one sequence")
    examples = [
        make_teacher_forcing_example(
            t,
            decoder_start_token_id=decoder_start_token_id,
            eos_token_id=eos_token_id,
            append_eos=append_eos,
        )
        for t in targets
    ]
    max_len = max(len(e.labels) for e in examples)
    batch_size = len(examples)
    labels = torch.full((batch_size, max_len), ignore_index, dtype=torch.long)
    decoder_attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    for row, example in enumerate(examples):
        n = len(example.labels)
        labels[row, :n] = torch.tensor(example.labels, dtype=torch.long)
        decoder_attention_mask[row, :n] = 1
    # Must equal the collator's shift_right output, including pad-query positions.
    decoder_input_ids = labels.new_empty(labels.shape)
    decoder_input_ids[:, 1:] = labels[:, :-1]
    decoder_input_ids[:, 0] = decoder_start_token_id
    decoder_input_ids = decoder_input_ids.masked_fill(decoder_input_ids == ignore_index, pad_token_id)
    return {
        "decoder_input_ids": decoder_input_ids,
        "labels": labels,
        "decoder_attention_mask": decoder_attention_mask,
    }
