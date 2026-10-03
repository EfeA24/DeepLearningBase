# Purpose: Collates variable-length causal-LM examples into padded input_ids, attention_mask, labels (-100 at pad) and optional loss_mask.
"""Causal-LM collator.

Input examples: mappings with
    "input_ids": list[int] or 1D LongTensor (required)
    "labels":    optional, same length; aligned with input_ids (template convention). If absent,
                 labels = input_ids.
    "loss_mask": optional 0/1 per token, same length; 0 -> label -100.

Output batch (B = number of examples, T = padded length):
    input_ids      LongTensor (B, T), pad_id at padding
    attention_mask LongTensor (B, T), 1 = real token, 0 = padding
    labels         LongTensor (B, T), -100 at padding and where loss_mask == 0
    loss_mask      BoolTensor (B, T), only when ``return_loss_mask=True`` (True = scored label;
                   padding False). Note the loss still applies the shift: logits[:, :-1] vs labels[:, 1:].

Padding side defaults to "right" (training). Labels never rely on pad_id (pad_id may equal
eos_id): padding is tracked by position. Examples longer than ``max_length`` are truncated
on the right when ``max_length`` is set.

Requires: torch.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch


def _as_long_list(value: Any, name: str) -> List[int]:
    if isinstance(value, torch.Tensor):
        if value.dim() != 1:
            raise ValueError(f"{name} must be 1D, got shape {tuple(value.shape)}")
        return [int(v) for v in value.tolist()]
    return [int(v) for v in value]


class CausalLMCollator:
    """Pads causal-LM examples into a batch.

    Args:
        pad_id: Token id used to pad ``input_ids``.
        ignore_index: Label value for ignored positions.
        padding_side: "right" (default) or "left".
        max_length: Optional truncation length (right truncation).
        pad_to_multiple_of: Round T up to a multiple of this value.
        return_loss_mask: Include a bool ``loss_mask`` in the batch.
    """

    def __init__(
        self,
        pad_id: int,
        ignore_index: int = -100,
        padding_side: str = "right",
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_loss_mask: bool = False,
    ) -> None:
        if padding_side not in ("right", "left"):
            raise ValueError("padding_side must be 'right' or 'left'")
        if max_length is not None and max_length <= 0:
            raise ValueError("max_length must be positive")
        if pad_to_multiple_of is not None and pad_to_multiple_of <= 0:
            raise ValueError("pad_to_multiple_of must be positive")
        self.pad_id = pad_id
        self.ignore_index = ignore_index
        self.padding_side = padding_side
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_loss_mask = return_loss_mask

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
        if len(examples) == 0:
            raise ValueError("Cannot collate an empty list of examples")
        rows = []
        for i, ex in enumerate(examples):
            if "input_ids" not in ex:
                raise KeyError(f"Example {i} has no 'input_ids'")
            ids = _as_long_list(ex["input_ids"], "input_ids")
            labels = _as_long_list(ex["labels"], "labels") if ex.get("labels") is not None else list(ids)
            mask = _as_long_list(ex["loss_mask"], "loss_mask") if ex.get("loss_mask") is not None else [1] * len(ids)
            if not (len(ids) == len(labels) == len(mask)):
                raise ValueError(f"Example {i}: input_ids/labels/loss_mask lengths differ")
            if self.max_length is not None:
                ids, labels, mask = ids[: self.max_length], labels[: self.max_length], mask[: self.max_length]
            labels = [lab if m else self.ignore_index for lab, m in zip(labels, mask)]
            rows.append((ids, labels, mask))

        longest = max(len(r[0]) for r in rows)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            longest = ((longest + m - 1) // m) * m
        b = len(rows)
        input_ids = torch.full((b, longest), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((b, longest), dtype=torch.long)
        labels_t = torch.full((b, longest), self.ignore_index, dtype=torch.long)
        loss_mask = torch.zeros((b, longest), dtype=torch.bool)
        for i, (ids, labels, mask) in enumerate(rows):
            n = len(ids)
            sl = slice(0, n) if self.padding_side == "right" else slice(longest - n, longest)
            input_ids[i, sl] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, sl] = 1
            labels_t[i, sl] = torch.tensor(labels, dtype=torch.long)
            loss_mask[i, sl] = torch.tensor(labels, dtype=torch.long) != self.ignore_index
        batch = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels_t}
        if self.return_loss_mask:
            batch["loss_mask"] = loss_mask
        return batch
