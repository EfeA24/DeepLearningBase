# Purpose: Collates instruction-tuning examples so that only response tokens (per a provided loss mask) contribute to the causal-LM loss.
"""Instruction-tuning collator.

Input examples (e.g. from data/datasets/instruction/*.py): mappings with
    "input_ids": list[int] or 1D LongTensor — prompt + response tokens
    "loss_mask": REQUIRED 0/1 per token; 1 = response/assistant token whose label is trained,
                 0 = prompt/system/user token (label -100)

Output (right padding by default):
    input_ids      LongTensor (B, T)
    attention_mask LongTensor (B, T), 1 = real token
    labels         LongTensor (B, T) = input_ids where loss_mask == 1, else -100 (and -100 at pad)
    loss_mask      BoolTensor (B, T)

Label shift reminder: labels are aligned with input_ids; the loss compares logits[:, :-1]
with labels[:, 1:], so the first response token is predicted from the last prompt token.

``drop_examples_without_targets`` controls examples whose loss mask is all zero after
truncation: True -> silently removed from the batch (raise if the batch becomes empty);
False (default) -> ValueError, so data problems are not hidden.

Requires: torch.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch


def _to_list(value: Any) -> List[int]:
    if isinstance(value, torch.Tensor):
        return [int(v) for v in value.reshape(-1).tolist()]
    return [int(v) for v in value]


class InstructionCollator:
    """Pads instruction examples and builds prompt-masked labels.

    Args:
        pad_id: Padding token id.
        ignore_index: Label value for non-trained positions.
        max_length: Optional right truncation length.
        padding_side: "right" or "left".
        drop_examples_without_targets: See module docstring.
    """

    def __init__(
        self,
        pad_id: int,
        ignore_index: int = -100,
        max_length: Optional[int] = None,
        padding_side: str = "right",
        drop_examples_without_targets: bool = False,
    ) -> None:
        if padding_side not in ("right", "left"):
            raise ValueError("padding_side must be 'right' or 'left'")
        if max_length is not None and max_length <= 1:
            raise ValueError("max_length must be > 1")
        self.pad_id = pad_id
        self.ignore_index = ignore_index
        self.max_length = max_length
        self.padding_side = padding_side
        self.drop_examples_without_targets = drop_examples_without_targets

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
        rows = []
        for i, ex in enumerate(examples):
            if "input_ids" not in ex or ex.get("loss_mask") is None:
                raise KeyError(f"Example {i} must contain 'input_ids' and 'loss_mask'")
            ids = _to_list(ex["input_ids"])
            mask = [1 if m else 0 for m in _to_list(ex["loss_mask"])]
            if len(ids) != len(mask):
                raise ValueError(f"Example {i}: input_ids and loss_mask lengths differ")
            if self.max_length is not None:
                ids, mask = ids[: self.max_length], mask[: self.max_length]
            if not any(mask[1:]):
                if self.drop_examples_without_targets:
                    continue
                raise ValueError(f"Example {i} has no trainable (loss_mask == 1) target after position 0")
            rows.append((ids, mask))
        if not rows:
            raise ValueError("No examples with trainable targets in this batch")

        longest = max(len(ids) for ids, _ in rows)
        b = len(rows)
        input_ids = torch.full((b, longest), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((b, longest), dtype=torch.long)
        loss_mask = torch.zeros((b, longest), dtype=torch.bool)
        for i, (ids, mask) in enumerate(rows):
            n = len(ids)
            sl = slice(0, n) if self.padding_side == "right" else slice(longest - n, longest)
            input_ids[i, sl] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, sl] = 1
            loss_mask[i, sl] = torch.tensor(mask, dtype=torch.bool)
        labels = input_ids.masked_fill(~loss_mask, self.ignore_index)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "loss_mask": loss_mask,
        }
