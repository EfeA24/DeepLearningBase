# Purpose: Collates tokenized text-classification examples by padding input_ids/attention_mask/token_type_ids and stacking single-label or multilabel targets.
"""Text classification collator.

Input features: list of dicts with
    "input_ids": list[int] or 1-D long tensor (already tokenized, special tokens included),
    optional "token_type_ids": same length as input_ids,
    optional label under ``label_key``: int (single-label) or sequence/1-D tensor of floats (multilabel).
Output dict (batch-first):
    input_ids (B, T) long, padded with ``pad_id``;
    attention_mask (B, T) ``mask_dtype`` (default long; 1/True = valid token, 0/False = pad);
    token_type_ids (B, T) long padded with 0 (only if every feature has them);
    labels: (B,) long for int labels or (B, C) float for multilabel vectors (only if present).
T = longest sequence, optionally rounded up to ``pad_to_multiple_of``. Sequences longer than
``max_length`` raise unless ``truncate=True`` (truncation keeps the head and may drop a final [SEP];
prefer truncating in the tokenizer).

Requires: torch only.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

__all__ = ["ClassificationCollator", "pad_sequences"]


def pad_sequences(
    sequences: Sequence[Sequence[int]],
    pad_value: int,
    padding_side: str = "right",
    pad_to_multiple_of: Optional[int] = None,
) -> torch.Tensor:
    """Pad integer sequences into a (B, T) long tensor."""
    if padding_side not in ("right", "left"):
        raise ValueError(f"padding_side must be 'right' or 'left', got {padding_side!r}")
    tensors = [torch.as_tensor(seq, dtype=torch.long).flatten() for seq in sequences]
    length = max((t.numel() for t in tensors), default=0)
    if pad_to_multiple_of:
        length = ((length + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
    out = torch.full((len(tensors), length), pad_value, dtype=torch.long)
    for row, tensor in enumerate(tensors):
        n = tensor.numel()
        if padding_side == "right":
            out[row, :n] = tensor
        elif n > 0:
            out[row, length - n:] = tensor
    return out


class ClassificationCollator:
    """Callable collate_fn for DataLoader."""

    def __init__(
        self,
        pad_id: int,
        label_key: str = "label",
        padding_side: str = "right",
        pad_to_multiple_of: Optional[int] = None,
        max_length: Optional[int] = None,
        truncate: bool = False,
        mask_dtype: torch.dtype = torch.long,
    ) -> None:
        if mask_dtype not in (torch.long, torch.bool, torch.int32):
            raise ValueError("mask_dtype must be torch.long, torch.int32 or torch.bool")
        self.pad_id = pad_id
        self.label_key = label_key
        self.padding_side = padding_side
        self.pad_to_multiple_of = pad_to_multiple_of
        self.max_length = max_length
        self.truncate = truncate
        self.mask_dtype = mask_dtype

    def _clip(self, seq: Any) -> List[int]:
        values = seq.tolist() if isinstance(seq, torch.Tensor) else list(seq)
        if self.max_length is not None and len(values) > self.max_length:
            if not self.truncate:
                raise ValueError(f"sequence length {len(values)} > max_length={self.max_length}")
            values = values[: self.max_length]
        return values

    def _labels(self, features: Sequence[Mapping[str, Any]]) -> Optional[torch.Tensor]:
        present = [self.label_key in f for f in features]
        if not any(present):
            return None
        if not all(present):
            raise ValueError(f"some features lack {self.label_key!r}")
        labels = [f[self.label_key] for f in features]
        first = labels[0]
        is_vector = isinstance(first, (list, tuple)) or (isinstance(first, torch.Tensor) and first.dim() > 0)
        if not is_vector:
            return torch.as_tensor([int(l) for l in labels], dtype=torch.long)
        vectors = [torch.as_tensor(l, dtype=torch.float32).flatten() for l in labels]
        if len({v.numel() for v in vectors}) != 1:
            raise ValueError("multilabel target vectors must all have the same length")
        return torch.stack(vectors)

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
        if not features:
            raise ValueError("cannot collate an empty batch")
        ids = [self._clip(f["input_ids"]) for f in features]
        batch: Dict[str, torch.Tensor] = {
            "input_ids": pad_sequences(ids, self.pad_id, self.padding_side, self.pad_to_multiple_of)
        }
        ones = [[1] * len(seq) for seq in ids]
        batch["attention_mask"] = pad_sequences(ones, 0, self.padding_side, self.pad_to_multiple_of).to(self.mask_dtype)
        if all("token_type_ids" in f for f in features):
            types = [self._clip(f["token_type_ids"]) for f in features]
            if any(len(t) != len(i) for t, i in zip(types, ids)):
                raise ValueError("token_type_ids length must match input_ids length")
            batch["token_type_ids"] = pad_sequences(types, 0, self.padding_side, self.pad_to_multiple_of)
        labels = self._labels(features)
        if labels is not None:
            batch["labels"] = labels
        return batch
