# Purpose: Collates token-classification (NER/POS) examples by padding inputs with the pad id and token-aligned labels with a separate ignore id (-100).
"""Token classification collator.

Input features: list of dicts with
    "input_ids": list[int] / 1-D tensor (tokenized, special tokens included),
    "labels": list[int] / 1-D tensor of the SAME length, already aligned to tokens
        (see tasks/ner/ner_alignment.py; special tokens / non-first subwords = -100),
    optional "token_type_ids", optional "word_ids" (list[int | None]) metadata.
Output:
    input_ids (B, T) long padded with ``pad_id``;
    attention_mask (B, T) ``mask_dtype`` (1/True = valid token);
    labels (B, T) long padded with ``label_pad_id`` (default -100, ignored by losses);
    token_type_ids (B, T) long padded with 0 (if all features have them);
    word_ids: list of per-example lists padded with None (kept as Python metadata, if present).

Requires: torch only.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

__all__ = ["TokenClassificationCollator"]


def _to_list(values: Any) -> List[int]:
    return values.tolist() if isinstance(values, torch.Tensor) else list(values)


def _pad(rows: Sequence[List[int]], value: int, length: int, side: str) -> torch.Tensor:
    out = torch.full((len(rows), length), value, dtype=torch.long)
    for i, row in enumerate(rows):
        if not row:
            continue
        tensor = torch.as_tensor(row, dtype=torch.long)
        if side == "right":
            out[i, : len(row)] = tensor
        else:
            out[i, length - len(row):] = tensor
    return out


class TokenClassificationCollator:
    """Callable collate_fn for token-level tasks."""

    def __init__(
        self,
        pad_id: int,
        label_pad_id: int = -100,
        padding_side: str = "right",
        pad_to_multiple_of: Optional[int] = None,
        mask_dtype: torch.dtype = torch.long,
    ) -> None:
        if padding_side not in ("right", "left"):
            raise ValueError(f"padding_side must be 'right' or 'left', got {padding_side!r}")
        self.pad_id = pad_id
        self.label_pad_id = label_pad_id
        self.padding_side = padding_side
        self.pad_to_multiple_of = pad_to_multiple_of
        self.mask_dtype = mask_dtype

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        if not features:
            raise ValueError("cannot collate an empty batch")
        ids = [_to_list(f["input_ids"]) for f in features]
        length = max(len(seq) for seq in ids)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            length = ((length + m - 1) // m) * m
        side = self.padding_side
        batch: Dict[str, Any] = {
            "input_ids": _pad(ids, self.pad_id, length, side),
            "attention_mask": _pad([[1] * len(s) for s in ids], 0, length, side).to(self.mask_dtype),
        }
        if all("labels" in f for f in features):
            labels = [_to_list(f["labels"]) for f in features]
            for index, (lab, seq) in enumerate(zip(labels, ids)):
                if len(lab) != len(seq):
                    raise ValueError(
                        f"example {index}: labels length {len(lab)} != input_ids length {len(seq)}; align labels first"
                    )
            batch["labels"] = _pad(labels, self.label_pad_id, length, side)
        elif any("labels" in f for f in features):
            raise ValueError("either all or none of the features must have 'labels'")
        if all("token_type_ids" in f for f in features):
            batch["token_type_ids"] = _pad([_to_list(f["token_type_ids"]) for f in features], 0, length, side)
        if all("word_ids" in f for f in features):
            padded_word_ids = []
            for f in features:
                word_ids = list(f["word_ids"])
                pad = [None] * (length - len(word_ids))
                padded_word_ids.append(word_ids + pad if side == "right" else pad + word_ids)
            batch["word_ids"] = padded_word_ids
        return batch
