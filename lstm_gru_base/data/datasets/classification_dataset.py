# Purpose: Provides a sequence classification dataset for numeric feature sequences or tokenized text consumed by LSTM/GRU classifiers.
"""Sequence classification dataset.

Contract:
    - ``sequences[i]``: token ids ``(T_i,)`` int64, or numeric features ``(T_i, F)`` float, or a raw string
      when ``encode`` (``str -> list[int]``) is supplied.
    - ``labels[i]``: int class id in ``[0, num_classes)``.
    - Items: ``{"inputs": Tensor, "length": int, "label": int64 scalar tensor}``; padding/masking is
      done by the collator.

Requires: torch only.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset


class SequenceClassificationDataset(Dataset):
    """Variable-length sequences with one integer label each."""

    def __init__(
        self,
        sequences: Sequence[Any],
        labels: Sequence[int],
        num_classes: int,
        encode: Optional[Callable[[str], List[int]]] = None,
        input_dtype: Optional[torch.dtype] = None,
        max_length: Optional[int] = None,
    ) -> None:
        if len(sequences) != len(labels):
            raise ValueError(f"sequences ({len(sequences)}) and labels ({len(labels)}) must have equal length.")
        if num_classes < 1:
            raise ValueError("num_classes must be >= 1.")
        for i, y in enumerate(labels):
            if not 0 <= int(y) < num_classes:
                raise ValueError(f"Label {y} at index {i} outside [0, {num_classes}).")
        self.sequences = sequences
        self.labels = labels
        self.num_classes = num_classes
        self.encode = encode
        self.input_dtype = input_dtype
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        raw = self.sequences[index]
        if isinstance(raw, str):
            if self.encode is None:
                raise ValueError("Raw string examples require an encode callable.")
            x = torch.tensor(self.encode(raw), dtype=torch.long)
        else:
            x = torch.as_tensor(raw, dtype=self.input_dtype)
        if self.max_length is not None:
            x = x[: self.max_length]
        if x.shape[0] == 0:
            raise ValueError(f"Example {index} is empty after encoding/truncation.")
        return {"inputs": x, "length": int(x.shape[0]), "label": torch.tensor(int(self.labels[index]), dtype=torch.long)}
