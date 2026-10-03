# Purpose: Provides a general map-style input/target sequence dataset for LSTM/GRU sequence-to-sequence-aligned or sequence-to-value tasks.
"""Generic sequence dataset over caller-supplied, already prepared examples.

Contract:
    - ``inputs[i]`` is a tensor ``(T_i,)`` (token ids, int64) or ``(T_i, F)`` (features, float).
    - ``targets[i]`` is optional and can be per-step ``(T_i, ...)`` or per-sequence (scalar / ``(K,)``).
      ``aligned_targets=True`` validates that per-step targets share the input length.
    - Each item is a dict ``{"inputs", "length", ["targets"]}``; padding happens in a collator
      (data/collators/padded_sequence_collator.py), not here. Optional ``transform`` is applied to inputs only.

Requires: torch only.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence

import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """Map-style dataset of variable-length sequences with optional targets."""

    def __init__(
        self,
        inputs: Sequence[Any],
        targets: Optional[Sequence[Any]] = None,
        aligned_targets: bool = False,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        input_dtype: Optional[torch.dtype] = None,
        target_dtype: Optional[torch.dtype] = None,
    ) -> None:
        if targets is not None and len(targets) != len(inputs):
            raise ValueError(f"inputs ({len(inputs)}) and targets ({len(targets)}) must have equal length.")
        self.inputs = inputs
        self.targets = targets
        self.aligned_targets = aligned_targets
        self.transform = transform
        self.input_dtype = input_dtype
        self.target_dtype = target_dtype

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        x = torch.as_tensor(self.inputs[index], dtype=self.input_dtype)
        if x.dim() not in (1, 2):
            raise ValueError(f"Input {index} must be (T,) or (T, F); got {tuple(x.shape)}.")
        if self.transform is not None:
            x = self.transform(x)
        item: Dict[str, Any] = {"inputs": x, "length": int(x.shape[0])}
        if self.targets is not None:
            y = torch.as_tensor(self.targets[index], dtype=self.target_dtype)
            if self.aligned_targets and (y.dim() == 0 or y.shape[0] != x.shape[0]):
                raise ValueError(f"Aligned target {index} has shape {tuple(y.shape)} but input length is {x.shape[0]}.")
            item["targets"] = y
        return item
