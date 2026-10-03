# Purpose: Collates variable-length sequence examples into padded inputs, ignore-index-padded targets, lengths and validity masks.
"""Padding collator for variable-length sequences.

Example formats (all examples in a batch use the same one):
- tuple/list ``(inputs, targets)`` or ``(inputs,)``;
- dict with ``input_key`` and optionally ``target_key``; other keys are kept as lists.
``inputs``: (T_i,) ids or (T_i, F) features (list or tensor). ``targets``: either per-step
(T_i, ...) sequences or one scalar label per example.

Output dict (batch-first):
- ``inputs``  (B, T_max[, F])  padded with ``input_pad_value`` (pad id for tokens);
- ``targets`` (B, T_max[, ...]) padded with ``ignore_index`` (default -100) for per-step
  targets, or (B,) stacked for scalar labels; absent when examples have no targets;
- ``lengths`` (B,) int64 input lengths;
- ``mask``    (B, T_max) bool, True = valid position, False = padding;
- any extra dict fields as lists.
``padding_side`` ("right"/"left") applies to inputs, targets and mask together.

Requires: torch only. Self-contained padding (same contract as data/sequences/padding.py).
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch


def _as_tensor(value: Any, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    return value if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=dtype)


def _pad(
    tensors: List[torch.Tensor], value: Union[int, float], padding_side: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = [t.shape[0] for t in tensors]
    max_len = max(lengths)
    trailing = tensors[0].shape[1:]
    for index, tensor in enumerate(tensors):
        if tensor.shape[1:] != trailing:
            raise ValueError(f"Example {index} has trailing shape {tuple(tensor.shape[1:])}, expected {tuple(trailing)}.")
    out = tensors[0].new_full((len(tensors), max_len, *trailing), value)
    mask = torch.zeros((len(tensors), max_len), dtype=torch.bool, device=out.device)
    for index, (tensor, length) in enumerate(zip(tensors, lengths)):
        if padding_side == "right":
            out[index, :length] = tensor
            mask[index, :length] = True
        else:
            out[index, max_len - length :] = tensor
            mask[index, max_len - length :] = True
    return out, mask


class PaddedCollator:
    """collate_fn producing padded inputs/targets, lengths and a True=valid mask."""

    def __init__(
        self,
        input_pad_value: Union[int, float] = 0,
        ignore_index: int = -100,
        padding_side: str = "right",
        input_key: str = "inputs",
        target_key: str = "targets",
    ) -> None:
        if padding_side not in ("right", "left"):
            raise ValueError(f"padding_side must be 'right' or 'left', got {padding_side!r}.")
        self.input_pad_value = input_pad_value
        self.ignore_index = ignore_index
        self.padding_side = padding_side
        self.input_key = input_key
        self.target_key = target_key

    def _split(self, example: Any) -> Tuple[Any, Any, Dict[str, Any]]:
        if isinstance(example, Mapping):
            if self.input_key not in example:
                raise KeyError(f"Example is missing input key {self.input_key!r}.")
            extras = {k: v for k, v in example.items() if k not in (self.input_key, self.target_key)}
            return example[self.input_key], example.get(self.target_key), extras
        if isinstance(example, (tuple, list)) and len(example) in (1, 2):
            return example[0], (example[1] if len(example) == 2 else None), {}
        raise TypeError("Examples must be dicts or (inputs[, targets]) tuples.")

    def __call__(self, batch: Sequence[Any]) -> Dict[str, Any]:
        if len(batch) == 0:
            raise ValueError("Cannot collate an empty batch.")
        parts = [self._split(example) for example in batch]
        inputs = [_as_tensor(p[0]) for p in parts]
        for index, tensor in enumerate(inputs):
            if tensor.dim() == 0:
                raise ValueError(f"Example {index}: inputs must have a time dimension.")
        padded_inputs, mask = _pad(inputs, self.input_pad_value, self.padding_side)
        result: Dict[str, Any] = {
            "inputs": padded_inputs,
            "lengths": torch.tensor([t.shape[0] for t in inputs], dtype=torch.long),
            "mask": mask,
        }

        raw_targets = [p[1] for p in parts]
        has_targets = [t is not None for t in raw_targets]
        if any(has_targets):
            if not all(has_targets):
                raise ValueError("Either every example or no example must provide targets.")
            targets = [_as_tensor(t) for t in raw_targets]
            if all(t.dim() == 0 for t in targets):
                result["targets"] = torch.stack(targets)
            else:
                for index, (target, source) in enumerate(zip(targets, inputs)):
                    if target.dim() == 0 or target.shape[0] != source.shape[0]:
                        raise ValueError(f"Example {index}: per-step targets must match the input length.")
                result["targets"], _ = _pad(targets, self.ignore_index, self.padding_side)

        extra_keys = list(parts[0][2].keys())
        for key in extra_keys:
            result[key] = [p[2].get(key) for p in parts]
        return result
