# Purpose: Collates variable-length sequence examples into padded batch-first tensors with lengths/mask while preserving labels and metadata for LSTM/GRU training.
"""Padded sequence collator.

Contract:
    - Each example is a dict. ``input_key`` holds a ``(T_i,)`` or ``(T_i, F)`` tensor.
    - Output dict:
        ``input_key``  -> ``(B, T_max[, F])`` padded with ``pad_value`` (right padding by default)
        ``"lengths"``  -> ``(B,)`` int64
        ``"mask"``     -> ``(B, T_max)`` bool, True = valid position
      Keys in ``sequence_label_keys`` (per-step labels, e.g. LM/token targets) are padded with
      ``label_pad_value`` (default -100 = ignore_index). Other tensor fields with equal shapes are stacked
      (e.g. class labels); Python scalars become tensors; anything else (strings, ids, dicts) is returned
      as a list unchanged, preserving metadata.
    - Use right padding when the batch is fed through pack_padded_sequence.

Requires: torch only.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

import torch


def _pad(tensors: List[torch.Tensor], value: float, side: str) -> torch.Tensor:
    max_len = max(t.shape[0] for t in tensors)
    out = tensors[0].new_full((len(tensors), max_len, *tensors[0].shape[1:]), value)
    for i, t in enumerate(tensors):
        n = t.shape[0]
        if n == 0:
            continue
        if side == "right":
            out[i, :n] = t
        else:
            out[i, max_len - n:] = t
    return out


class PaddedSequenceCollator:
    """Callable ``collate_fn`` for DataLoader."""

    def __init__(
        self,
        input_key: str = "inputs",
        pad_value: float = 0,
        sequence_label_keys: Sequence[str] = (),
        label_pad_value: float = -100,
        side: str = "right",
    ) -> None:
        if side not in ("right", "left"):
            raise ValueError(f"side must be 'right' or 'left', got {side!r}.")
        self.input_key = input_key
        self.pad_value = pad_value
        self.sequence_label_keys = tuple(sequence_label_keys)
        self.label_pad_value = label_pad_value
        self.side = side

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        if len(examples) == 0:
            raise ValueError("Cannot collate an empty batch.")
        for ex in examples:
            if not isinstance(ex, Mapping):
                raise TypeError(f"Examples must be dicts, got {type(ex).__name__}.")
            if self.input_key not in ex:
                raise KeyError(f"Example missing input key {self.input_key!r}.")

        inputs = [torch.as_tensor(ex[self.input_key]) for ex in examples]
        lengths = torch.tensor([t.shape[0] for t in inputs], dtype=torch.long)
        batch: Dict[str, Any] = {self.input_key: _pad(inputs, self.pad_value, self.side), "lengths": lengths}
        positions = torch.arange(batch[self.input_key].shape[1]).unsqueeze(0)
        if self.side == "right":
            batch["mask"] = positions < lengths.unsqueeze(1)
        else:
            batch["mask"] = positions >= (batch[self.input_key].shape[1] - lengths).unsqueeze(1)

        reserved = {self.input_key, "lengths", "mask"}
        for key in examples[0].keys():
            if key in reserved:
                continue
            values = [ex[key] for ex in examples]
            if key in self.sequence_label_keys:
                seqs = [torch.as_tensor(v) for v in values]
                for s, n in zip(seqs, lengths.tolist()):
                    if s.shape[0] != n:
                        raise ValueError(f"Sequence label {key!r} length {s.shape[0]} != input length {n}.")
                batch[key] = _pad(seqs, self.label_pad_value, self.side)
            elif all(isinstance(v, torch.Tensor) for v in values) and len({tuple(v.shape) for v in values}) == 1:
                batch[key] = torch.stack(values)
            elif all(isinstance(v, (int, float, bool)) for v in values):
                batch[key] = torch.tensor(values)
            else:
                batch[key] = list(values)
        return batch
