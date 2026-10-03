# Purpose: Turns a token-id stream into language-model input / next-token target pairs for recurrent language models.
"""Token-id stream -> fixed-length LM examples.

Contract for window ``i`` (start ``s = i * stride``):
- ``inputs  = ids[s : s + seq_len]``          shape (seq_len,), int64
- ``targets = ids[s + 1 : s + seq_len + 1]``  shape (seq_len,), int64 (next-token labels)
- Full windows need ``seq_len + 1`` tokens. With ``drop_last=False`` one final partial window
  is added: inputs are right-padded with ``pad_id`` and targets with ``ignore_index``
  (default -100, matching ``F.cross_entropy``), so padded steps do not contribute to loss.
- ``stride < seq_len`` creates overlapping windows; overlapped targets are then counted more
  than once in an epoch average.

Tokenization happens before this dataset; build the vocabulary from training text only.

Requires: torch only. No sibling files.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset


class TextSequenceDataset(Dataset):
    """Lazy next-token (inputs, targets) windows over a 1-D token-id stream."""

    def __init__(
        self,
        token_ids: Union[Sequence[int], torch.Tensor],
        seq_len: int,
        stride: Optional[int] = None,
        drop_last: bool = True,
        pad_id: Optional[int] = None,
        ignore_index: int = -100,
    ) -> None:
        ids = torch.as_tensor(token_ids, dtype=torch.long)
        if ids.dim() != 1:
            raise ValueError(f"token_ids must be 1-D, got shape {tuple(ids.shape)}.")
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}.")
        stride = seq_len if stride is None else stride
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}.")
        if not drop_last and pad_id is None:
            raise ValueError("drop_last=False requires pad_id for padding the partial window.")

        self.token_ids = ids
        self.seq_len = seq_len
        self.stride = stride
        self.pad_id = pad_id
        self.ignore_index = ignore_index

        num_predictable = ids.shape[0] - 1
        full = 0 if num_predictable < seq_len else (num_predictable - seq_len) // stride + 1
        self._starts = [index * stride for index in range(full)]
        if not drop_last:
            covered_end = self._starts[-1] + seq_len if self._starts else 0
            next_start = self._starts[-1] + stride if self._starts else 0
            if covered_end < num_predictable and next_start < num_predictable:
                self._starts.append(next_start)

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self._starts[index]
        inputs = self.token_ids[start : start + self.seq_len]
        targets = self.token_ids[start + 1 : start + self.seq_len + 1]
        missing = self.seq_len - inputs.shape[0]
        if missing > 0:
            inputs = torch.cat([inputs, inputs.new_full((missing,), self.pad_id)])
        missing = self.seq_len - targets.shape[0]
        if missing > 0:
            targets = torch.cat([targets, targets.new_full((missing,), self.ignore_index)])
        return inputs, targets
