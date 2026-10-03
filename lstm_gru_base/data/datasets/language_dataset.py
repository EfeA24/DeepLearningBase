# Purpose: Builds next-token language-modelling windows from a token-id stream with an explicit shift/window policy for LSTM/GRU LMs.
"""Language-modelling dataset over a pre-tokenized id stream.

Contract:
    - ``token_ids``: 1-D int64 tensor/list of the concatenated TRAINING (or eval) stream; the caller
      decides whether documents are joined with EOS.
    - Window ``i`` starts at ``i * stride`` and yields ``input_ids = ids[s : s + seq_len]`` and
      ``labels = ids[s + 1 : s + seq_len + 1]`` (shift by one; no padding, so no ignore_index needed).
    - ``stride == seq_len`` gives non-overlapping windows (typical for TBPTT with state carry);
      ``stride < seq_len`` overlaps, so overlapping targets are counted more than once.
    - A trailing remainder shorter than ``seq_len + 1`` tokens is dropped.

Requires: torch only.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import torch
from torch.utils.data import Dataset


class LanguageModelingDataset(Dataset):
    """Fixed-length shifted windows over a token stream."""

    def __init__(
        self,
        token_ids: Union[torch.Tensor, Sequence[int]],
        seq_len: int,
        stride: Optional[int] = None,
    ) -> None:
        ids = torch.as_tensor(token_ids, dtype=torch.long)
        if ids.dim() != 1:
            raise ValueError(f"token_ids must be 1-D, got shape {tuple(ids.shape)}.")
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}.")
        stride = seq_len if stride is None else stride
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}.")
        if ids.numel() < seq_len + 1:
            raise ValueError(f"Need at least seq_len + 1 = {seq_len + 1} tokens, got {ids.numel()}.")
        self.token_ids = ids
        self.seq_len = seq_len
        self.stride = stride
        self._num_windows = (ids.numel() - (seq_len + 1)) // stride + 1

    def __len__(self) -> int:
        return self._num_windows

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if index < 0:
            index += self._num_windows
        if not 0 <= index < self._num_windows:
            raise IndexError(f"Index {index} out of range for {self._num_windows} windows.")
        s = index * self.stride
        chunk = self.token_ids[s : s + self.seq_len + 1]
        return {"input_ids": chunk[:-1], "labels": chunk[1:]}


def batchify_stream(token_ids: Union[torch.Tensor, Sequence[int]], batch_size: int) -> torch.Tensor:
    """Reshape a stream into ``(B, N)`` parallel streams for stateful TBPTT (remainder dropped).

    Row ``b`` is a contiguous slice of the stream, so hidden state can be carried across chunks
    taken along dim 1 (see training/truncated_bptt.py).
    """
    ids = torch.as_tensor(token_ids, dtype=torch.long)
    if ids.dim() != 1:
        raise ValueError("token_ids must be 1-D.")
    n = ids.numel() // batch_size
    if n < 2:
        raise ValueError(f"Stream of {ids.numel()} tokens is too short for batch_size={batch_size}.")
    return ids[: n * batch_size].view(batch_size, n)
