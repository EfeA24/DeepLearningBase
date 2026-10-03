# Purpose: Builds causal-LM windows that never cross document boundaries, with an explicit padding policy for short windows.
"""Document-preserving causal-LM dataset.

Every window comes from exactly ONE document (no cross-document concatenation). Each document
is tokenized as ``[bos] + tokenize(doc) + [eos]`` (special ids optional) and cut into windows of
``context_length`` with ``stride`` (default: no overlap).

Short windows (documents shorter than L, or a document's tail) follow ``short_window_policy``:
    "pad"  pad to L with pad_id; attention_mask 0 and labels -100 on padding (fixed-size batches)
    "keep" return the shorter window unpadded (use data/collators/causal_lm_collator.py)
    "drop" discard windows shorter than L

Item contract (aligned labels; loss compares logits[:, :-1] with labels[:, 1:]):
    input_ids      LongTensor (L or shorter)
    labels         LongTensor, -100 on padding (and on overlapped tokens if mask_overlap)
    attention_mask LongTensor, 1 = real token
    doc_index      LongTensor scalar, index of the source document

Requires: torch.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

_POLICIES = ("pad", "keep", "drop")


class DocumentLMDataset(Dataset):
    """Per-document causal-LM windows.

    Args:
        documents: Sequence of raw document strings.
        tokenize: Callable ``str -> list[int]``.
        context_length: Window length L (>= 2).
        stride: Step between window starts within a document (default L).
        short_window_policy: "pad", "keep" or "drop".
        pad_id: Required for "pad".
        bos_id, eos_id: Optional special ids added around each document.
        mask_overlap: -100 labels for tokens already scored by the previous window of the same doc.
        ignore_index: Ignored-label value.
    """

    def __init__(
        self,
        documents: Sequence[str],
        tokenize: Callable[[str], List[int]],
        context_length: int,
        stride: Optional[int] = None,
        short_window_policy: str = "pad",
        pad_id: Optional[int] = None,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        mask_overlap: bool = False,
        ignore_index: int = -100,
    ) -> None:
        if context_length < 2:
            raise ValueError(f"context_length must be >= 2, got {context_length}")
        stride = context_length if stride is None else stride
        if not 0 < stride <= context_length:
            raise ValueError(f"stride must be in [1, context_length], got {stride}")
        if short_window_policy not in _POLICIES:
            raise ValueError(f"short_window_policy must be one of {_POLICIES}")
        if short_window_policy == "pad" and pad_id is None:
            raise ValueError("short_window_policy='pad' requires pad_id")
        self.context_length = context_length
        self.policy = short_window_policy
        self.pad_id = pad_id
        self.mask_overlap = mask_overlap
        self.ignore_index = ignore_index

        self.doc_tokens: List[torch.Tensor] = []
        self.index: List[Tuple[int, int, int]] = []
        for doc_idx, doc in enumerate(documents):
            ids = [int(t) for t in tokenize(doc)]
            if bos_id is not None:
                ids = [bos_id] + ids
            if eos_id is not None:
                ids = ids + [eos_id]
            self.doc_tokens.append(torch.tensor(ids, dtype=torch.long))
            n = len(ids)
            if n < 2:
                continue
            prev_end = 0
            start = 0
            while start < n:
                end = min(start + context_length, n)
                if end - start < context_length and short_window_policy == "drop":
                    break
                overlap = max(0, prev_end - start) if mask_overlap else 0
                self.index.append((doc_idx, start, overlap))
                prev_end = end
                if end == n:
                    break
                start += stride

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        doc_idx, start, overlap = self.index[idx]
        window = self.doc_tokens[doc_idx][start : start + self.context_length]
        labels = window.clone()
        labels[:overlap] = self.ignore_index
        attention_mask = torch.ones_like(window)
        n = window.numel()
        if self.policy == "pad" and n < self.context_length:
            pad = self.context_length - n
            window = torch.cat((window, torch.full((pad,), self.pad_id, dtype=torch.long)))
            labels = torch.cat((labels, torch.full((pad,), self.ignore_index, dtype=torch.long)))
            attention_mask = torch.cat((attention_mask, torch.zeros(pad, dtype=torch.long)))
        return {
            "input_ids": window.clone(),
            "labels": labels,
            "attention_mask": attention_mask,
            "doc_index": torch.tensor(doc_idx),
        }
