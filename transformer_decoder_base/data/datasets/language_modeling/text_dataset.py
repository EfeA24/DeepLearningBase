# Purpose: Tokenizes supplied texts with a caller tokenizer and chunks the concatenated stream into causal-LM windows with explicit stride.
"""Causal-LM dataset over a concatenated token stream.

Construction (explicit, happens in ``__init__``, never at import):
    stream = tokenize(text_0) [+ eos] + tokenize(text_1) [+ eos] + ...
    windows start at 0, stride, 2*stride, ... and have exactly ``context_length`` tokens
    (a shorter final window is kept only with ``keep_last_partial=True``).

Texts ARE concatenated across boundaries (standard GPT pre-training packing); pass
``separator_id`` (usually EOS) so the model sees document breaks. Use
document_dataset.py if windows must not cross documents.

Item contract (aligned-label convention; loss compares logits[:, :-1] with labels[:, 1:]):
    {"input_ids": LongTensor (L,), "labels": LongTensor (L,)}
With ``mask_overlap=True`` labels of tokens already scored by the previous window are -100
(overlap is only context; every token is scored once). Fit any tokenizer on TRAINING text only.

Requires: torch.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional

import torch
from torch.utils.data import Dataset


class CausalLMTextDataset(Dataset):
    """Fixed-length windows over the concatenation of tokenized texts.

    Args:
        texts: Iterable of raw strings (consumed once in __init__).
        tokenize: Callable ``str -> list[int]`` (e.g. ``tokenizer.encode``).
        context_length: Window length L (>= 2).
        stride: Step between window starts (default L = no overlap); must be in [1, L].
        separator_id: Optional id appended after every text (e.g. EOS).
        mask_overlap: Set labels of overlapping (already scored) tokens to -100.
        keep_last_partial: Keep a final window shorter than L (needs a padding collator).
        ignore_index: Ignored-label value.
    """

    def __init__(
        self,
        texts: Iterable[str],
        tokenize: Callable[[str], List[int]],
        context_length: int,
        stride: Optional[int] = None,
        separator_id: Optional[int] = None,
        mask_overlap: bool = False,
        keep_last_partial: bool = False,
        ignore_index: int = -100,
    ) -> None:
        if context_length < 2:
            raise ValueError(f"context_length must be >= 2, got {context_length}")
        stride = context_length if stride is None else stride
        if not 0 < stride <= context_length:
            raise ValueError(f"stride must be in [1, context_length], got {stride}")
        stream: List[int] = []
        for text in texts:
            if not isinstance(text, str):
                raise TypeError(f"texts must contain str, got {type(text).__name__}")
            stream.extend(int(t) for t in tokenize(text))
            if separator_id is not None:
                stream.append(separator_id)
        self.tokens = torch.tensor(stream, dtype=torch.long)
        self.context_length = context_length
        self.stride = stride
        self.mask_overlap = mask_overlap
        self.ignore_index = ignore_index

        n = len(stream)
        starts = list(range(0, max(n - context_length, 0) + 1, stride)) if n >= context_length else []
        if keep_last_partial and n > 0:
            last_end = (starts[-1] + context_length) if starts else 0
            if last_end < n:
                starts.append(starts[-1] + stride if starts else 0)
        self.starts = starts

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        start = self.starts[index]
        window = self.tokens[start : start + self.context_length]
        labels = window.clone()
        if self.mask_overlap and index > 0:
            prev_end = self.starts[index - 1] + self.context_length
            overlap = max(0, min(prev_end - start, labels.numel()))
            labels[:overlap] = self.ignore_index
        return {"input_ids": window.clone(), "labels": labels}
