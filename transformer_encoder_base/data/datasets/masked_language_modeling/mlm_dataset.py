# Purpose: Masked-language-modeling text datasets (per-text examples and fixed-length token blocks); masking is intentionally left to the MLM collator.
"""MLM datasets. Neither class masks tokens - corruption belongs to data/collators/mlm_collator.py so
every epoch sees fresh masks.

``MLMTextDataset``: one example per text.
    without ``tokenize_fn``: {"text": str}
    with ``tokenize_fn``:    tokenize_fn(text) -> dict with at least "input_ids" (special tokens
        included) and ideally "special_tokens_mask".
``ChunkedTokenDataset``: concatenates already-tokenized documents (lists of ids WITHOUT special
    tokens) and cuts the stream into blocks of ``block_size`` content tokens; optional
    ``cls_id``/``sep_id`` wrap each block ([CLS] block [SEP] -> block_size + 2 ids).
    Items: {"input_ids": list[int], "special_tokens_mask": list[int]}.
    ``drop_last=True`` drops the final short block. Blocks may span document boundaries (common for
    BERT/RoBERTa pre-training); insert a separator id between documents via ``document_separator_id``
    if desired.

Requires: torch only.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from torch.utils.data import Dataset

__all__ = ["MLMTextDataset", "ChunkedTokenDataset"]


class MLMTextDataset(Dataset):
    """Raw or tokenized texts for MLM."""

    def __init__(
        self,
        texts: Sequence[str],
        tokenize_fn: Optional[Callable[[str], Mapping[str, Any]]] = None,
        skip_empty: bool = True,
    ) -> None:
        self.texts = [t for t in texts if t.strip()] if skip_empty else list(texts)
        self.tokenize_fn = tokenize_fn

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        text = self.texts[index]
        if self.tokenize_fn is None:
            return {"text": text}
        item = dict(self.tokenize_fn(text))
        if "input_ids" not in item:
            raise KeyError("tokenize_fn must return a dict containing 'input_ids'")
        return item


class ChunkedTokenDataset(Dataset):
    """Fixed-length blocks cut from a concatenated token stream."""

    def __init__(
        self,
        token_streams: Iterable[Sequence[int]],
        block_size: int,
        drop_last: bool = True,
        cls_id: Optional[int] = None,
        sep_id: Optional[int] = None,
        document_separator_id: Optional[int] = None,
    ) -> None:
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if (cls_id is None) != (sep_id is None):
            raise ValueError("pass both cls_id and sep_id, or neither")
        stream: List[int] = []
        for document in token_streams:
            if stream and document_separator_id is not None:
                stream.append(document_separator_id)
            stream.extend(int(t) for t in document)
        usable = (len(stream) // block_size) * block_size if drop_last else len(stream)
        self.blocks = [stream[i: i + block_size] for i in range(0, usable, block_size)]
        self.cls_id, self.sep_id = cls_id, sep_id

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        block = list(self.blocks[index])
        special = [0] * len(block)
        if self.cls_id is not None:
            block = [self.cls_id] + block + [self.sep_id]
            special = [1] + special + [1]
        return {"input_ids": block, "special_tokens_mask": special}
