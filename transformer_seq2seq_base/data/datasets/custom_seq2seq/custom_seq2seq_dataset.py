# Purpose: Provides an adaptation scaffold for a project-specific seq2seq dataset with explicit source/target record hooks (NotImplementedError).
"""Scaffold for a custom seq2seq dataset.

Subclass and implement the hooks; nothing here reads data or fabricates examples.

    __len__()                 -> number of examples
    load_record(index)        -> one raw record (any type), e.g. a row from caller-provided storage
    get_source_text(record)   -> str
    get_target_text(record)   -> str | None   (None allowed for inference-only data)

``__getitem__`` composes the hooks into the template-wide example contract:
    {"source_text": str, "target_text": str | None,
     "input_ids": list[int]  (if source_encode given),
     "labels":    list[int]  (if target_encode given and target exists; ends with EOS by
                              tokenizer contract, no decoder start token)}
Decoder inputs / padding / -100 labels are produced by data/collators/seq2seq_collator.py.

Requires: torch (Dataset base class).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Dict, Optional

from torch.utils.data import Dataset

EncodeFn = Callable[[str], Sequence[int]]


class CustomSeq2SeqDataset(Dataset):
    """Adaptation scaffold; override the hooks below in a subclass."""

    def __init__(
        self,
        *,
        source_encode: Optional[EncodeFn] = None,
        target_encode: Optional[EncodeFn] = None,
    ) -> None:
        self.source_encode = source_encode
        self.target_encode = target_encode

    def __len__(self) -> int:
        raise NotImplementedError("Return the number of examples in your dataset.")

    def load_record(self, index: int) -> Any:
        raise NotImplementedError(
            "Return the raw record at `index` from storage passed explicitly to __init__."
        )

    def get_source_text(self, record: Any) -> str:
        raise NotImplementedError("Extract the source text (encoder input) from a record.")

    def get_target_text(self, record: Any) -> Optional[str]:
        raise NotImplementedError(
            "Extract the target text from a record, or return None for inference-only data."
        )

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.load_record(index)
        source = self.get_source_text(record)
        target = self.get_target_text(record)
        if not isinstance(source, str):
            raise TypeError(f"get_source_text must return str, got {type(source).__name__}")
        if target is not None and not isinstance(target, str):
            raise TypeError(f"get_target_text must return str or None, got {type(target).__name__}")
        example: Dict[str, Any] = {"source_text": source, "target_text": target}
        if self.source_encode is not None:
            example["input_ids"] = list(self.source_encode(source))
        if self.target_encode is not None and target is not None:
            example["labels"] = list(self.target_encode(target))
        return example
