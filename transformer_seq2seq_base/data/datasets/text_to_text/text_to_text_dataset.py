# Purpose: Provides a generic paired-text seq2seq dataset (input_field/target_field with optional task prefix) for text-to-text projects.
"""Generic text-to-text dataset (T5-style task framing).

Each record is a mapping with ``input_field`` and ``target_field`` strings. The optional
``prefix`` (e.g. "summarize: ") is prepended to the INPUT text before encoding; it is never
added to targets.

``__getitem__`` returns "source_text" (prefixed) / "target_text" when ``return_text``,
and "input_ids" / "labels" (lists of ints) when the encode callables are supplied.

Requires: torch (Dataset base class). Tokenizers are injected callables.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Dict, Optional

from torch.utils.data import Dataset

EncodeFn = Callable[[str], Sequence[int]]


class TextToTextDataset(Dataset):
    """Map-style paired-text dataset.

    Args:
        records: sequence of mappings.
        input_field / target_field: explicit text keys.
        prefix: string prepended to every input text ("" = none).
        source_encode / target_encode: optional text -> ids callables.
        require_target: raise if the target is missing.
        return_text: include raw texts.
    """

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        input_field: str,
        target_field: str,
        prefix: str = "",
        source_encode: Optional[EncodeFn] = None,
        target_encode: Optional[EncodeFn] = None,
        require_target: bool = True,
        return_text: bool = True,
    ) -> None:
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise TypeError("records must be a sequence of mappings")
        if not input_field or not target_field or input_field == target_field:
            raise ValueError("input_field and target_field must be distinct non-empty strings")
        if not isinstance(prefix, str):
            raise TypeError("prefix must be a string")
        if not return_text and source_encode is None:
            raise ValueError("return_text=False requires source_encode, otherwise examples are empty")
        self.records = records
        self.input_field = input_field
        self.target_field = target_field
        self.prefix = prefix
        self.source_encode = source_encode
        self.target_encode = target_encode
        self.require_target = require_target
        self.return_text = return_text

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        if not isinstance(record, Mapping):
            raise TypeError(f"record {index} must be a mapping, got {type(record).__name__}")
        text = record.get(self.input_field)
        if not isinstance(text, str):
            raise KeyError(f"record {index}: {self.input_field!r} missing or not a string")
        target = record.get(self.target_field)
        if target is not None and not isinstance(target, str):
            raise TypeError(f"record {index}: {self.target_field!r} must be a string")
        if target is None and self.require_target:
            raise KeyError(f"record {index}: {self.target_field!r} is missing")

        source = self.prefix + text
        example: Dict[str, Any] = {}
        if self.return_text:
            example["source_text"] = source
            example["target_text"] = target
        if self.source_encode is not None:
            example["input_ids"] = list(self.source_encode(source))
        if self.target_encode is not None and target is not None:
            example["labels"] = list(self.target_encode(target))
        return example
