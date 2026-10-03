# Purpose: Adapts caller-supplied translation records into seq2seq examples with explicit source/target field names and translation direction.
"""Translation dataset over in-memory records.

Records are mappings, either flat ``{"en": "...", "de": "..."}`` or nested
``{"translation": {"en": "...", "de": "..."}}`` (pass ``nested_field="translation"``).
The direction is exactly ``source_field -> target_field``; no automatic reversal.

``__getitem__`` returns:
    "source_text", "target_text"   when ``return_text`` (target may be None if not required)
    "input_ids"  (list[int])        when ``source_encode`` is supplied
    "labels"     (list[int])        when ``target_encode`` is supplied and a target exists
The encode callables own special tokens (e.g. SourceTokenizer adds EOS, TargetTokenizer
adds EOS to labels). Decoder inputs and padding are built by the collator.

Requires: torch (Dataset base class). Tokenizers are injected callables.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Dict, List, Optional, Tuple

from torch.utils.data import Dataset

EncodeFn = Callable[[str], Sequence[int]]


class TranslationDataset(Dataset):
    """Map-style translation dataset.

    Args:
        records: sequence of mappings (kept by reference, not copied).
        source_field: key of the source-language text (e.g. "en").
        target_field: key of the target-language text (e.g. "de").
        nested_field: optional key holding the language dict (e.g. "translation").
        source_encode / target_encode: optional text -> ids callables.
        require_target: raise if a record lacks the target text (False for test-time input).
        return_text: include raw texts in each example.
    """

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        source_field: str,
        target_field: str,
        nested_field: Optional[str] = None,
        source_encode: Optional[EncodeFn] = None,
        target_encode: Optional[EncodeFn] = None,
        require_target: bool = True,
        return_text: bool = True,
    ) -> None:
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise TypeError("records must be a sequence of mappings")
        if not source_field or not target_field:
            raise ValueError("source_field and target_field must be non-empty strings")
        if source_field == target_field:
            raise ValueError("source_field and target_field must differ")
        if not return_text and source_encode is None:
            raise ValueError("return_text=False requires source_encode, otherwise examples are empty")
        self.records = records
        self.source_field = source_field
        self.target_field = target_field
        self.nested_field = nested_field
        self.source_encode = source_encode
        self.target_encode = target_encode
        self.require_target = require_target
        self.return_text = return_text

    @property
    def direction(self) -> str:
        return f"{self.source_field}->{self.target_field}"

    def __len__(self) -> int:
        return len(self.records)

    def _fields(self, index: int) -> Mapping[str, Any]:
        record = self.records[index]
        if not isinstance(record, Mapping):
            raise TypeError(f"record {index} must be a mapping, got {type(record).__name__}")
        if self.nested_field is None:
            return record
        nested = record.get(self.nested_field)
        if not isinstance(nested, Mapping):
            raise KeyError(f"record {index} has no mapping under {self.nested_field!r}")
        return nested

    def get_pair(self, index: int) -> Tuple[str, Optional[str]]:
        """Return (source_text, target_text or None) for ``index``."""
        fields = self._fields(index)
        source = fields.get(self.source_field)
        if not isinstance(source, str):
            raise KeyError(f"record {index}: source field {self.source_field!r} missing or not a string")
        target = fields.get(self.target_field)
        if target is not None and not isinstance(target, str):
            raise TypeError(f"record {index}: target field {self.target_field!r} must be a string")
        if target is None and self.require_target:
            raise KeyError(f"record {index}: target field {self.target_field!r} is missing")
        return source, target

    def __getitem__(self, index: int) -> Dict[str, Any]:
        source, target = self.get_pair(index)
        example: Dict[str, Any] = {}
        if self.return_text:
            example["source_text"] = source
            example["target_text"] = target
        if self.source_encode is not None:
            example["input_ids"] = list(self.source_encode(source))
        if self.target_encode is not None and target is not None:
            example["labels"] = list(self.target_encode(target))
        return example

    def source_texts(self) -> List[str]:
        """All source texts (e.g. for fitting a tokenizer on the TRAINING split only)."""
        return [self.get_pair(i)[0] for i in range(len(self))]
