# Purpose: Adapts caller-supplied document/summary records into seq2seq summarization examples with explicit source truncation and target-length policy.
"""Summarization dataset over in-memory records.

Fields are passed explicitly (``document_field``, ``summary_field``). Length policy:
    source: token ids from ``source_encode`` are cut to ``max_source_length`` from the
            right (document start is kept). If ``eos_token_id`` is given and the encoded
            source ended with EOS, the last kept token is replaced by EOS.
    target: if the summary exceeds ``max_target_length`` then
            ``target_overflow="truncate"`` cuts it (EOS restored like the source) and
            ``target_overflow="error"`` raises ValueError (useful for auditing data).
Length limits require the corresponding encode callable; raw-text mode never truncates.

``__getitem__`` returns "source_text"/"target_text" (if ``return_text``), "input_ids" and
"labels" (lists of ints) when encoders are supplied.

Requires: torch (Dataset base class). Tokenizers are injected callables.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Dict, List, Optional

from torch.utils.data import Dataset

EncodeFn = Callable[[str], Sequence[int]]


def _truncate_keep_eos(ids: List[int], max_length: int, eos_token_id: Optional[int]) -> List[int]:
    if len(ids) <= max_length:
        return ids
    ended_with_eos = eos_token_id is not None and ids and ids[-1] == eos_token_id
    cut = ids[:max_length]
    if ended_with_eos and cut:
        cut[-1] = eos_token_id
    return cut


class SummarizationDataset(Dataset):
    """Map-style summarization dataset.

    Args:
        records: sequence of mappings.
        document_field / summary_field: explicit text keys.
        source_encode / target_encode: optional text -> ids callables (own special tokens).
        max_source_length / max_target_length: token limits (None = no limit).
        target_overflow: "truncate" or "error" for over-long summaries.
        eos_token_id: EOS id restored at the cut position (None = plain cut).
        require_target: raise if the summary is missing.
        return_text: include raw texts.
    """

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        document_field: str,
        summary_field: str,
        source_encode: Optional[EncodeFn] = None,
        target_encode: Optional[EncodeFn] = None,
        max_source_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        target_overflow: str = "truncate",
        eos_token_id: Optional[int] = None,
        require_target: bool = True,
        return_text: bool = True,
    ) -> None:
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise TypeError("records must be a sequence of mappings")
        if not document_field or not summary_field or document_field == summary_field:
            raise ValueError("document_field and summary_field must be distinct non-empty strings")
        if target_overflow not in ("truncate", "error"):
            raise ValueError(f"target_overflow must be 'truncate' or 'error', got {target_overflow!r}")
        if max_source_length is not None and (source_encode is None or max_source_length < 1):
            raise ValueError("max_source_length requires source_encode and must be >= 1")
        if max_target_length is not None and (target_encode is None or max_target_length < 1):
            raise ValueError("max_target_length requires target_encode and must be >= 1")
        if not return_text and source_encode is None:
            raise ValueError("return_text=False requires source_encode, otherwise examples are empty")
        self.records = records
        self.document_field = document_field
        self.summary_field = summary_field
        self.source_encode = source_encode
        self.target_encode = target_encode
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.target_overflow = target_overflow
        self.eos_token_id = eos_token_id
        self.require_target = require_target
        self.return_text = return_text

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        if not isinstance(record, Mapping):
            raise TypeError(f"record {index} must be a mapping, got {type(record).__name__}")
        document = record.get(self.document_field)
        if not isinstance(document, str):
            raise KeyError(f"record {index}: {self.document_field!r} missing or not a string")
        summary = record.get(self.summary_field)
        if summary is not None and not isinstance(summary, str):
            raise TypeError(f"record {index}: {self.summary_field!r} must be a string")
        if summary is None and self.require_target:
            raise KeyError(f"record {index}: {self.summary_field!r} is missing")

        example: Dict[str, Any] = {}
        if self.return_text:
            example["source_text"] = document
            example["target_text"] = summary
        if self.source_encode is not None:
            ids = [int(t) for t in self.source_encode(document)]
            if self.max_source_length is not None:
                ids = _truncate_keep_eos(ids, self.max_source_length, self.eos_token_id)
            example["input_ids"] = ids
        if self.target_encode is not None and summary is not None:
            labels = [int(t) for t in self.target_encode(summary)]
            if self.max_target_length is not None and len(labels) > self.max_target_length:
                if self.target_overflow == "error":
                    raise ValueError(
                        f"record {index}: summary has {len(labels)} tokens > "
                        f"max_target_length={self.max_target_length}"
                    )
                labels = _truncate_keep_eos(labels, self.max_target_length, self.eos_token_id)
            example["labels"] = labels
        return example
