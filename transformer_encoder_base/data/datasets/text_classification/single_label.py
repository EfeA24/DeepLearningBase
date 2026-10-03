# Purpose: Map-style single-label text classification dataset over explicit texts and integer labels, with an optional injected tokenize function.
"""Single-label text classification dataset.

Item schema:
    without ``tokenize_fn``: {"text": str, "label": int}
    with ``tokenize_fn``:    {**tokenize_fn(text), "label": int}
        ``tokenize_fn`` returns a dict such as {"input_ids": [...], "token_type_ids": [...]} and is
        supplied by the caller (e.g. ``lambda t: tok.encode(t, max_length=128, truncation=True).to_dict()``).
Labels are ints in [0, num_classes). ``from_records`` maps string labels through an explicit
``label_to_id`` mapping (build it from the TRAINING split only and reuse it for val/test).
Pair with data/collators/classification_collator.py.

Requires: torch only.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from torch.utils.data import Dataset

__all__ = ["SingleLabelTextDataset", "build_label_map"]

TokenizeFn = Callable[[str], Mapping[str, Any]]


def build_label_map(labels: Iterable[str]) -> Dict[str, int]:
    """Sorted, stable {label_name: id} mapping. Call on training labels only."""
    return {name: index for index, name in enumerate(sorted(set(labels)))}


class SingleLabelTextDataset(Dataset):
    """Texts + int labels; tokenization optional and injected."""

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        num_classes: Optional[int] = None,
        tokenize_fn: Optional[TokenizeFn] = None,
    ) -> None:
        if len(texts) != len(labels):
            raise ValueError(f"got {len(texts)} texts but {len(labels)} labels")
        int_labels: List[int] = []
        for index, label in enumerate(labels):
            value = int(label)
            if value != label:
                raise TypeError(f"label at index {index} is not an integer: {label!r}")
            if value < 0 or (num_classes is not None and value >= num_classes):
                raise ValueError(f"label {value} at index {index} outside [0, {num_classes})")
            int_labels.append(value)
        self.texts = list(texts)
        self.labels = int_labels
        self.num_classes = num_classes
        self.tokenize_fn = tokenize_fn

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        text_field: str,
        label_field: str,
        label_to_id: Optional[Mapping[str, int]] = None,
        tokenize_fn: Optional[TokenizeFn] = None,
    ) -> "SingleLabelTextDataset":
        """Build from dict records; string labels require ``label_to_id`` (unknown labels raise)."""
        texts: List[str] = []
        labels: List[int] = []
        for index, record in enumerate(records):
            if text_field not in record or label_field not in record:
                raise KeyError(f"record {index} lacks {text_field!r} or {label_field!r}")
            raw = record[label_field]
            if label_to_id is not None:
                if raw not in label_to_id:
                    raise ValueError(f"record {index}: unknown label {raw!r}")
                raw = label_to_id[raw]
            texts.append(str(record[text_field]))
            labels.append(raw)
        num_classes = len(label_to_id) if label_to_id is not None else None
        return cls(texts, labels, num_classes=num_classes, tokenize_fn=tokenize_fn)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        text, label = self.texts[index], self.labels[index]
        if self.tokenize_fn is None:
            return {"text": text, "label": label}
        item = dict(self.tokenize_fn(text))
        item["label"] = label
        return item
