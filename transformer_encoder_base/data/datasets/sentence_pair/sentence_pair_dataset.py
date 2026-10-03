# Purpose: Sentence-pair dataset (NLI, paraphrase, similarity) over explicit text pairs with optional labels; pair tokenization is an injected responsibility.
"""Sentence-pair dataset.

Item schema:
    without ``pair_tokenize_fn``: {"text_a": str, "text_b": str[, "label": int | float]}
    with ``pair_tokenize_fn``:    {**pair_tokenize_fn(text_a, text_b)[, "label": ...]}
        e.g. ``lambda a, b: tok.encode(a, b, max_length=256, truncation=True).to_dict()`` producing
        "input_ids" and "token_type_ids" (0 = first segment, 1 = second segment).
Labels: ``label_type="class"`` -> int class ids (NLI); ``"regression"`` -> float scores (STS);
omit labels entirely for unlabeled inference data.

Requires: torch only.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from torch.utils.data import Dataset

__all__ = ["SentencePairDataset"]

PairTokenizeFn = Callable[[str, str], Mapping[str, Any]]


class SentencePairDataset(Dataset):
    """Pairs of texts with optional class or regression labels."""

    def __init__(
        self,
        first_texts: Sequence[str],
        second_texts: Sequence[str],
        labels: Optional[Sequence[Union[int, float]]] = None,
        label_type: str = "class",
        pair_tokenize_fn: Optional[PairTokenizeFn] = None,
    ) -> None:
        if len(first_texts) != len(second_texts):
            raise ValueError("first_texts and second_texts must have the same length")
        if labels is not None and len(labels) != len(first_texts):
            raise ValueError("labels must have the same length as the texts")
        if label_type not in ("class", "regression"):
            raise ValueError(f"label_type must be 'class' or 'regression', got {label_type!r}")
        self.first_texts = list(first_texts)
        self.second_texts = list(second_texts)
        self.label_type = label_type
        self.labels: Optional[List[Union[int, float]]] = None
        if labels is not None:
            convert = int if label_type == "class" else float
            self.labels = [convert(label) for label in labels]
        self.pair_tokenize_fn = pair_tokenize_fn

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        first_field: str,
        second_field: str,
        label_field: Optional[str] = None,
        label_to_id: Optional[Mapping[str, int]] = None,
        label_type: str = "class",
        pair_tokenize_fn: Optional[PairTokenizeFn] = None,
    ) -> "SentencePairDataset":
        """Build from dict records with explicit field names; string labels need ``label_to_id``."""
        first, second, labels = [], [], []
        for index, record in enumerate(records):
            first.append(str(record[first_field]))
            second.append(str(record[second_field]))
            if label_field is not None:
                raw = record[label_field]
                if label_to_id is not None:
                    if raw not in label_to_id:
                        raise ValueError(f"record {index}: unknown label {raw!r}")
                    raw = label_to_id[raw]
                labels.append(raw)
        return cls(first, second, labels if label_field is not None else None, label_type, pair_tokenize_fn)

    def __len__(self) -> int:
        return len(self.first_texts)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        text_a, text_b = self.first_texts[index], self.second_texts[index]
        if self.pair_tokenize_fn is None:
            item: Dict[str, Any] = {"text_a": text_a, "text_b": text_b}
        else:
            item = dict(self.pair_tokenize_fn(text_a, text_b))
        if self.labels is not None:
            item["label"] = self.labels[index]
        return item
