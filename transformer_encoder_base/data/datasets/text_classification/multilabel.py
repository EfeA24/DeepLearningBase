# Purpose: Map-style multilabel text classification dataset with a stable, caller-defined class order and float multi-hot target vectors.
"""Multilabel text classification dataset.

Class order: exactly the order of ``class_names`` passed by the caller; target index i corresponds to
``class_names[i]``. Keep this list fixed across train/val/test and inference.
Item schema:
    without ``tokenize_fn``: {"text": str, "labels": FloatTensor (C,) in {0, 1}}
    with ``tokenize_fn``:    {**tokenize_fn(text), "labels": FloatTensor (C,)}
An example may have zero active labels (all-zero vector). Unknown label names raise ValueError.
Use with BCE-with-logits (losses/classification_loss.py, multilabel=True).
Note: the classification collator reads ``label_key``; pass label_key="labels" for this dataset.

Requires: torch only.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import torch
from torch.utils.data import Dataset

__all__ = ["MultiLabelTextDataset"]

TokenizeFn = Callable[[str], Mapping[str, Any]]
LabelSet = Iterable[Union[str, int]]


class MultiLabelTextDataset(Dataset):
    """Texts + sets of labels (names or indices) -> multi-hot vectors."""

    def __init__(
        self,
        texts: Sequence[str],
        label_sets: Sequence[LabelSet],
        class_names: Sequence[str],
        tokenize_fn: Optional[TokenizeFn] = None,
    ) -> None:
        if len(texts) != len(label_sets):
            raise ValueError(f"got {len(texts)} texts but {len(label_sets)} label sets")
        if len(set(class_names)) != len(class_names) or not class_names:
            raise ValueError("class_names must be non-empty and unique")
        self.class_names = list(class_names)
        self.class_to_index = {name: i for i, name in enumerate(self.class_names)}
        self.texts = list(texts)
        self.targets = torch.stack([self._encode(i, labels) for i, labels in enumerate(label_sets)])
        self.tokenize_fn = tokenize_fn

    def _encode(self, example_index: int, labels: LabelSet) -> torch.Tensor:
        target = torch.zeros(len(self.class_names), dtype=torch.float32)
        for label in labels:
            if isinstance(label, str):
                if label not in self.class_to_index:
                    raise ValueError(f"example {example_index}: unknown class {label!r}")
                target[self.class_to_index[label]] = 1.0
            else:
                index = int(label)
                if not 0 <= index < len(self.class_names):
                    raise ValueError(f"example {example_index}: class index {index} out of range")
                target[index] = 1.0
        return target

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def label_frequencies(self) -> torch.Tensor:
        """Positive count per class (C,), e.g. to derive BCE pos_weight from the training split."""
        return self.targets.sum(dim=0)

    def decode(self, target: torch.Tensor, threshold: float = 0.5) -> List[str]:
        """Multi-hot / probability vector (C,) -> active class names (> threshold)."""
        return [self.class_names[i] for i in torch.nonzero(target > threshold).flatten().tolist()]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        text, target = self.texts[index], self.targets[index].clone()
        if self.tokenize_fn is None:
            return {"text": text, "labels": target}
        item = dict(self.tokenize_fn(text))
        item["labels"] = target
        return item
