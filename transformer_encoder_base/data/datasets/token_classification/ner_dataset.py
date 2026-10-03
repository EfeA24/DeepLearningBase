# Purpose: Named-entity-recognition dataset holding word-level tokens and BIO labels with alignment metadata; subword alignment is deliberately left to tasks/ner/ner_alignment.py.
"""NER dataset (word level).

Item schema: {"words": list[str], "word_labels": list[int], "num_words": int, "id": str | None}
    word_labels[i] is the label id of words[i] under the caller's ``label_to_id`` mapping
    (e.g. {"O": 0, "B-PER": 1, "I-PER": 2, ...}). This dataset does NOT tokenize into subwords and
    does NOT align labels; do that explicitly with tasks/ner/ner_alignment.py so the subword label
    policy (first subword only vs all subwords, -100 elsewhere) is visible in your pipeline.
``read_conll`` parses CoNLL-style lines supplied by the caller (the caller opens the file).
``build_label_map`` should be called on the TRAINING split only.

Requires: torch only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from torch.utils.data import Dataset

__all__ = ["NERExample", "NERDataset", "read_conll", "build_label_map"]


@dataclass(frozen=True)
class NERExample:
    """One sentence: words and their string labels (same length)."""

    words: Sequence[str]
    labels: Sequence[str]
    id: Optional[str] = None

    def __post_init__(self) -> None:
        if len(self.words) != len(self.labels):
            raise ValueError(f"example {self.id}: {len(self.words)} words but {len(self.labels)} labels")


def read_conll(
    lines: Iterable[str],
    word_column: int = 0,
    label_column: int = -1,
    separator: Optional[str] = None,
    docstart_prefix: str = "-DOCSTART-",
) -> List[NERExample]:
    """Parse CoNLL lines: one token per line, columns split by ``separator`` (default whitespace),
    blank lines separate sentences, lines starting with ``docstart_prefix`` are skipped.
    """
    examples: List[NERExample] = []
    words: List[str] = []
    labels: List[str] = []

    def flush() -> None:
        if words:
            examples.append(NERExample(list(words), list(labels), id=str(len(examples))))
            words.clear()
            labels.clear()

    for line_number, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            flush()
            continue
        if line.startswith(docstart_prefix):
            flush()
            continue
        columns = line.split(separator)
        try:
            words.append(columns[word_column])
            labels.append(columns[label_column])
        except IndexError as exc:
            raise ValueError(f"line {line_number}: not enough columns in {line!r}") from exc
    flush()
    return examples


def build_label_map(examples: Iterable[NERExample], outside_label: str = "O") -> Dict[str, int]:
    """Stable mapping with ``outside_label`` first, then remaining labels sorted. Training split only."""
    names = {label for example in examples for label in example.labels}
    names.discard(outside_label)
    return {outside_label: 0, **{name: i + 1 for i, name in enumerate(sorted(names))}}


class NERDataset(Dataset):
    """Word-level NER examples mapped to label ids."""

    def __init__(self, examples: Sequence[NERExample], label_to_id: Mapping[str, int]) -> None:
        self.examples = list(examples)
        self.label_to_id = dict(label_to_id)
        self.id_to_label = {i: name for name, i in self.label_to_id.items()}
        for example in self.examples:
            unknown = set(example.labels) - set(self.label_to_id)
            if unknown:
                raise ValueError(f"example {example.id}: labels {sorted(unknown)} not in label_to_id")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self.examples[index]
        return {
            "words": list(example.words),
            "word_labels": [self.label_to_id[label] for label in example.labels],
            "num_words": len(example.words),
            "id": example.id,
        }
