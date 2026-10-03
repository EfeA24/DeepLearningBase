# Purpose: Part-of-speech tagging dataset with a strict one-tag-per-token alignment contract and explicit unknown-tag policy.
"""POS tagging dataset (token level).

Alignment contract: ``tokens[i]`` has exactly one tag ``tags[i]``; lengths must match. Tokens are
linguistic words, not subwords - align to subwords with tasks/ner/ner_alignment.py (the same
first-subword / all-subword policies apply to POS).
Item schema: {"tokens": list[str], "tag_ids": list[int], "num_tokens": int}
Unknown tags (not in ``tag_to_id``): ``unknown_tag_policy`` = "error" (default) or "ignore"
(mapped to ``ignore_index`` = -100 so losses skip them).

Requires: torch only.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from torch.utils.data import Dataset

__all__ = ["POSDataset", "build_tag_map"]


def build_tag_map(tag_sequences: Iterable[Sequence[str]]) -> Dict[str, int]:
    """Sorted {tag: id} from TRAINING tag sequences only."""
    return {tag: i for i, tag in enumerate(sorted({t for seq in tag_sequences for t in seq}))}


class POSDataset(Dataset):
    """Sentences of (token, tag) pairs mapped to tag ids."""

    def __init__(
        self,
        sentences: Sequence[Tuple[Sequence[str], Sequence[str]]],
        tag_to_id: Mapping[str, int],
        unknown_tag_policy: str = "error",
        ignore_index: int = -100,
    ) -> None:
        if unknown_tag_policy not in ("error", "ignore"):
            raise ValueError(f"unknown_tag_policy must be 'error' or 'ignore', got {unknown_tag_policy!r}")
        self.tag_to_id = dict(tag_to_id)
        self.id_to_tag = {i: t for t, i in self.tag_to_id.items()}
        self.ignore_index = ignore_index
        self.items: List[Tuple[List[str], List[int]]] = []
        for index, (tokens, tags) in enumerate(sentences):
            if len(tokens) != len(tags):
                raise ValueError(f"sentence {index}: {len(tokens)} tokens but {len(tags)} tags")
            tag_ids = []
            for tag in tags:
                if tag in self.tag_to_id:
                    tag_ids.append(self.tag_to_id[tag])
                elif unknown_tag_policy == "ignore":
                    tag_ids.append(ignore_index)
                else:
                    raise ValueError(f"sentence {index}: unknown tag {tag!r}")
            self.items.append((list(tokens), tag_ids))

    @property
    def num_tags(self) -> int:
        return len(self.tag_to_id)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        tokens, tag_ids = self.items[index]
        return {"tokens": list(tokens), "tag_ids": list(tag_ids), "num_tokens": len(tokens)}
