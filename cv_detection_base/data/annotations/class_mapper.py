# Purpose: Provides a stable class-name <-> integer-id mapping with an explicit unknown-category policy for detection annotation pipelines.
"""Stable class name/id mapping.

Contract:
    Ids are contiguous integers ``start_id .. start_id + K - 1`` assigned in the
    exact order of the supplied ``class_names`` (never sorted implicitly, so
    the mapping is stable across runs and machines). Detection labels in the
    target schema are these ids (LongTensor). Whether a background class exists
    is the caller's decision (e.g. pass start_id=1 and reserve 0 for background).

Unknown-category policy (``unknown_policy``):
    "error"     - raise KeyError (default; safest)
    "skip"      - return None so the caller drops the object
    "map_to_id" - return ``unknown_id`` (must be supplied)

Requires: torch only (for the tensor helpers).
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor

UNKNOWN_POLICIES = ("error", "skip", "map_to_id")


class ClassMapper:
    """Bidirectional mapping between class names and integer ids."""

    def __init__(
        self,
        class_names: Sequence[str],
        start_id: int = 0,
        unknown_policy: str = "error",
        unknown_id: Optional[int] = None,
    ) -> None:
        if isinstance(class_names, str):
            raise TypeError("class_names must be a sequence of strings, not a single string")
        names = list(class_names)
        if not names:
            raise ValueError("class_names must not be empty")
        if len(set(names)) != len(names):
            duplicates = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"class_names contains duplicates: {duplicates}")
        if unknown_policy not in UNKNOWN_POLICIES:
            raise ValueError(f"unknown_policy must be one of {UNKNOWN_POLICIES}, got {unknown_policy!r}")
        if unknown_policy == "map_to_id" and unknown_id is None:
            raise ValueError("unknown_id is required when unknown_policy='map_to_id'")

        self.class_names: List[str] = names
        self.start_id = int(start_id)
        self.unknown_policy = unknown_policy
        self.unknown_id = unknown_id
        self._name_to_id: Dict[str, int] = {n: self.start_id + i for i, n in enumerate(names)}
        self._id_to_name: Dict[int, str] = {i: n for n, i in self._name_to_id.items()}

    def __len__(self) -> int:
        return len(self.class_names)

    def __contains__(self, name: object) -> bool:
        return name in self._name_to_id

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def name_to_id(self, name: str) -> Optional[int]:
        """Map a name to its id, applying ``unknown_policy`` for unknown names."""
        if name in self._name_to_id:
            return self._name_to_id[name]
        if self.unknown_policy == "error":
            raise KeyError(f"Unknown class name {name!r}; known: {self.class_names}")
        if self.unknown_policy == "skip":
            return None
        return self.unknown_id

    def id_to_name(self, class_id: int) -> str:
        """Map an id back to its name; raises KeyError for unmapped ids."""
        class_id = int(class_id)
        if class_id not in self._id_to_name:
            raise KeyError(f"Unknown class id {class_id}; valid range {self.start_id}..{self.start_id + len(self) - 1}")
        return self._id_to_name[class_id]

    def encode(self, names: Iterable[str]) -> List[Optional[int]]:
        """Map many names; skipped unknowns appear as None (filter with ``encode_tensor``)."""
        return [self.name_to_id(n) for n in names]

    def encode_tensor(self, names: Iterable[str]) -> Tensor:
        """Map names to an int64 tensor (N,), dropping names skipped by the policy.

        When the policy is "skip", callers must drop the matching boxes too; use
        ``encode_with_keep_mask`` to get the keep mask.
        """
        ids, _ = self.encode_with_keep_mask(names)
        return ids

    def encode_with_keep_mask(self, names: Iterable[str]) -> Tuple[Tensor, Tensor]:
        """Return (ids (K,) int64, keep_mask (N,) bool) so boxes can be filtered consistently."""
        mapped = self.encode(names)
        keep = torch.tensor([m is not None for m in mapped], dtype=torch.bool)
        ids = torch.tensor([m for m in mapped if m is not None], dtype=torch.long)
        return ids, keep

    def decode(self, class_ids: Iterable[int]) -> List[str]:
        return [self.id_to_name(int(i)) for i in class_ids]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_names": list(self.class_names),
            "start_id": self.start_id,
            "unknown_policy": self.unknown_policy,
            "unknown_id": self.unknown_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ClassMapper":
        return cls(
            class_names=data["class_names"],
            start_id=data.get("start_id", 0),
            unknown_policy=data.get("unknown_policy", "error"),
            unknown_id=data.get("unknown_id"),
        )
