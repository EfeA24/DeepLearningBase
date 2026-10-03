# Purpose: Declares configurable PAD/UNK/BOS/EOS special-token strings and ids with validation for sequence-model vocabularies.
"""Special-token configuration.

- Each special token is a (string, id) pair; a token is disabled by setting both to None
  (PAD is always required because batching relies on it).
- Validation: non-empty strings, unique strings, unique non-negative ids, and token/id both
  set or both None.
- ``as_vocabulary_kwargs`` produces keyword arguments for data/vocabulary/vocabulary.py, which
  assigns ids 0..K-1 in pad, unk, bos, eos order; it raises if this config uses other ids.

Requires: no third-party packages. No sibling imports.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class SpecialTokens:
    """Validated special-token strings and ids."""

    pad_token: str = "<pad>"
    unk_token: Optional[str] = "<unk>"
    bos_token: Optional[str] = "<bos>"
    eos_token: Optional[str] = "<eos>"
    pad_id: int = 0
    unk_id: Optional[int] = 1
    bos_id: Optional[int] = 2
    eos_id: Optional[int] = 3

    def __post_init__(self) -> None:
        if self.pad_token is None or self.pad_id is None:
            raise ValueError("pad_token and pad_id are required.")
        seen_tokens = set()
        seen_ids = set()
        for name, token, index in self._pairs(include_absent=True):
            if (token is None) != (index is None):
                raise ValueError(f"{name}: token and id must both be set or both be None.")
            if token is None:
                continue
            if not isinstance(token, str) or not token:
                raise ValueError(f"{name}: token must be a non-empty string.")
            if not isinstance(index, int) or index < 0:
                raise ValueError(f"{name}: id must be a non-negative int, got {index!r}.")
            if token in seen_tokens:
                raise ValueError(f"Duplicate special token string {token!r}.")
            if index in seen_ids:
                raise ValueError(f"Duplicate special token id {index}.")
            seen_tokens.add(token)
            seen_ids.add(index)

    def _pairs(self, include_absent: bool = False) -> List[Tuple[str, Optional[str], Optional[int]]]:
        pairs = [
            ("pad", self.pad_token, self.pad_id),
            ("unk", self.unk_token, self.unk_id),
            ("bos", self.bos_token, self.bos_id),
            ("eos", self.eos_token, self.eos_id),
        ]
        return pairs if include_absent else [p for p in pairs if p[1] is not None]

    def tokens(self) -> List[str]:
        """Enabled special-token strings ordered by id."""
        return [token for _, token, _ in sorted(self._pairs(), key=lambda p: p[2])]  # type: ignore[misc]

    def ids(self) -> List[int]:
        """Enabled special-token ids in ascending order."""
        return sorted(index for _, _, index in self._pairs())  # type: ignore[misc]

    def token_to_id(self) -> Dict[str, int]:
        return {token: index for _, token, index in self._pairs()}  # type: ignore[misc]

    def as_vocabulary_kwargs(self) -> Dict[str, Optional[str]]:
        """Keyword arguments for ``Vocabulary(...)``; requires ids 0..K-1 in pad, unk, bos, eos order."""
        enabled = self._pairs()
        for expected, (name, _, index) in enumerate(enabled):
            if index != expected:
                raise ValueError(
                    f"Vocabulary assigns {name} id {expected}, but this config uses {index}; "
                    "use contiguous ids in pad, unk, bos, eos order."
                )
        return {
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpecialTokens":
        return cls(**data)
