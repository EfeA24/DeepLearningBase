# Purpose: Encodes and decodes text character-by-character with a caller-supplied vocabulary and explicit unknown-character policy.
"""Character tokenizer over a supplied char -> id mapping.

- The mapping is supplied by the caller (built from training text only). It must be
  one-to-one so decoding is well defined.
- Unknown-character policy (``unknown_policy``): "error" raises KeyError, "unk" maps to
  ``unk_id`` (required), "skip" drops the character.
- ``encode`` returns a list of int ids (no BOS/EOS added unless ``bos_id``/``eos_id`` are passed).
- ``decode`` joins characters; ids in ``skip_ids`` (e.g. pad/bos/eos) are dropped; ids not in
  the mapping raise ValueError.

Requires: no third-party packages. Optional duck-typed interop with
data/vocabulary/vocabulary.py via :meth:`CharacterTokenizer.from_vocabulary` (no import needed).
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

_POLICIES = ("error", "unk", "skip")


class CharacterTokenizer:
    """Character-level encode/decode with an explicit unknown-token policy."""

    def __init__(
        self,
        char_to_id: Mapping[str, int],
        unknown_policy: str = "error",
        unk_id: Optional[int] = None,
        special_tokens: Optional[Mapping[str, int]] = None,
    ) -> None:
        if unknown_policy not in _POLICIES:
            raise ValueError(f"unknown_policy must be one of {_POLICIES}, got {unknown_policy!r}.")
        if unknown_policy == "unk" and unk_id is None:
            raise ValueError("unknown_policy='unk' requires unk_id.")
        for char in char_to_id:
            if len(char) != 1:
                raise ValueError(f"char_to_id keys must be single characters, got {char!r}.")

        self.char_to_id: Dict[str, int] = dict(char_to_id)
        self.special_tokens: Dict[str, int] = dict(special_tokens or {})
        self.unknown_policy = unknown_policy
        self.unk_id = unk_id

        self.id_to_token: Dict[int, str] = {}
        for token, index in list(self.char_to_id.items()) + list(self.special_tokens.items()):
            if index in self.id_to_token:
                raise ValueError(f"id {index} is assigned to both {self.id_to_token[index]!r} and {token!r}.")
            self.id_to_token[index] = token

    @classmethod
    def from_vocabulary(cls, vocabulary: Any, unknown_policy: str = "unk") -> "CharacterTokenizer":
        """Build from an object exposing ``get_stoi()``, ``special_tokens`` and ``unk_id`` (see vocabulary.py)."""
        stoi = vocabulary.get_stoi()
        specials = set(vocabulary.special_tokens)
        chars = {token: index for token, index in stoi.items() if token not in specials}
        special_map = {token: stoi[token] for token in specials}
        return cls(chars, unknown_policy=unknown_policy, unk_id=vocabulary.unk_id, special_tokens=special_map)

    def __len__(self) -> int:
        return len(self.id_to_token)

    def encode(self, text: str, bos_id: Optional[int] = None, eos_id: Optional[int] = None) -> List[int]:
        """Encode ``text`` to ids, optionally wrapped with ``bos_id`` / ``eos_id``."""
        ids: List[int] = [] if bos_id is None else [bos_id]
        for char in text:
            index = self.char_to_id.get(char)
            if index is not None:
                ids.append(index)
            elif self.unknown_policy == "unk":
                ids.append(self.unk_id)  # type: ignore[arg-type]
            elif self.unknown_policy == "error":
                raise KeyError(f"Character {char!r} is not in the vocabulary.")
        if eos_id is not None:
            ids.append(eos_id)
        return ids

    def decode(self, ids: Iterable[int], skip_ids: Iterable[int] = ()) -> str:
        """Decode ids to text; special tokens not in ``skip_ids`` are emitted as their string form."""
        skip = set(int(i) for i in skip_ids)
        pieces = []
        for index in ids:
            index = int(index)
            if index in skip:
                continue
            if index not in self.id_to_token:
                raise ValueError(f"id {index} is not in the vocabulary.")
            pieces.append(self.id_to_token[index])
        return "".join(pieces)
