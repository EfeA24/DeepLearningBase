# Purpose: Provides a bidirectional token/id vocabulary with stable special-token ids and explicit unknown handling for sequence models.
"""Token <-> id vocabulary.

Id layout (stable and deterministic):
- Special tokens first, in this order: pad, unk, bos, eos (each optional; None = absent),
  then ``extra_special_tokens``; they receive ids 0..K-1 in that order.
  With the defaults: <pad>=0, <unk>=1, <bos>=2, <eos>=3.
- Regular tokens follow in the order supplied (ids K..N-1).
- Duplicate tokens (including a regular token equal to a special token) raise ValueError.

Unknown handling: lookups of unknown tokens return ``unk_id`` when an unk token exists,
otherwise raise KeyError. Serialization via ``to_dict``/``from_dict`` (JSON-friendly; the
caller decides where/how to write it).

Requires: no third-party packages. No sibling files.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence


class Vocabulary:
    """Bidirectional token/id mapping with special tokens at fixed leading ids."""

    def __init__(
        self,
        tokens: Iterable[str] = (),
        pad_token: Optional[str] = "<pad>",
        unk_token: Optional[str] = "<unk>",
        bos_token: Optional[str] = "<bos>",
        eos_token: Optional[str] = "<eos>",
        extra_special_tokens: Sequence[str] = (),
    ) -> None:
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.extra_special_tokens = list(extra_special_tokens)

        self._itos: List[str] = []
        self._stoi: Dict[str, int] = {}
        for token in self.special_tokens:
            self._add(token)
        for token in tokens:
            self._add(token)

    def _add(self, token: str) -> None:
        if not isinstance(token, str):
            raise TypeError(f"Tokens must be str, got {type(token).__name__}.")
        if token in self._stoi:
            raise ValueError(f"Duplicate token {token!r} in vocabulary.")
        self._stoi[token] = len(self._itos)
        self._itos.append(token)

    @property
    def special_tokens(self) -> List[str]:
        """Special tokens in id order (pad, unk, bos, eos, extras), skipping absent ones."""
        named = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        return [t for t in named if t is not None] + self.extra_special_tokens

    def _special_id(self, token: Optional[str]) -> Optional[int]:
        return None if token is None else self._stoi[token]

    @property
    def pad_id(self) -> Optional[int]:
        return self._special_id(self.pad_token)

    @property
    def unk_id(self) -> Optional[int]:
        return self._special_id(self.unk_token)

    @property
    def bos_id(self) -> Optional[int]:
        return self._special_id(self.bos_token)

    @property
    def eos_id(self) -> Optional[int]:
        return self._special_id(self.eos_token)

    @property
    def special_ids(self) -> List[int]:
        return [self._stoi[t] for t in self.special_tokens]

    def __len__(self) -> int:
        return len(self._itos)

    def __contains__(self, token: object) -> bool:
        return token in self._stoi

    def lookup_id(self, token: str) -> int:
        """Id of ``token``; unknown tokens map to ``unk_id`` or raise KeyError when there is no unk token."""
        index = self._stoi.get(token)
        if index is not None:
            return index
        if self.unk_token is None:
            raise KeyError(f"Token {token!r} is not in the vocabulary and no unk_token is configured.")
        return self._stoi[self.unk_token]

    def lookup_token(self, index: int) -> str:
        if not 0 <= index < len(self._itos):
            raise IndexError(f"id {index} out of range for vocabulary of size {len(self._itos)}.")
        return self._itos[index]

    def encode(self, tokens: Iterable[str], add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Map tokens to ids, optionally adding BOS/EOS (which must then be configured)."""
        ids = [self.lookup_id(token) for token in tokens]
        if add_bos:
            if self.bos_id is None:
                raise ValueError("add_bos=True but the vocabulary has no bos_token.")
            ids.insert(0, self.bos_id)
        if add_eos:
            if self.eos_id is None:
                raise ValueError("add_eos=True but the vocabulary has no eos_token.")
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> List[str]:
        """Map ids back to tokens; with ``skip_special`` pad/bos/eos/extra specials are dropped.

        The unk token is kept because it stands in for real content.
        """
        special = set(self.special_ids) if skip_special else set()
        if self.unk_id is not None:
            special.discard(self.unk_id)
        return [self.lookup_token(int(i)) for i in ids if int(i) not in special]

    def get_stoi(self) -> Dict[str, int]:
        """Copy of the token -> id mapping."""
        return dict(self._stoi)

    def get_itos(self) -> List[str]:
        """Copy of the id -> token list."""
        return list(self._itos)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tokens": self._itos[len(self.special_tokens) :],
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "extra_special_tokens": list(self.extra_special_tokens),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Vocabulary":
        required = ("tokens", "pad_token", "unk_token", "bos_token", "eos_token", "extra_special_tokens")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"Vocabulary dict is missing keys: {missing}.")
        return cls(
            tokens=data["tokens"],
            pad_token=data["pad_token"],
            unk_token=data["unk_token"],
            bos_token=data["bos_token"],
            eos_token=data["eos_token"],
            extra_special_tokens=data["extra_special_tokens"],
        )
