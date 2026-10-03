# Purpose: Provides a bidirectional token/id vocabulary with stable special-token ids and unknown handling for LSTM/GRU text projects.
"""Self-contained vocabulary (same contract as rnn_base/data/vocabulary/vocabulary.py, no import from it).

Contract:
    - Special tokens occupy the first ids in the order given (default ``<pad>=0, <unk>=1, <bos>=2, <eos>=3``),
      so ids stay stable regardless of corpus content.
    - Regular tokens follow in insertion order; ids are contiguous ``0..len-1``.
    - Unknown tokens map to ``unk_token`` if configured, otherwise ``KeyError`` is raised.
    - ``to_dict``/``from_dict`` round-trip exactly (JSON-serializable plain types).

Requires: torch not needed; stdlib only.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence


class Vocabulary:
    """Token <-> id mapping with explicit special tokens."""

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
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: List[str] = []
        specials = [t for t in (pad_token, unk_token, bos_token, eos_token) if t is not None]
        specials.extend(extra_special_tokens)
        if len(set(specials)) != len(specials):
            raise ValueError(f"Special tokens must be unique, got {specials}.")
        self.special_tokens: List[str] = list(specials)
        for tok in self.special_tokens:
            self.add_token(tok)
        for tok in tokens:
            self.add_token(tok)

    def add_token(self, token: str) -> int:
        """Add ``token`` if missing and return its id."""
        if not isinstance(token, str):
            raise TypeError(f"Tokens must be str, got {type(token).__name__}.")
        if token in self._token_to_id:
            return self._token_to_id[token]
        idx = len(self._id_to_token)
        self._token_to_id[token] = idx
        self._id_to_token.append(token)
        return idx

    def __len__(self) -> int:
        return len(self._id_to_token)

    def __contains__(self, token: object) -> bool:
        return token in self._token_to_id

    def _special_id(self, token: Optional[str]) -> Optional[int]:
        return None if token is None else self._token_to_id[token]

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

    def token_to_id(self, token: str) -> int:
        """Return the id of ``token``; falls back to ``unk_id`` or raises ``KeyError``."""
        idx = self._token_to_id.get(token)
        if idx is not None:
            return idx
        if self.unk_token is None:
            raise KeyError(f"Token {token!r} not in vocabulary and no unk_token configured.")
        return self._token_to_id[self.unk_token]

    def id_to_token(self, idx: int) -> str:
        if not 0 <= idx < len(self._id_to_token):
            raise IndexError(f"Id {idx} out of range for vocabulary of size {len(self)}.")
        return self._id_to_token[idx]

    def encode(self, tokens: Sequence[str], add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Map tokens to ids, optionally wrapping with BOS/EOS (which must be configured)."""
        ids = [self.token_to_id(t) for t in tokens]
        if add_bos:
            if self.bos_id is None:
                raise ValueError("add_bos=True but no bos_token configured.")
            ids.insert(0, self.bos_id)
        if add_eos:
            if self.eos_id is None:
                raise ValueError("add_eos=True but no eos_token configured.")
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = False) -> List[str]:
        """Map ids back to tokens; optionally drop special tokens."""
        specials = set(self.special_tokens)
        out = []
        for idx in ids:
            tok = self.id_to_token(int(idx))
            if skip_special_tokens and tok in specials:
                continue
            out.append(tok)
        return out

    def to_dict(self) -> Dict[str, object]:
        return {
            "tokens": list(self._id_to_token),
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "special_tokens": list(self.special_tokens),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "Vocabulary":
        """Rebuild a vocabulary saved with ``to_dict``; ids are preserved exactly."""
        special_tokens = list(data["special_tokens"])  # type: ignore[arg-type]
        named = [data.get(k) for k in ("pad_token", "unk_token", "bos_token", "eos_token")]
        extras = [t for t in special_tokens if t not in named]
        vocab = cls(
            pad_token=data.get("pad_token"),  # type: ignore[arg-type]
            unk_token=data.get("unk_token"),  # type: ignore[arg-type]
            bos_token=data.get("bos_token"),  # type: ignore[arg-type]
            eos_token=data.get("eos_token"),  # type: ignore[arg-type]
            extra_special_tokens=extras,
        )
        tokens = list(data["tokens"])  # type: ignore[arg-type]
        if tokens[: len(vocab.special_tokens)] != vocab.special_tokens:
            raise ValueError("Serialized vocabulary does not start with its special tokens in order.")
        for tok in tokens[len(vocab.special_tokens):]:
            vocab.add_token(tok)
        return vocab
