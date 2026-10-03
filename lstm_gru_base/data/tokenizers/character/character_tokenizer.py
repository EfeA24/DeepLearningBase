# Purpose: Encodes text to character ids and decodes back using a caller-supplied character vocabulary for LSTM/GRU char-level models.
"""Character-level tokenizer adapter.

Contract:
    - The vocabulary is a ``dict[str, int]`` of single characters (plus optional special tokens),
      supplied by the caller (fit on training text, e.g. with ``build_char_vocab``).
    - Unknown characters follow ``unknown_policy``: ``"error"`` (raise), ``"unk"`` (map to ``unk_token``),
      or ``"skip"`` (drop the character).
    - ``encode`` returns ``list[int]``; ``decode`` joins characters with no separator.

Requires: stdlib only.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence


def build_char_vocab(
    train_texts: Iterable[str],
    special_tokens: Sequence[str] = ("<pad>", "<unk>", "<bos>", "<eos>"),
    min_freq: int = 1,
) -> Dict[str, int]:
    """Build a deterministic char->id mapping from TRAINING texts (specials first, then sorted chars)."""
    counts = Counter()
    for text in train_texts:
        counts.update(text)
    chars = sorted(c for c, n in counts.items() if n >= min_freq and c not in special_tokens)
    mapping: Dict[str, int] = {}
    for tok in list(special_tokens) + chars:
        mapping.setdefault(tok, len(mapping))
    return mapping


class CharacterTokenizer:
    """Maps characters to ids using an explicit vocabulary."""

    def __init__(
        self,
        vocab: Dict[str, int],
        unknown_policy: str = "unk",
        unk_token: Optional[str] = "<unk>",
        bos_token: Optional[str] = "<bos>",
        eos_token: Optional[str] = "<eos>",
        pad_token: Optional[str] = "<pad>",
    ) -> None:
        if unknown_policy not in ("error", "unk", "skip"):
            raise ValueError(f"unknown_policy must be 'error', 'unk' or 'skip', got {unknown_policy!r}.")
        if unknown_policy == "unk" and (unk_token is None or unk_token not in vocab):
            raise ValueError("unknown_policy='unk' requires unk_token to be present in vocab.")
        ids = list(vocab.values())
        if len(set(ids)) != len(ids):
            raise ValueError("Vocabulary ids must be unique.")
        self.vocab = dict(vocab)
        self.id_to_token = {i: t for t, i in self.vocab.items()}
        self.unknown_policy = unknown_policy
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.special_tokens = {t for t in (unk_token, bos_token, eos_token, pad_token) if t is not None and t in vocab}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _special_id(self, token: Optional[str], name: str) -> int:
        if token is None or token not in self.vocab:
            raise ValueError(f"{name} is not configured in the vocabulary.")
        return self.vocab[token]

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(self._special_id(self.bos_token, "bos_token"))
        for ch in text:
            idx = self.vocab.get(ch)
            if idx is not None:
                ids.append(idx)
            elif self.unknown_policy == "unk":
                ids.append(self.vocab[self.unk_token])  # type: ignore[index]
            elif self.unknown_policy == "error":
                raise KeyError(f"Character {ch!r} not in vocabulary.")
        if add_eos:
            ids.append(self._special_id(self.eos_token, "eos_token"))
        return ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        out = []
        for idx in ids:
            tok = self.id_to_token.get(int(idx))
            if tok is None:
                raise KeyError(f"Id {idx} not in vocabulary.")
            if skip_special_tokens and tok in self.special_tokens:
                continue
            out.append(tok)
        return "".join(out)
