# Purpose: Reference byte-level BPE tokenizer (GPT-2 style) that applies caller-supplied merges and vocab for decoder LM encode/decode.
"""Byte-level BPE tokenizer (reference implementation, not optimized).

Model files contract (GPT-2 format):
    vocab:  dict token_string -> id, where token strings use the GPT-2 byte-to-unicode alphabet
            (every byte 0..255 maps to one printable unicode char; e.g. space -> "Ġ").
            A complete byte-level vocab contains all 256 single-byte symbols, so no text is
            ever unknown.
    merges: ordered list of (left, right) symbol pairs; earlier = higher priority.
            ``merges.txt`` lines are "left right" (an optional "#version" header is skipped).
    special_tokens: optional dict string -> id (e.g. {"<|endoftext|>": 50256}); matched
            verbatim before BPE when ``allow_special=True`` and never split.

Pipeline: text -> split out special tokens -> regex pre-tokenization -> UTF-8 bytes ->
byte-to-unicode symbols -> greedy lowest-rank pair merging -> vocab ids.
decode: ids -> token strings -> unicode-to-byte -> UTF-8 decode (errors="replace" by default,
because an arbitrary id slice may cut a multi-byte character).

Pre-tokenization uses the GPT-2 pattern with the third-party ``regex`` module when installed
(\\p{L}/\\p{N} classes). Without it a stdlib ``re`` approximation is used (letters = [^\\W\\d_]);
results can differ from GPT-2 for some non-ASCII text — install ``regex`` for exact behavior.

Requires: Python stdlib only (json/re imported lazily); optional ``regex`` (pip install regex).
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

GPT2_SPLIT_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_FALLBACK_SPLIT_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+"""


def bytes_to_unicode() -> Dict[int, str]:
    """GPT-2 reversible byte -> printable unicode character table (256 entries)."""
    printable = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    byte_values = printable[:]
    codepoints = printable[:]
    extra = 0
    for b in range(256):
        if b not in byte_values:
            byte_values.append(b)
            codepoints.append(256 + extra)
            extra += 1
    return {b: chr(c) for b, c in zip(byte_values, codepoints)}


def _get_pairs(symbols: Sequence[str]) -> set:
    return {(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)}


class ByteLevelBPETokenizer:
    """Apply a supplied byte-level BPE model.

    Args:
        vocab: token string -> id (byte-unicode alphabet).
        merges: ordered merge pairs.
        special_tokens: optional special string -> id.
        unk_id: id for tokens missing from ``vocab`` (None -> raise KeyError).
        split_pattern: pre-tokenization regex (default GPT-2 pattern).
    """

    def __init__(
        self,
        vocab: Mapping[str, int],
        merges: Sequence[Tuple[str, str]],
        special_tokens: Optional[Mapping[str, int]] = None,
        unk_id: Optional[int] = None,
        split_pattern: Optional[str] = None,
    ) -> None:
        self.encoder: Dict[str, int] = dict(vocab)
        self.decoder: Dict[int, str] = {i: t for t, i in self.encoder.items()}
        if len(self.decoder) != len(self.encoder):
            raise ValueError("vocab contains duplicate ids")
        self.bpe_ranks: Dict[Tuple[str, str], int] = {}
        for rank, pair in enumerate(merges):
            if len(pair) != 2:
                raise ValueError(f"merge #{rank} must be a pair, got {pair!r}")
            self.bpe_ranks.setdefault((pair[0], pair[1]), rank)
        self.special_tokens: Dict[str, int] = dict(special_tokens or {})
        self.special_ids: Dict[int, str] = {i: s for s, i in self.special_tokens.items()}
        for s, i in self.special_tokens.items():
            if i in self.decoder and self.decoder[i] != s:
                raise ValueError(f"special token {s!r} id {i} collides with vocab token {self.decoder[i]!r}")
        self.unk_id = unk_id
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {c: b for b, c in self.byte_encoder.items()}
        self._split_regex = self._compile_split(split_pattern)
        self._special_regex = self._compile_special()
        self._cache: Dict[str, List[str]] = {}

    @classmethod
    def from_files(
        cls,
        vocab_path: str,
        merges_path: str,
        special_tokens: Optional[Mapping[str, int]] = None,
        unk_id: Optional[int] = None,
    ) -> "ByteLevelBPETokenizer":
        """Load GPT-2 style ``vocab.json`` + ``merges.txt`` from explicit paths."""
        import json

        with open(vocab_path, "r", encoding="utf-8") as fh:
            vocab = json.load(fh)
        merges: List[Tuple[str, str]] = []
        with open(merges_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line or line.startswith("#version"):
                    continue
                parts = line.split(" ")
                if len(parts) != 2:
                    raise ValueError(f"Malformed merges line: {line!r}")
                merges.append((parts[0], parts[1]))
        return cls(vocab, merges, special_tokens=special_tokens, unk_id=unk_id)

    @staticmethod
    def _compile_split(pattern: Optional[str]):
        try:
            import regex

            return regex.compile(pattern or GPT2_SPLIT_PATTERN)
        except ImportError:
            import re

            if pattern is not None and "\\p{" in pattern:
                raise ImportError("This split pattern needs the 'regex' package: pip install regex")
            return re.compile(pattern or _FALLBACK_SPLIT_PATTERN)

    def _compile_special(self):
        if not self.special_tokens:
            return None
        import re

        alternatives = sorted(self.special_tokens, key=len, reverse=True)
        return re.compile("(" + "|".join(re.escape(s) for s in alternatives) + ")")

    @property
    def vocab_size(self) -> int:
        ids = list(self.decoder) + list(self.special_ids)
        return max(ids) + 1 if ids else 0

    def bpe(self, word: str) -> List[str]:
        """Merge one pre-token (already in byte-unicode symbols) into BPE symbols."""
        if word in self._cache:
            return self._cache[word]
        symbols = list(word)
        while len(symbols) > 1:
            pairs = _get_pairs(symbols)
            best = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if best not in self.bpe_ranks:
                break
            merged: List[str] = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == best[0] and symbols[i + 1] == best[1]:
                    merged.append(best[0] + best[1])
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            symbols = merged
        self._cache[word] = symbols
        return symbols

    def _encode_ordinary(self, text: str) -> List[int]:
        ids: List[int] = []
        for piece in self._split_regex.findall(text):
            word = "".join(self.byte_encoder[b] for b in piece.encode("utf-8"))
            for symbol in self.bpe(word):
                token_id = self.encoder.get(symbol)
                if token_id is None:
                    if self.unk_id is None:
                        raise KeyError(f"BPE symbol {symbol!r} not in vocab (incomplete byte-level vocab?)")
                    token_id = self.unk_id
                ids.append(token_id)
        return ids

    def encode(
        self,
        text: str,
        allow_special: bool = True,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
    ) -> List[int]:
        """Encode ``text``; special-token strings are matched only when ``allow_special``."""
        ids: List[int] = [] if bos_id is None else [bos_id]
        if allow_special and self._special_regex is not None:
            for part in self._special_regex.split(text):
                if not part:
                    continue
                if part in self.special_tokens:
                    ids.append(self.special_tokens[part])
                else:
                    ids.extend(self._encode_ordinary(part))
        else:
            ids.extend(self._encode_ordinary(text))
        if eos_id is not None:
            ids.append(eos_id)
        return ids

    def encode_batch(self, texts: Iterable[str], **kwargs) -> List[List[int]]:
        return [self.encode(t, **kwargs) for t in texts]

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = False, errors: str = "replace") -> str:
        """Decode ids to text; unknown ids raise KeyError."""
        out: List[str] = []
        buffer = bytearray()
        for token_id in ids:
            token_id = int(token_id)
            if token_id in self.special_ids:
                if buffer:
                    out.append(buffer.decode("utf-8", errors=errors))
                    buffer = bytearray()
                if not skip_special_tokens:
                    out.append(self.special_ids[token_id])
                continue
            if token_id not in self.decoder:
                raise KeyError(f"Unknown token id {token_id}")
            buffer.extend(self.byte_decoder[c] for c in self.decoder[token_id])
        if buffer:
            out.append(buffer.decode("utf-8", errors=errors))
        return "".join(out)
