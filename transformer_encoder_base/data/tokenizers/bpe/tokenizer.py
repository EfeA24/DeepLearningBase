# Purpose: Reference character-level BPE tokenizer that applies caller-supplied merges and vocab, with encode/decode, special tokens and character offsets for encoder models.
"""Reference BPE tokenizer (Sennrich-style, end-of-word suffix; pure Python, not fast).

Model files contract (supplied by the caller, e.g. exported by data/tokenizers/bpe/trainer.py):
    vocab:  Mapping[str, int] containing every symbol the merges can produce, single characters
            (with and without ``end_of_word_suffix``), and the special tokens.
    merges: ordered sequence of (left, right) pairs; earlier = higher priority.

Pipeline: optional lowercasing -> split on whitespace and punctuation (each punctuation char is its
own word) -> per word, start from characters (the last one gets ``end_of_word_suffix``, default
"</w>") and repeatedly merge the adjacent pair with the best (lowest) merge rank -> map symbols to
ids (unknown symbols -> ``unk_token``; if ``unk_token`` is None an error is raised).

This is NOT byte-level BPE (GPT-2/RoBERTa): characters absent from the vocab become unk.

Encoding (``BPEEncoding``): input_ids, token_type_ids, attention_mask, special_tokens_mask, offsets
((start, end) into the original text of the segment, (0, 0) for added specials), word_ids.
Template with ``cls_token``/``sep_token`` set: "<s> A </s>" and "<s> A </s> B </s>"; set them to
None to add nothing. No padding (collator's job); truncation only on request.

Decode: tokens are concatenated and the end-of-word suffix becomes a space.

Requires: no third-party packages.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = ["BPEEncoding", "BPETokenizer"]

_Piece = Tuple[str, int, int, int]


@dataclass
class BPEEncoding:
    """Unpadded encoding of one example."""

    input_ids: List[int] = field(default_factory=list)
    token_type_ids: List[int] = field(default_factory=list)
    attention_mask: List[int] = field(default_factory=list)
    special_tokens_mask: List[int] = field(default_factory=list)
    offsets: List[Tuple[int, int]] = field(default_factory=list)
    word_ids: List[Optional[int]] = field(default_factory=list)


def _is_punctuation(char: str) -> bool:
    cp = ord(char)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    import unicodedata

    return unicodedata.category(char).startswith("P")


class BPETokenizer:
    """Applies a fixed merge table; never learns."""

    def __init__(
        self,
        vocab: Mapping[str, int],
        merges: Sequence[Tuple[str, str]],
        unk_token: Optional[str] = "<unk>",
        cls_token: Optional[str] = "<s>",
        sep_token: Optional[str] = "</s>",
        pad_token: Optional[str] = "<pad>",
        mask_token: Optional[str] = "<mask>",
        end_of_word_suffix: str = "</w>",
        lowercase: bool = False,
        max_cache_size: int = 100_000,
    ) -> None:
        if not vocab:
            raise ValueError("vocab must not be empty")
        self.vocab: Dict[str, int] = dict(vocab)
        self.id_to_token: Dict[int, str] = {i: t for t, i in self.vocab.items()}
        if len(self.id_to_token) != len(self.vocab):
            raise ValueError("vocab ids must be unique")
        self.ranks: Dict[Tuple[str, str], int] = {}
        for rank, pair in enumerate(merges):
            if len(pair) != 2:
                raise ValueError(f"merge #{rank} must be a (left, right) pair, got {pair!r}")
            self.ranks.setdefault((pair[0], pair[1]), rank)
        self.unk_token, self.cls_token, self.sep_token = unk_token, cls_token, sep_token
        self.pad_token, self.mask_token = pad_token, mask_token
        for name, token in (("unk", unk_token), ("cls", cls_token), ("sep", sep_token), ("pad", pad_token), ("mask", mask_token)):
            if token is not None and token not in self.vocab:
                raise ValueError(f"{name}_token {token!r} is not in the vocabulary")
        if (cls_token is None) != (sep_token is None):
            raise ValueError("set both cls_token and sep_token, or neither")
        self.end_of_word_suffix = end_of_word_suffix
        self.lowercase = lowercase
        self.max_cache_size = max_cache_size
        self._cache: Dict[str, List[Tuple[str, int, int]]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def special_tokens(self) -> List[str]:
        tokens = [self.unk_token, self.cls_token, self.sep_token, self.pad_token, self.mask_token]
        return [t for t in tokens if t is not None]

    @property
    def special_token_ids(self) -> List[int]:
        return [self.vocab[t] for t in self.special_tokens]

    def _pre_tokenize(self, text: str) -> List[Tuple[str, List[int]]]:
        words: List[Tuple[str, List[int]]] = []
        chars: List[str] = []
        char_map: List[int] = []

        def flush() -> None:
            if chars:
                words.append(("".join(chars), list(char_map)))
                chars.clear()
                char_map.clear()

        for index, char in enumerate(text):
            if char.isspace():
                flush()
                continue
            norm = char.lower() if self.lowercase else char
            if _is_punctuation(char):
                flush()
                words.append((norm, [index] * len(norm)))
            else:
                chars.append(norm)
                char_map.extend([index] * len(norm))
        flush()
        return words

    def bpe(self, word: str) -> List[Tuple[str, int, int]]:
        """Apply merges to one word -> [(symbol, start, end)] in word character indices."""
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        symbols = list(word)
        spans = [(i, i + 1) for i in range(len(word))]
        if self.end_of_word_suffix and symbols:
            symbols[-1] += self.end_of_word_suffix
        while len(symbols) > 1:
            best_rank, best_pair = None, None
            for pair in zip(symbols, symbols[1:]):
                rank = self.ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_pair = rank, pair
            if best_pair is None:
                break
            merged_symbols, merged_spans = [], []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best_pair:
                    merged_symbols.append(symbols[i] + symbols[i + 1])
                    merged_spans.append((spans[i][0], spans[i + 1][1]))
                    i += 2
                else:
                    merged_symbols.append(symbols[i])
                    merged_spans.append(spans[i])
                    i += 1
            symbols, spans = merged_symbols, merged_spans
        result = [(s, a, b) for s, (a, b) in zip(symbols, spans)]
        if len(self._cache) < self.max_cache_size:
            self._cache[word] = result
        return result

    def _pieces(self, text: str) -> List[_Piece]:
        pieces: List[_Piece] = []
        for word_index, (word, char_map) in enumerate(self._pre_tokenize(text)):
            for symbol, start, end in self.bpe(word):
                pieces.append((symbol, char_map[start], char_map[end - 1] + 1, word_index))
        return pieces

    def tokenize(self, text: str) -> List[str]:
        """Text -> BPE symbols (no special tokens)."""
        return [p[0] for p in self._pieces(text)]

    def _token_id(self, token: str) -> int:
        if token in self.vocab:
            return self.vocab[token]
        if self.unk_token is None:
            raise ValueError(f"symbol {token!r} not in vocab and unk_token is None")
        return self.vocab[self.unk_token]

    def encode(
        self,
        text: str,
        text_pair: Optional[str] = None,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
    ) -> BPEEncoding:
        """Encode text (optionally a pair) into ids with offsets and word ids."""
        first = self._pieces(text)
        second = self._pieces(text_pair) if text_pair is not None else None
        wrap = add_special_tokens and self.cls_token is not None
        n_special = (3 if second is not None else 2) if wrap else 0
        if max_length is not None:
            budget = max_length - n_special
            if budget < 0:
                raise ValueError(f"max_length={max_length} cannot hold the special tokens")
            if len(first) + len(second or []) > budget:
                if not truncation:
                    raise ValueError(f"encoded length exceeds max_length={max_length}; pass truncation=True")
                if second is None:
                    del first[budget:]
                else:
                    while len(first) + len(second) > budget:
                        (first if len(first) > len(second) else second).pop()

        enc = BPEEncoding()

        def add(token: str, type_id: int, special: bool, offset: Tuple[int, int], word: Optional[int]) -> None:
            enc.input_ids.append(self._token_id(token))
            enc.token_type_ids.append(type_id)
            enc.attention_mask.append(1)
            enc.special_tokens_mask.append(int(special))
            enc.offsets.append(offset)
            enc.word_ids.append(word)

        if wrap:
            add(self.cls_token, 0, True, (0, 0), None)
        for token, start, end, word in first:
            add(token, 0, False, (start, end), word)
        if wrap:
            add(self.sep_token, 0, True, (0, 0), None)
        if second is not None:
            for token, start, end, word in second:
                add(token, 1, False, (start, end), word)
            if wrap:
                add(self.sep_token, 1, True, (0, 0), None)
        return enc

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        """Concatenate symbols; the end-of-word suffix marks word boundaries (-> space)."""
        special = set(self.special_tokens) if skip_special_tokens else set()
        try:
            tokens = [self.id_to_token[int(i)] for i in ids]
        except KeyError as exc:
            raise ValueError(f"unknown token id {exc.args[0]}") from exc
        text = "".join(t for t in tokens if t not in special)
        if self.end_of_word_suffix:
            text = text.replace(self.end_of_word_suffix, " ")
        return text.strip()
