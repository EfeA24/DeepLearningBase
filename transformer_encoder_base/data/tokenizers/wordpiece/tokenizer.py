# Purpose: Reference WordPiece tokenizer (BERT-style basic pre-tokenization + greedy longest-match-first over a caller-supplied vocab) with special tokens, offsets and word ids.
"""Reference WordPiece tokenizer (pure Python; readable, not fast).

Pipeline for raw text:
    1. Special-token strings present in the text (e.g. "[MASK]") are kept atomic when
       ``recognize_special_tokens=True``.
    2. Basic pre-tokenization: drop control chars, split on whitespace, split every punctuation
       char (and CJK ideograph) into its own word, optional lowercasing / accent stripping.
    3. WordPiece: per word, greedy longest-match-first; continuation pieces carry ``continuing_prefix``
       ("##"). A word with an unmatchable remainder or longer than ``max_input_chars_per_word`` becomes
       a single ``unk_token``.

Encoding output (``Encoding`` dataclass, plain Python lists, no padding):
    input_ids, token_type_ids (0 for [CLS] A [SEP], 1 for B [SEP]), attention_mask (all 1),
    special_tokens_mask (1 = added [CLS]/[SEP]), offsets (start, end) character spans into the
    ORIGINAL text of the segment the token came from ((0, 0) for added special tokens), word_ids
    (index of the source word, None for added special tokens), sequence_ids (0/1, None for specials).
Template: single "[CLS] A [SEP]"; pair "[CLS] A [SEP] B [SEP]".
Truncation only when ``truncation=True`` and ``max_length`` is set ("longest_first" for pairs);
padding is the collator's job.

Requires: stdlib unicodedata (imported lazily). Load vocab files yourself and pass lines/mappings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = ["Encoding", "WordPieceTokenizer"]

# (token, start, end, word_index); offsets are into the original segment text.
_Piece = Tuple[str, int, int, int]


@dataclass
class Encoding:
    """Unpadded encoding of one example."""

    input_ids: List[int] = field(default_factory=list)
    token_type_ids: List[int] = field(default_factory=list)
    attention_mask: List[int] = field(default_factory=list)
    special_tokens_mask: List[int] = field(default_factory=list)
    offsets: List[Tuple[int, int]] = field(default_factory=list)
    word_ids: List[Optional[int]] = field(default_factory=list)
    sequence_ids: List[Optional[int]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, list]:
        return {
            "input_ids": self.input_ids,
            "token_type_ids": self.token_type_ids,
            "attention_mask": self.attention_mask,
            "special_tokens_mask": self.special_tokens_mask,
            "offsets": self.offsets,
            "word_ids": self.word_ids,
            "sequence_ids": self.sequence_ids,
        }


def _is_punctuation(char: str, category: str) -> bool:
    cp = ord(char)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    return category.startswith("P")


def _is_cjk(cp: int) -> bool:
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0x20000 <= cp <= 0x2A6DF
        or 0x2A700 <= cp <= 0x2B73F
        or 0x2B740 <= cp <= 0x2B81F
        or 0x2B820 <= cp <= 0x2CEAF
        or 0xF900 <= cp <= 0xFAFF
        or 0x2F800 <= cp <= 0x2FA1F
    )


class WordPieceTokenizer:
    """Greedy longest-match-first WordPiece over an explicit vocabulary."""

    def __init__(
        self,
        vocab: Mapping[str, int],
        unk_token: str = "[UNK]",
        cls_token: str = "[CLS]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        mask_token: Optional[str] = "[MASK]",
        continuing_prefix: str = "##",
        lowercase: bool = True,
        strip_accents: Optional[bool] = None,
        split_on_punctuation: bool = True,
        handle_chinese_chars: bool = True,
        max_input_chars_per_word: int = 100,
        recognize_special_tokens: bool = True,
    ) -> None:
        if not vocab:
            raise ValueError("vocab must not be empty")
        ids = list(vocab.values())
        if len(set(ids)) != len(ids):
            raise ValueError("vocab ids must be unique")
        self.vocab: Dict[str, int] = dict(vocab)
        self.id_to_token: Dict[int, str] = {i: t for t, i in self.vocab.items()}
        self.unk_token, self.cls_token, self.sep_token = unk_token, cls_token, sep_token
        self.pad_token, self.mask_token = pad_token, mask_token
        for name, token in (("unk", unk_token), ("cls", cls_token), ("sep", sep_token), ("pad", pad_token), ("mask", mask_token)):
            if token is not None and token not in self.vocab:
                raise ValueError(f"{name}_token {token!r} is not in the vocabulary")
        self.continuing_prefix = continuing_prefix
        self.lowercase = lowercase
        self.strip_accents = lowercase if strip_accents is None else strip_accents
        self.split_on_punctuation = split_on_punctuation
        self.handle_chinese_chars = handle_chinese_chars
        if max_input_chars_per_word <= 0:
            raise ValueError("max_input_chars_per_word must be positive")
        self.max_input_chars_per_word = max_input_chars_per_word
        self.recognize_special_tokens = recognize_special_tokens

    @classmethod
    def from_vocab_lines(cls, lines: Iterable[str], **kwargs) -> "WordPieceTokenizer":
        """Build from vocab.txt-style lines (token on line i gets id i). The caller opens the file."""
        vocab: Dict[str, int] = {}
        for index, line in enumerate(lines):
            token = line.rstrip("\r\n")
            if token in vocab:
                raise ValueError(f"duplicate token {token!r} at line {index}")
            vocab[token] = index
        return cls(vocab, **kwargs)

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

    @property
    def pad_token_id(self) -> int:
        return self.vocab[self.pad_token]

    @property
    def mask_token_id(self) -> Optional[int]:
        return None if self.mask_token is None else self.vocab[self.mask_token]

    def _split_special(self, text: str) -> List[Tuple[str, int, bool]]:
        """Split text into (segment, start_offset, is_special) chunks."""
        if not self.recognize_special_tokens:
            return [(text, 0, False)]
        specials = sorted(self.special_tokens, key=len, reverse=True)
        chunks: List[Tuple[str, int, bool]] = []
        start = index = 0
        while index < len(text):
            match = next((s for s in specials if text.startswith(s, index)), None)
            if match is None:
                index += 1
                continue
            if index > start:
                chunks.append((text[start:index], start, False))
            chunks.append((match, index, True))
            index += len(match)
            start = index
        if start < len(text):
            chunks.append((text[start:], start, False))
        return chunks

    def _pre_tokenize(self, text: str, base: int = 0) -> List[Tuple[str, List[int]]]:
        """Return (normalized_word, char_map) where char_map[i] is the original index of normalized char i."""
        import unicodedata

        words: List[Tuple[str, List[int]]] = []
        chars: List[str] = []
        char_map: List[int] = []

        def flush() -> None:
            if chars:
                words.append(("".join(chars), list(char_map)))
                chars.clear()
                char_map.clear()

        for index, char in enumerate(text):
            category = unicodedata.category(char)
            if char in " \t\n\r" or category == "Zs":
                flush()
                continue
            if ord(char) in (0, 0xFFFD) or category in ("Cc", "Cf"):
                continue
            norm = char.lower() if self.lowercase else char
            if self.strip_accents:
                norm = "".join(c for c in unicodedata.normalize("NFD", norm) if unicodedata.category(c) != "Mn")
            if not norm:
                continue
            isolated = (self.split_on_punctuation and _is_punctuation(char, category)) or (
                self.handle_chinese_chars and _is_cjk(ord(char))
            )
            if isolated:
                flush()
                words.append((norm, [base + index] * len(norm)))
            else:
                chars.append(norm)
                char_map.extend([base + index] * len(norm))
        flush()
        return words

    def wordpiece(self, word: str) -> List[Tuple[str, int, int]]:
        """Greedy longest-match-first split of one normalized word -> [(piece, start, end)] in word indices."""
        if len(word) > self.max_input_chars_per_word:
            return [(self.unk_token, 0, len(word))]
        pieces: List[Tuple[str, int, int]] = []
        start = 0
        while start < len(word):
            end = len(word)
            match = None
            while start < end:
                candidate = word[start:end]
                if start > 0:
                    candidate = self.continuing_prefix + candidate
                if candidate in self.vocab:
                    match = candidate
                    break
                end -= 1
            if match is None:
                return [(self.unk_token, 0, len(word))]
            pieces.append((match, start, end))
            start = end
        return pieces

    def _pieces_for_text(self, text: str) -> List[_Piece]:
        result: List[_Piece] = []
        word_index = 0
        for segment, seg_start, is_special in self._split_special(text):
            if is_special:
                result.append((segment, seg_start, seg_start + len(segment), word_index))
                word_index += 1
                continue
            for word, char_map in self._pre_tokenize(segment, base=seg_start):
                for piece, start, end in self.wordpiece(word):
                    result.append((piece, char_map[start], char_map[end - 1] + 1, word_index))
                word_index += 1
        return result

    def tokenize(self, text: str) -> List[str]:
        """Text -> WordPiece token strings (no special tokens added)."""
        return [piece for piece, _, _, _ in self._pieces_for_text(text)]

    def convert_tokens_to_ids(self, tokens: Sequence[str]) -> List[int]:
        unk = self.vocab[self.unk_token]
        return [self.vocab.get(token, unk) for token in tokens]

    def convert_ids_to_tokens(self, ids: Sequence[int]) -> List[str]:
        try:
            return [self.id_to_token[int(i)] for i in ids]
        except KeyError as exc:
            raise ValueError(f"unknown token id {exc.args[0]}") from exc

    def _truncate(self, first: List[_Piece], second: Optional[List[_Piece]], max_length: int) -> None:
        budget = max_length - (3 if second is not None else 2)
        if budget < 0:
            raise ValueError(f"max_length={max_length} cannot even hold the special tokens")
        if second is None:
            del first[budget:]
            return
        while len(first) + len(second) > budget:
            (first if len(first) > len(second) else second).pop()

    def _build(self, first: List[_Piece], second: Optional[List[_Piece]], add_special_tokens: bool) -> Encoding:
        enc = Encoding()

        def add(token: str, type_id: int, special: bool, offset: Tuple[int, int], word: Optional[int], seq: Optional[int]) -> None:
            enc.input_ids.append(self.vocab.get(token, self.vocab[self.unk_token]))
            enc.token_type_ids.append(type_id)
            enc.attention_mask.append(1)
            enc.special_tokens_mask.append(1 if special else 0)
            enc.offsets.append(offset)
            enc.word_ids.append(word)
            enc.sequence_ids.append(seq)

        if add_special_tokens:
            add(self.cls_token, 0, True, (0, 0), None, None)
        for token, start, end, word in first:
            add(token, 0, False, (start, end), word, 0)
        if add_special_tokens:
            add(self.sep_token, 0, True, (0, 0), None, None)
        if second is not None:
            for token, start, end, word in second:
                add(token, 1, False, (start, end), word, 1)
            if add_special_tokens:
                add(self.sep_token, 1, True, (0, 0), None, None)
        return enc

    def _finish(
        self,
        first: List[_Piece],
        second: Optional[List[_Piece]],
        add_special_tokens: bool,
        max_length: Optional[int],
        truncation: bool,
    ) -> Encoding:
        if max_length is not None:
            n_special = (3 if second is not None else 2) if add_special_tokens else 0
            total = len(first) + (len(second) if second is not None else 0) + n_special
            if total > max_length:
                if not truncation:
                    raise ValueError(f"encoded length {total} > max_length={max_length}; pass truncation=True")
                if not add_special_tokens:
                    raise ValueError("truncation is only implemented together with add_special_tokens=True")
                self._truncate(first, second, max_length)
        return self._build(first, second, add_special_tokens)

    def encode(
        self,
        text: str,
        text_pair: Optional[str] = None,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
    ) -> Encoding:
        """Encode raw text (optionally a pair). Offsets are into ``text`` / ``text_pair`` respectively."""
        first = self._pieces_for_text(text)
        second = self._pieces_for_text(text_pair) if text_pair is not None else None
        return self._finish(first, second, add_special_tokens, max_length, truncation)

    def encode_words(
        self,
        words: Sequence[str],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
    ) -> Encoding:
        """Encode pre-split words (NER/POS). ``word_ids`` index into ``words``; offsets are within each word.

        A word may yield several pieces (sharing its word id) or none (e.g. whitespace-only).
        """
        pieces: List[_Piece] = []
        for word_index, word in enumerate(words):
            for piece, start, end, _ in self._pieces_for_text(word):
                pieces.append((piece, start, end, word_index))
        return self._finish(pieces, None, add_special_tokens, max_length, truncation)

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        """Join pieces: continuation pieces are glued to the previous token, others separated by a space."""
        special = set(self.special_tokens) if skip_special_tokens else set()
        words: List[str] = []
        for token in self.convert_ids_to_tokens(ids):
            if token in special:
                continue
            if token.startswith(self.continuing_prefix) and words:
                words[-1] += token[len(self.continuing_prefix):]
            else:
                words.append(token)
        return " ".join(words)
