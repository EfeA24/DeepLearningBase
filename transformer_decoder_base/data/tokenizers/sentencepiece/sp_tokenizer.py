# Purpose: Lazily adapts a SentencePiece model (explicit model path or processor) to the decoder-LM tokenizer interface.
"""SentencePiece tokenizer adapter.

Construction: ``SentencePieceTokenizer.from_model_file(path)`` loads the ``.model`` file the
caller points to (inside the classmethod, never at import), or wrap an existing
``sentencepiece.SentencePieceProcessor`` with the constructor.

Interface:
    encode(text, add_bos=False, add_eos=False) -> list[int]
    decode(ids, skip_special_tokens=True)      -> str
    vocab_size, bos_id, eos_id, pad_id, unk_id -> int (None if the id is disabled, i.e. -1 in
                                                  SentencePiece)
BOS/EOS are added only when explicitly requested. ``skip_special_tokens`` removes the
model's control ids (bos/eos/pad) before decoding.

Requires: sentencepiece (pip install sentencepiece).
"""
from __future__ import annotations

from typing import Any, Iterable, List, Optional


def _import_sentencepiece():
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise ImportError("Install sentencepiece to use this adapter: pip install sentencepiece") from exc
    return spm


class SentencePieceTokenizer:
    """Thin wrapper around a ``SentencePieceProcessor``."""

    def __init__(self, processor: Any) -> None:
        for attr in ("encode", "decode", "get_piece_size"):
            if not hasattr(processor, attr):
                raise TypeError(f"processor lacks '{attr}'; expected a SentencePieceProcessor")
        self.processor = processor

    @classmethod
    def from_model_file(cls, model_path: str) -> "SentencePieceTokenizer":
        spm = _import_sentencepiece()
        processor = spm.SentencePieceProcessor()
        if not processor.Load(model_path):
            raise OSError(f"Failed to load SentencePiece model from {model_path!r}")
        return cls(processor)

    @staticmethod
    def _optional_id(value: int) -> Optional[int]:
        return None if value is None or value < 0 else int(value)

    @property
    def vocab_size(self) -> int:
        return int(self.processor.get_piece_size())

    @property
    def bos_id(self) -> Optional[int]:
        return self._optional_id(self.processor.bos_id())

    @property
    def eos_id(self) -> Optional[int]:
        return self._optional_id(self.processor.eos_id())

    @property
    def pad_id(self) -> Optional[int]:
        return self._optional_id(self.processor.pad_id())

    @property
    def unk_id(self) -> Optional[int]:
        return self._optional_id(self.processor.unk_id())

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = [int(i) for i in self.processor.encode(text, out_type=int)]
        if add_bos:
            if self.bos_id is None:
                raise ValueError("This SentencePiece model has no BOS id")
            ids = [self.bos_id] + ids
        if add_eos:
            if self.eos_id is None:
                raise ValueError("This SentencePiece model has no EOS id")
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        ids = [int(i) for i in ids]
        if skip_special_tokens:
            special = {i for i in (self.bos_id, self.eos_id, self.pad_id) if i is not None}
            ids = [i for i in ids if i not in special]
        return self.processor.decode(ids)

    def id_to_piece(self, token_id: int) -> str:
        return self.processor.id_to_piece(int(token_id))

    def piece_to_id(self, piece: str) -> int:
        return int(self.processor.piece_to_id(piece))
