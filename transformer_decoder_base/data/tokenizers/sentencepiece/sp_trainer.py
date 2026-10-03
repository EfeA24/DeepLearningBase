# Purpose: Lazily trains a SentencePiece (BPE or unigram) model for decoder LMs from an explicit input file list or sentence iterator.
"""SentencePiece training adapter.

Exactly one corpus source must be given: ``input_files`` (explicit local text files, one
sentence per line) or ``sentence_iterator`` (iterable of TRAINING strings). Output files are
written only to the explicit ``model_prefix`` the caller passes:
``{model_prefix}.model`` and ``{model_prefix}.vocab`` (returned as a tuple).

Special-id defaults follow SentencePiece (unk=0, bos=1, eos=2, pad disabled=-1). Set
``pad_id`` to a free id (e.g. 3) if your collators need a dedicated pad token. Extra trainer
flags can be passed through ``extra_options`` (forwarded verbatim; see the SentencePiece
trainer documentation — flag names are version-sensitive).

Requires: sentencepiece (pip install sentencepiece).
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple


def train_sentencepiece(
    model_prefix: str,
    vocab_size: int,
    input_files: Optional[Sequence[str]] = None,
    sentence_iterator: Optional[Iterable[str]] = None,
    model_type: str = "bpe",
    character_coverage: float = 1.0,
    user_defined_symbols: Sequence[str] = (),
    unk_id: int = 0,
    bos_id: int = 1,
    eos_id: int = 2,
    pad_id: int = -1,
    byte_fallback: bool = False,
    extra_options: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """Train a SentencePiece model and return ``(model_path, vocab_path)``."""
    if (input_files is None) == (sentence_iterator is None):
        raise ValueError("Provide exactly one of input_files or sentence_iterator")
    if model_type not in ("bpe", "unigram", "char", "word"):
        raise ValueError(f"Unsupported model_type {model_type!r}")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if not 0.0 < character_coverage <= 1.0:
        raise ValueError("character_coverage must be in (0, 1]")
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise ImportError("Install sentencepiece to train a model: pip install sentencepiece") from exc

    options: Dict[str, Any] = dict(
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        unk_id=unk_id,
        bos_id=bos_id,
        eos_id=eos_id,
        pad_id=pad_id,
        byte_fallback=byte_fallback,
    )
    if user_defined_symbols:
        options["user_defined_symbols"] = list(user_defined_symbols)
    if extra_options:
        options.update(extra_options)
    if input_files is not None:
        if not input_files:
            raise ValueError("input_files is empty")
        options["input"] = ",".join(input_files)
    else:
        options["sentence_iterator"] = iter(sentence_iterator)
    spm.SentencePieceTrainer.train(**options)
    return f"{model_prefix}.model", f"{model_prefix}.vocab"
