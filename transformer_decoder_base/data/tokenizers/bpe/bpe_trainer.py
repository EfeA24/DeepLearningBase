# Purpose: Optional adapter that trains a byte-level BPE tokenizer for decoder LMs with the Hugging Face `tokenizers` library.
"""Byte-level BPE training adapter (lazy ``tokenizers`` import).

Inputs are explicit: an iterator of TRAINING texts (never validation/test), the target
``vocab_size``, ``min_frequency`` and the special tokens (added first, so their ids are
0..len(special_tokens)-1 in the given order). Nothing is downloaded and nothing is written
unless the caller passes ``save_path`` (single ``tokenizer.json``) or calls
:func:`export_vocab_and_merges` (GPT-2 style ``vocab.json`` + ``merges.txt`` usable by
``bpe_tokenizer.ByteLevelBPETokenizer.from_files``).

The initial alphabet is the full 256-symbol byte-level alphabet, so the trained tokenizer
can encode any UTF-8 text without an unknown token.

Requires: tokenizers (pip install tokenizers).
"""
from __future__ import annotations

from typing import Any, Iterable, Iterator, List, Optional, Sequence


def _import_tokenizers():
    try:
        import tokenizers
        from tokenizers import decoders, models, pre_tokenizers, trainers
    except ImportError as exc:
        raise ImportError("Install tokenizers to train a BPE tokenizer: pip install tokenizers") from exc
    return tokenizers, decoders, models, pre_tokenizers, trainers


def train_byte_level_bpe(
    corpus: Iterable[str],
    vocab_size: int,
    special_tokens: Sequence[str] = (),
    min_frequency: int = 2,
    add_prefix_space: bool = False,
    save_path: Optional[str] = None,
    show_progress: bool = False,
    corpus_length: Optional[int] = None,
) -> Any:
    """Train and return a ``tokenizers.Tokenizer`` (byte-level BPE).

    Args:
        corpus: Iterable of training strings (consumed once).
        vocab_size: Final vocabulary size including special tokens and the 256 byte symbols.
        special_tokens: Special strings, e.g. ("<|endoftext|>", "<pad>").
        min_frequency: Minimum pair frequency for a merge.
        add_prefix_space: Treat the first word like a word preceded by a space (GPT-2: False).
        save_path: If given, write ``tokenizer.json`` to this explicit path.
        show_progress: Show the library's progress bar.
        corpus_length: Optional number of texts (only for the progress bar).
    """
    if vocab_size <= 256 + len(special_tokens):
        raise ValueError("vocab_size must exceed 256 byte symbols + number of special tokens")
    if min_frequency < 1:
        raise ValueError("min_frequency must be >= 1")
    tokenizers, decoders, models, pre_tokenizers, trainers = _import_tokenizers()
    tokenizer = tokenizers.Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=add_prefix_space)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=list(special_tokens),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=show_progress,
    )

    def _iterate() -> Iterator[str]:
        for text in corpus:
            if not isinstance(text, str):
                raise TypeError(f"corpus must yield str, got {type(text).__name__}")
            yield text

    tokenizer.train_from_iterator(_iterate(), trainer=trainer, length=corpus_length)
    if save_path is not None:
        tokenizer.save(save_path)
    return tokenizer


def export_vocab_and_merges(tokenizer: Any, directory: str, prefix: Optional[str] = None) -> List[str]:
    """Write ``[prefix-]vocab.json`` and ``[prefix-]merges.txt`` into an existing ``directory``.

    Returns the written file paths. The directory is not created implicitly.
    """
    _import_tokenizers()
    model = getattr(tokenizer, "model", None)
    if model is None or not hasattr(model, "save"):
        raise TypeError("tokenizer must be a tokenizers.Tokenizer with a BPE model")
    return list(model.save(directory, prefix))
