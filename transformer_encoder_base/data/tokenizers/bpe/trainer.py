# Purpose: Optional BPE training adapter over the Hugging Face `tokenizers` library with an explicit corpus iterator and export of vocab/merges for the reference BPE tokenizer.
"""BPE training adapter (optional dependency).

Train ONLY on training-split text. No corpus download, no implicit save.

    config = BPETrainingConfig(vocab_size=8000)
    tokenizer = train_bpe(iter_train_texts(), config)
    vocab, merges = export_vocab_and_merges(tokenizer)   # feed data/tokenizers/bpe/tokenizer.py
    save_tokenizer(tokenizer, "path/to/tokenizer.json")  # only when the caller asks

The model uses ``end_of_word_suffix`` (default "</w>") and BertPreTokenizer (whitespace +
punctuation split), the same conventions as the reference BPETokenizer.

Requires: tokenizers (pip install tokenizers); json (stdlib) for merge export; both imported lazily.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

__all__ = ["BPETrainingConfig", "train_bpe", "export_vocab_and_merges", "save_tokenizer"]


def _import_tokenizers() -> Any:
    try:
        import tokenizers
    except ImportError as exc:
        raise ImportError("Install tokenizers to train BPE models: pip install tokenizers") from exc
    return tokenizers


@dataclass
class BPETrainingConfig:
    """Explicit BPE training options."""

    vocab_size: int = 8000
    min_frequency: int = 2
    special_tokens: List[str] = field(default_factory=lambda: ["<pad>", "<unk>", "<s>", "</s>", "<mask>"])
    unk_token: str = "<unk>"
    end_of_word_suffix: str = "</w>"
    lowercase: bool = False
    limit_alphabet: Optional[int] = None

    def __post_init__(self) -> None:
        if self.vocab_size <= len(self.special_tokens):
            raise ValueError("vocab_size must exceed the number of special tokens")
        if self.unk_token not in self.special_tokens:
            raise ValueError("unk_token must be listed in special_tokens")


def train_bpe(corpus: Iterable[str], config: BPETrainingConfig, length: Optional[int] = None) -> Any:
    """Train a ``tokenizers.Tokenizer`` with a BPE model on an iterable of strings."""
    tk = _import_tokenizers()
    from tokenizers import models, normalizers, pre_tokenizers, trainers

    tokenizer = tk.Tokenizer(models.BPE(unk_token=config.unk_token, end_of_word_suffix=config.end_of_word_suffix))
    if config.lowercase:
        tokenizer.normalizer = normalizers.Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    trainer_kwargs: Dict[str, Any] = dict(
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        special_tokens=list(config.special_tokens),
        end_of_word_suffix=config.end_of_word_suffix,
    )
    if config.limit_alphabet is not None:
        trainer_kwargs["limit_alphabet"] = config.limit_alphabet
    trainer = trainers.BpeTrainer(**trainer_kwargs)

    def iterate() -> Iterator[str]:
        for text in corpus:
            if not isinstance(text, str):
                raise TypeError(f"corpus items must be str, got {type(text).__name__}")
            yield text

    tokenizer.train_from_iterator(iterate(), trainer=trainer, length=length)
    return tokenizer


def export_vocab_and_merges(tokenizer: Any) -> Tuple[Dict[str, int], List[Tuple[str, str]]]:
    """Extract ({token: id}, [(left, right), ...]) from a trained BPE tokenizer.

    Handles both serialized merge formats ("a b" strings in older versions, [a, b] lists in newer).
    """
    import json

    model = json.loads(tokenizer.to_str())["model"]
    if model.get("type") != "BPE":
        raise ValueError(f"expected a BPE model, got {model.get('type')!r}")
    merges: List[Tuple[str, str]] = []
    for item in model["merges"]:
        if isinstance(item, str):
            left, right = item.split(" ", 1)
        else:
            left, right = item
        merges.append((left, right))
    return dict(tokenizer.get_vocab(with_added_tokens=True)), merges


def save_tokenizer(tokenizer: Any, path: str) -> None:
    """Write tokenizer.json to an explicit caller-chosen path."""
    if not path:
        raise ValueError("an explicit path is required")
    tokenizer.save(path)
