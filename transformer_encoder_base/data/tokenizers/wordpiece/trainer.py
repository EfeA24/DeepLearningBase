# Purpose: Optional WordPiece training adapter over the Hugging Face `tokenizers` library with an explicit corpus iterator, config and opt-in save path.
"""WordPiece training adapter (optional dependency).

Train ONLY on training-split text (never validation/test) to avoid leakage.

    config = WordPieceTrainingConfig(vocab_size=30522)
    tokenizer = train_wordpiece(iter_train_texts(), config)
    vocab = export_vocab(tokenizer)        # Mapping[str, int] for data/tokenizers/wordpiece/tokenizer.py
    save_tokenizer(tokenizer, "path/to/tokenizer.json")   # only when the caller asks

Normalization/pre-tokenization uses BertNormalizer + BertPreTokenizer, matching the reference
WordPieceTokenizer closely (minor Unicode edge cases may differ). Nothing is written to disk unless
``save_tokenizer`` / ``save_vocab_lines`` is called with an explicit path.

Requires: tokenizers (pip install tokenizers), imported lazily inside functions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional

__all__ = ["WordPieceTrainingConfig", "train_wordpiece", "export_vocab", "save_tokenizer", "save_vocab_lines"]


def _import_tokenizers() -> Any:
    try:
        import tokenizers
    except ImportError as exc:
        raise ImportError("Install tokenizers to train WordPiece vocabularies: pip install tokenizers") from exc
    return tokenizers


@dataclass
class WordPieceTrainingConfig:
    """Explicit WordPiece training options."""

    vocab_size: int = 30522
    min_frequency: int = 2
    special_tokens: List[str] = field(default_factory=lambda: ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    unk_token: str = "[UNK]"
    continuing_subword_prefix: str = "##"
    lowercase: bool = True
    strip_accents: Optional[bool] = None
    limit_alphabet: Optional[int] = 1000
    add_bert_post_processor: bool = True
    cls_token: str = "[CLS]"
    sep_token: str = "[SEP]"

    def __post_init__(self) -> None:
        if self.vocab_size <= len(self.special_tokens):
            raise ValueError("vocab_size must exceed the number of special tokens")
        if self.unk_token not in self.special_tokens:
            raise ValueError("unk_token must be listed in special_tokens")
        if self.add_bert_post_processor and not {self.cls_token, self.sep_token} <= set(self.special_tokens):
            raise ValueError("cls_token and sep_token must be in special_tokens for the BERT post-processor")


def train_wordpiece(
    corpus: Iterable[str],
    config: WordPieceTrainingConfig,
    length: Optional[int] = None,
) -> Any:
    """Train a ``tokenizers.Tokenizer`` with a WordPiece model on ``corpus`` (iterable of strings).

    ``length`` (number of items) only drives the progress display.
    """
    tk = _import_tokenizers()
    from tokenizers import models, normalizers, pre_tokenizers, processors, trainers

    tokenizer = tk.Tokenizer(
        models.WordPiece(unk_token=config.unk_token, continuing_subword_prefix=config.continuing_subword_prefix)
    )
    tokenizer.normalizer = normalizers.BertNormalizer(
        clean_text=True, handle_chinese_chars=True, strip_accents=config.strip_accents, lowercase=config.lowercase
    )
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    trainer_kwargs: Dict[str, Any] = dict(
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        special_tokens=list(config.special_tokens),
        continuing_subword_prefix=config.continuing_subword_prefix,
    )
    if config.limit_alphabet is not None:
        trainer_kwargs["limit_alphabet"] = config.limit_alphabet
    trainer = trainers.WordPieceTrainer(**trainer_kwargs)

    def iterate() -> Iterator[str]:
        for text in corpus:
            if not isinstance(text, str):
                raise TypeError(f"corpus items must be str, got {type(text).__name__}")
            yield text

    tokenizer.train_from_iterator(iterate(), trainer=trainer, length=length)
    if config.add_bert_post_processor:
        cls_id, sep_id = tokenizer.token_to_id(config.cls_token), tokenizer.token_to_id(config.sep_token)
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{config.cls_token} $A {config.sep_token}",
            pair=f"{config.cls_token} $A {config.sep_token} $B:1 {config.sep_token}:1",
            special_tokens=[(config.cls_token, cls_id), (config.sep_token, sep_id)],
        )
    return tokenizer


def export_vocab(tokenizer: Any) -> Dict[str, int]:
    """Return the trained vocabulary as {token: id} (includes special tokens)."""
    return dict(tokenizer.get_vocab(with_added_tokens=True))


def save_tokenizer(tokenizer: Any, path: str) -> None:
    """Write tokenizer.json to an explicit caller-chosen path."""
    if not path:
        raise ValueError("an explicit path is required")
    tokenizer.save(path)


def save_vocab_lines(vocab: Dict[str, int], path: str) -> None:
    """Write a BERT vocab.txt (one token per line, ordered by id) to an explicit path."""
    if not path:
        raise ValueError("an explicit path is required")
    ordered = sorted(vocab.items(), key=lambda item: item[1])
    if [i for _, i in ordered] != list(range(len(ordered))):
        raise ValueError("vocab ids must be contiguous 0..V-1 to write vocab.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(token for token, _ in ordered) + "\n")
