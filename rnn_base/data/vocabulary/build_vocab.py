# Purpose: Builds a token vocabulary from training-split token sequences with min_freq/max_size limits for sequence models.
"""Vocabulary construction from TRAINING data only.

Call these functions on the training split; never include validation/test text, or the
vocabulary (and unknown-token rates) leak information from held-out data.

Selection rule (deterministic): count tokens with ``collections.Counter``, drop special
tokens and tokens with frequency < ``min_freq``, sort by (descending frequency, token string),
then keep at most ``max_size`` regular tokens (``max_size`` excludes special tokens).

Requires: no third-party packages. ``build_vocab`` lazily imports
``data.vocabulary.vocabulary.Vocabulary`` (template root on sys.path) unless a
``vocab_factory`` callable is injected.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Iterable, List, Optional, Sequence


def count_tokens(token_sequences: Iterable[Sequence[str]]) -> Counter:
    """Count token frequencies over an iterable of tokenized training examples."""
    counter: Counter = Counter()
    for tokens in token_sequences:
        if isinstance(tokens, str):
            raise TypeError("Each example must be a sequence of tokens, not a raw string; tokenize first.")
        counter.update(tokens)
    return counter


def select_tokens(
    counter: Counter,
    min_freq: int = 1,
    max_size: Optional[int] = None,
    exclude: Iterable[str] = (),
) -> List[str]:
    """Deterministically select regular tokens from ``counter`` (see module docstring)."""
    if min_freq < 1:
        raise ValueError(f"min_freq must be >= 1, got {min_freq}.")
    if max_size is not None and max_size < 0:
        raise ValueError(f"max_size must be non-negative, got {max_size}.")
    excluded = set(exclude)
    candidates = [(token, freq) for token, freq in counter.items() if freq >= min_freq and token not in excluded]
    candidates.sort(key=lambda item: (-item[1], item[0]))
    if max_size is not None:
        candidates = candidates[:max_size]
    return [token for token, _ in candidates]


def build_vocab(
    train_token_sequences: Iterable[Sequence[str]],
    min_freq: int = 1,
    max_size: Optional[int] = None,
    pad_token: Optional[str] = "<pad>",
    unk_token: Optional[str] = "<unk>",
    bos_token: Optional[str] = "<bos>",
    eos_token: Optional[str] = "<eos>",
    extra_special_tokens: Sequence[str] = (),
    vocab_factory: Optional[Callable[..., Any]] = None,
) -> Any:
    """Build a vocabulary from tokenized TRAINING examples.

    Args:
        train_token_sequences: Iterable of token lists from the training split only.
        min_freq: Minimum count for a regular token to be kept.
        max_size: Maximum number of regular (non-special) tokens, or None for no limit.
        pad_token/unk_token/bos_token/eos_token/extra_special_tokens: Special tokens
            (None disables one); they receive the leading ids.
        vocab_factory: Optional callable with the ``Vocabulary`` constructor signature.

    Returns:
        A vocabulary instance (``Vocabulary`` by default).
    """
    if vocab_factory is None:
        from data.vocabulary.vocabulary import Vocabulary

        vocab_factory = Vocabulary
    specials = [t for t in (pad_token, unk_token, bos_token, eos_token) if t is not None]
    specials += list(extra_special_tokens)
    tokens = select_tokens(count_tokens(train_token_sequences), min_freq, max_size, exclude=specials)
    return vocab_factory(
        tokens=tokens,
        pad_token=pad_token,
        unk_token=unk_token,
        bos_token=bos_token,
        eos_token=eos_token,
        extra_special_tokens=extra_special_tokens,
    )
