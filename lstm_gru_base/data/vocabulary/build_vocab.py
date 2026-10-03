# Purpose: Builds a token vocabulary from training-split text only with min_freq/max_size limits for LSTM/GRU text projects.
"""Vocabulary construction from TRAINING data.

Contract:
    - Pass only training-split examples; fitting on validation/test text leaks information.
    - ``tokenize`` is a caller-supplied callable ``str -> list[str]`` (or pass pre-tokenized lists).
    - Ordering is deterministic: frequency descending, ties broken by token string ascending.
    - ``max_size`` counts regular tokens only (special tokens are always kept in front).

Requires: data/vocabulary/vocabulary.py (lazy import inside ``build_vocabulary``) unless a
``vocab_factory`` is injected.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Iterable, List, Optional, Sequence, Union

TextOrTokens = Union[str, Sequence[str]]


def count_tokens(
    examples: Iterable[TextOrTokens],
    tokenize: Optional[Callable[[str], List[str]]] = None,
) -> Counter:
    """Count token frequencies; strings require ``tokenize``, token lists are used as-is."""
    counter: Counter = Counter()
    for example in examples:
        if isinstance(example, str):
            if tokenize is None:
                raise ValueError("Raw string examples require a tokenize callable.")
            counter.update(tokenize(example))
        else:
            counter.update(example)
    return counter


def select_tokens(
    counter: Counter,
    min_freq: int = 1,
    max_size: Optional[int] = None,
    exclude: Iterable[str] = (),
) -> List[str]:
    """Filter and order tokens by frequency with deterministic tie-breaking."""
    if min_freq < 1:
        raise ValueError(f"min_freq must be >= 1, got {min_freq}.")
    if max_size is not None and max_size < 0:
        raise ValueError(f"max_size must be >= 0 or None, got {max_size}.")
    excluded = set(exclude)
    items = [(tok, c) for tok, c in counter.items() if c >= min_freq and tok not in excluded]
    items.sort(key=lambda kv: (-kv[1], kv[0]))
    if max_size is not None:
        items = items[:max_size]
    return [tok for tok, _ in items]


def build_vocabulary(
    train_examples: Iterable[TextOrTokens],
    tokenize: Optional[Callable[[str], List[str]]] = None,
    min_freq: int = 1,
    max_size: Optional[int] = None,
    vocab_factory: Optional[Callable[..., Any]] = None,
    **special_token_kwargs: Any,
) -> Any:
    """Build a vocabulary object from training examples.

    Args:
        train_examples: Training-split strings or token lists.
        tokenize: Required when examples are raw strings.
        min_freq: Minimum count to keep a token.
        max_size: Maximum number of regular (non-special) tokens.
        vocab_factory: Callable ``(tokens=list[str], **special_token_kwargs) -> vocab``. Defaults to
            ``data.vocabulary.vocabulary.Vocabulary`` (lazy absolute import from the template root).
            If the produced object has a ``special_tokens`` attribute, those strings are excluded
            from counting so ``max_size`` applies to regular tokens only.
        special_token_kwargs: Forwarded to the factory (e.g. ``pad_token="<pad>"``).
    """
    if vocab_factory is None:
        from data.vocabulary.vocabulary import Vocabulary

        vocab_factory = Vocabulary
    counter = count_tokens(train_examples, tokenize)
    specials = getattr(vocab_factory(tokens=[], **special_token_kwargs), "special_tokens", ())
    tokens = select_tokens(counter, min_freq=min_freq, max_size=max_size, exclude=specials)
    return vocab_factory(tokens=tokens, **special_token_kwargs)
