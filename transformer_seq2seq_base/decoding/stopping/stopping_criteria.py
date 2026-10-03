# Purpose: Provides composable seq2seq generation stopping criteria (max length, max new tokens, EOS, custom callables) returning per-sequence done flags.
"""Stopping criteria for autoregressive decoding.

Contract: ``criterion(sequences) -> done`` where
    sequences: (B, L) int64 decoder sequences INCLUDING the decoder start token(s);
    done:      (B,) bool, True = this sequence should stop.
Decoding loops keep finished flags sticky (once True, stays True) and pad afterwards.
``EosTokenCriterion`` ignores the first ``prefix_length`` positions so a decoder start
token that equals EOS (mBART style) does not stop generation immediately.

Requires: torch only.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from typing import List, Union

import torch


class StoppingCriterion(ABC):
    """Base class: return a (B,) bool tensor, True = stop."""

    @abstractmethod
    def __call__(self, sequences: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def _check(sequences: torch.Tensor) -> None:
    if sequences.dim() != 2:
        raise ValueError(f"sequences must be (B, L), got {tuple(sequences.shape)}")


class MaxLengthCriterion(StoppingCriterion):
    """Stop when total length (including start token) reaches ``max_length``."""

    def __init__(self, max_length: int) -> None:
        if max_length < 1:
            raise ValueError("max_length must be >= 1")
        self.max_length = max_length

    def __call__(self, sequences: torch.Tensor) -> torch.Tensor:
        _check(sequences)
        return torch.full((sequences.size(0),), sequences.size(1) >= self.max_length,
                          dtype=torch.bool, device=sequences.device)


class MaxNewTokensCriterion(StoppingCriterion):
    """Stop after ``max_new_tokens`` tokens beyond ``start_length`` prefix positions."""

    def __init__(self, start_length: int, max_new_tokens: int) -> None:
        if start_length < 0 or max_new_tokens < 1:
            raise ValueError("start_length must be >= 0 and max_new_tokens >= 1")
        self.start_length = start_length
        self.max_new_tokens = max_new_tokens

    def __call__(self, sequences: torch.Tensor) -> torch.Tensor:
        _check(sequences)
        done = sequences.size(1) - self.start_length >= self.max_new_tokens
        return torch.full((sequences.size(0),), done, dtype=torch.bool, device=sequences.device)


class EosTokenCriterion(StoppingCriterion):
    """Stop sequences that contain any EOS id after the first ``prefix_length`` positions."""

    def __init__(self, eos_token_ids: Union[int, Sequence[int]], prefix_length: int = 1) -> None:
        ids = [eos_token_ids] if isinstance(eos_token_ids, int) else list(eos_token_ids)
        if not ids:
            raise ValueError("at least one EOS id is required")
        if prefix_length < 0:
            raise ValueError("prefix_length must be >= 0")
        self.eos_token_ids = [int(i) for i in ids]
        self.prefix_length = prefix_length

    def __call__(self, sequences: torch.Tensor) -> torch.Tensor:
        _check(sequences)
        generated = sequences[:, self.prefix_length:]
        eos = torch.tensor(self.eos_token_ids, dtype=sequences.dtype, device=sequences.device)
        return torch.isin(generated, eos).any(dim=1)


class CallableCriterion(StoppingCriterion):
    """Adapts a custom ``fn(sequences) -> (B,) bool`` (e.g. stop on a token pattern)."""

    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        if not callable(fn):
            raise TypeError("fn must be callable")
        self.fn = fn

    def __call__(self, sequences: torch.Tensor) -> torch.Tensor:
        done = self.fn(sequences)
        if not isinstance(done, torch.Tensor) or done.shape != (sequences.size(0),):
            raise ValueError("custom stopping function must return a (B,) tensor")
        return done.to(torch.bool)


class StoppingCriteriaList(StoppingCriterion):
    """Combine criteria with ``mode="any"`` (default, OR) or ``mode="all"`` (AND)."""

    def __init__(self, criteria: Iterable[Callable[[torch.Tensor], torch.Tensor]], mode: str = "any") -> None:
        self.criteria: List[Callable[[torch.Tensor], torch.Tensor]] = list(criteria)
        if not self.criteria:
            raise ValueError("at least one criterion is required")
        if mode not in ("any", "all"):
            raise ValueError(f"mode must be 'any' or 'all', got {mode!r}")
        self.mode = mode

    def __call__(self, sequences: torch.Tensor) -> torch.Tensor:
        results = torch.stack([c(sequences).to(torch.bool) for c in self.criteria], dim=0)
        return results.any(dim=0) if self.mode == "any" else results.all(dim=0)
