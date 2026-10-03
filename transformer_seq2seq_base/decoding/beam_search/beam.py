# Purpose: Holds finished beam-search hypotheses for one seq2seq input with log-prob score accumulation, length normalization (alpha) and a sticky finished flag.
"""Beam hypothesis container.

Each hypothesis stores (normalized_score, token_ids, sum_logprobs). Scores:
    mode="average": score = sum_logprobs / length ** alpha
    mode="gnmt":    score = sum_logprobs / ((5 + length) / 6) ** alpha   (Wu et al., 2016)
``length`` = number of GENERATED tokens (decoder start excluded, EOS included).
alpha = 0 -> raw sum of log-probs (favors short outputs); alpha = 1 -> per-token average.

Done rule (heuristic, same idea as common beam-search implementations): with
``num_beams`` finished hypotheses, stop if ``early_stopping`` or if the worst kept score is
>= the best running beam's score normalized at the current length. ``finished`` is sticky.

Requires: torch only.
"""
from __future__ import annotations

from typing import List, Tuple

import torch


def length_normalized_score(sum_logprobs: float, length: int, alpha: float, mode: str = "average") -> float:
    """Apply the length penalty to a summed log-probability."""
    if length < 1:
        raise ValueError("length must be >= 1")
    if mode == "average":
        return sum_logprobs / (length ** alpha)
    if mode == "gnmt":
        return sum_logprobs / (((5.0 + length) / 6.0) ** alpha)
    raise ValueError(f"mode must be 'average' or 'gnmt', got {mode!r}")


class BeamHypotheses:
    """Top-``num_beams`` finished hypotheses for one batch element."""

    def __init__(
        self,
        num_beams: int,
        length_penalty: float = 1.0,
        early_stopping: bool = False,
        length_penalty_mode: str = "average",
    ) -> None:
        if num_beams < 1:
            raise ValueError("num_beams must be >= 1")
        length_normalized_score(0.0, 1, length_penalty, length_penalty_mode)
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.length_penalty_mode = length_penalty_mode
        self.beams: List[Tuple[float, torch.Tensor, float]] = []
        self.worst_score = float("inf")
        self.finished = False

    def __len__(self) -> int:
        return len(self.beams)

    def score(self, sum_logprobs: float, length: int) -> float:
        return length_normalized_score(sum_logprobs, length, self.length_penalty, self.length_penalty_mode)

    def add(self, token_ids: torch.Tensor, sum_logprobs: float, generated_length: int) -> None:
        """Offer a finished hypothesis; kept only if it ranks within the top ``num_beams``."""
        score = self.score(sum_logprobs, generated_length)
        if len(self.beams) < self.num_beams or score > self.worst_score:
            self.beams.append((score, token_ids.detach().clone(), sum_logprobs))
            if len(self.beams) > self.num_beams:
                worst_index = min(range(len(self.beams)), key=lambda i: self.beams[i][0])
                self.beams.pop(worst_index)
            self.worst_score = min(b[0] for b in self.beams)

    def is_done(self, best_running_sum_logprobs: float, current_length: int) -> bool:
        """Update and return the sticky ``finished`` flag."""
        if self.finished:
            return True
        if best_running_sum_logprobs == float("-inf") and self.beams:
            self.finished = True
            return True
        if len(self.beams) < self.num_beams:
            return False
        if self.early_stopping:
            self.finished = True
        else:
            self.finished = self.worst_score >= self.score(best_running_sum_logprobs, current_length)
        return self.finished

    def best(self, n: int = 1) -> List[Tuple[float, torch.Tensor]]:
        """Return up to ``n`` (score, token_ids) pairs sorted by descending score."""
        ranked = sorted(self.beams, key=lambda b: b[0], reverse=True)
        return [(score, tokens) for score, tokens, _ in ranked[:n]]
