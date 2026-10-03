# Purpose: Orchestrates decoder-LM generation from a supplied model callable, logits processors, sampler, stopping rules and optional KV cache.
"""Generic autoregressive generation.

Model callable contract (see generation/greedy/greedy_decode.py):
    model_fn(input_ids, past, attention_mask) -> (logits (B, T_new, V), new_past)
With ``use_cache=True`` the prompt is fed once, then one token per step together with the
returned ``past``; with ``use_cache=False`` the full sequence is re-fed every step.

Per step:
    logits = model logits at the last position, (B, V), cast to float32
    for proc in logits_processors: logits = proc(sequences, logits)   # e.g. temperature, top-k, top-p
    next   = sampler(logits)  -> (B,)   (default: argmax = greedy)
    finished rows receive pad_token_id instead of a new token
    a row finishes on eos_token_id or when any stopping criterion returns True for it

Processor order matters: the usual order is temperature -> top-k -> top-p.

Stopping criteria: ``criterion(sequences (B, T), new_tokens (B,)) -> BoolTensor (B,)`` (True = this
row is done). ``StopOnTokenSequence`` is provided. ``max_new_tokens`` always bounds the loop.

Batching: equal-length prompts, or LEFT-padded prompts with ``attention_mask`` (1 = real).
Returns :class:`GenerationResult` with full sequences, the generated part, and decoded texts
when a ``decode`` callable (e.g. ``tokenizer.decode``) is supplied.

Requires: torch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

import torch

ModelFn = Callable[[torch.Tensor, Any, Optional[torch.Tensor]], Tuple[torch.Tensor, Any]]
LogitsProcessor = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
Sampler = Callable[[torch.Tensor], torch.Tensor]
StoppingCriterion = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class GenerationResult:
    """sequences (B, T_prompt + N); new_tokens (B, N) incl. fill ids after a row finished;
    finished (B,) bool; lengths (B,) real generated tokens per row (incl. EOS/stop token);
    texts decoded from the first ``lengths`` tokens without a trailing EOS (if decode given)."""

    sequences: torch.Tensor
    new_tokens: torch.Tensor
    finished: torch.Tensor
    lengths: torch.Tensor
    texts: Optional[List[str]] = None


def greedy_sampler(logits: torch.Tensor) -> torch.Tensor:
    """Argmax over the vocabulary: (B, V) -> (B,)."""
    return logits.argmax(dim=-1)


class MultinomialSampler:
    """Sample from softmax(logits) with an optional explicit ``torch.Generator``."""

    def __init__(self, generator: Optional[torch.Generator] = None) -> None:
        self.generator = generator

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits.float(), dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=self.generator).squeeze(-1)


class StopOnTokenSequence:
    """Marks a row done when its sequence ends with ``stop_ids``."""

    def __init__(self, stop_ids: Sequence[int]) -> None:
        if len(stop_ids) == 0:
            raise ValueError("stop_ids must be non-empty")
        self.stop_ids = list(stop_ids)

    def __call__(self, sequences: torch.Tensor, new_tokens: torch.Tensor) -> torch.Tensor:
        n = len(self.stop_ids)
        if sequences.shape[1] < n:
            return torch.zeros(sequences.shape[0], dtype=torch.bool, device=sequences.device)
        target = torch.tensor(self.stop_ids, dtype=sequences.dtype, device=sequences.device)
        return (sequences[:, -n:] == target).all(dim=-1)


@torch.no_grad()
def generate(
    model_fn: ModelFn,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    logits_processors: Sequence[LogitsProcessor] = (),
    sampler: Sampler = greedy_sampler,
    stopping_criteria: Sequence[StoppingCriterion] = (),
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    attention_mask: Optional[torch.Tensor] = None,
    use_cache: bool = True,
    decode: Optional[Callable[[List[int]], str]] = None,
) -> GenerationResult:
    """Generate up to ``max_new_tokens`` tokens for a (B, T) prompt batch."""
    if input_ids.dim() != 2 or input_ids.shape[1] == 0:
        raise ValueError(f"input_ids must be a non-empty (B, T) tensor, got {tuple(input_ids.shape)}")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")
    if attention_mask is not None and attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must match input_ids shape")
    fill_id = pad_token_id if pad_token_id is not None else eos_token_id
    if fill_id is None and (stopping_criteria or eos_token_id is not None):
        raise ValueError("pad_token_id (or eos_token_id) is required to fill finished rows")

    batch, prompt_len = input_ids.shape
    sequences = input_ids
    mask = attention_mask
    finished = torch.zeros(batch, dtype=torch.bool, device=input_ids.device)
    lengths = torch.zeros(batch, dtype=torch.long, device=input_ids.device)
    past: Any = None
    step_input = input_ids
    for _ in range(max_new_tokens):
        logits, new_past = model_fn(step_input, past if use_cache else None, mask)
        if logits.dim() != 3 or logits.shape[0] != batch:
            raise ValueError(f"model_fn must return logits (B, T, V), got {tuple(logits.shape)}")
        next_logits = logits[:, -1, :].float()
        for processor in logits_processors:
            next_logits = processor(sequences, next_logits)
        next_token = sampler(next_logits).to(sequences.dtype)
        if fill_id is not None:
            next_token = torch.where(finished, torch.full_like(next_token, fill_id), next_token)
        sequences = torch.cat((sequences, next_token[:, None]), dim=1)
        lengths += (~finished).long()
        if mask is not None:
            mask = torch.cat((mask, torch.ones((batch, 1), dtype=mask.dtype, device=mask.device)), dim=1)

        newly_done = torch.zeros_like(finished)
        if eos_token_id is not None:
            newly_done |= next_token == eos_token_id
        for criterion in stopping_criteria:
            newly_done |= criterion(sequences, next_token).to(torch.bool)
        finished = finished | newly_done

        if use_cache:
            past = new_past
            step_input = next_token[:, None]
        else:
            step_input = sequences
        if bool(finished.all()):
            break

    new_tokens = sequences[:, prompt_len:]
    texts = None
    if decode is not None:
        texts = []
        for row, length in zip(new_tokens.tolist(), lengths.tolist()):
            row = row[:length]
            if eos_token_id is not None and row and row[-1] == eos_token_id:
                row = row[:-1]
            texts.append(decode(row))
    return GenerationResult(
        sequences=sequences, new_tokens=new_tokens, finished=finished, lengths=lengths, texts=texts
    )
