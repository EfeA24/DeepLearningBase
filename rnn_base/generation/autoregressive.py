# Purpose: Provides a general autoregressive generation loop for recurrent models with BOS/EOS/max_length rules and an injected token selector.
"""General autoregressive generation.

Step contract: ``step_fn(tokens (B,) int64, state) -> (logits (B, V), new_state)``.
Selector contract: ``select_fn(logits (B, V)) -> tokens (B,) int64`` (greedy, sampling, ...).
Helpers: ``greedy_select`` and ``make_temperature_sampler(temperature, generator)``.

Start: either ``prompt_ids`` (B, P), or ``bos_id`` + ``batch_size`` (prompt = BOS column).
``max_length`` is the TOTAL output length including the prompt (must be >= P).
Stopping: a row finishes when it emits ``eos_id`` (included in the output); later positions get
``pad_id`` (default ``eos_id``). The loop stops when all rows finish or ``max_length`` is reached.

Output: ``sequences`` (B, L <= max_length); ``lengths`` (B,) = valid length incl. prompt and
EOS; ``finished`` (B,) bool; ``state`` that has consumed every token except the last one in
``sequences`` (feed ``sequences[:, -1]`` first to continue); if nothing is generated
(``max_length == P``) the initial state is returned unchanged.
Runs under ``torch.no_grad()``; model mode/device are the caller's responsibility.

Requires: torch only. No sibling files.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Union

import torch

StepFn = Callable[[torch.Tensor, Any], Tuple[torch.Tensor, Any]]
SelectFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class GenerationOutput:
    sequences: torch.Tensor
    lengths: torch.Tensor
    finished: torch.Tensor
    state: Any


def greedy_select(logits: torch.Tensor) -> torch.Tensor:
    """Argmax over the vocabulary: (B, V) -> (B,)."""
    return logits.argmax(dim=-1)


def make_temperature_sampler(temperature: float = 1.0, generator: Optional[torch.Generator] = None) -> SelectFn:
    """Sample from softmax(logits / temperature); temperature must be > 0 (use greedy_select for 0)."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0; use greedy_select for deterministic decoding.")

    def select(logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits.float() / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)

    return select


def autoregressive_generate(
    step_fn: StepFn,
    select_fn: SelectFn,
    max_length: int,
    prompt_ids: Optional[torch.Tensor] = None,
    bos_id: Optional[int] = None,
    batch_size: Optional[int] = None,
    initial_state: Any = None,
    eos_id: Optional[int] = None,
    pad_id: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> GenerationOutput:
    """Generate sequences autoregressively; see module docstring for the full contract."""
    if prompt_ids is None:
        if bos_id is None or batch_size is None:
            raise ValueError("Provide prompt_ids, or both bos_id and batch_size.")
        prompt_ids = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
    if prompt_ids.dim() != 2 or prompt_ids.shape[1] < 1 or prompt_ids.dtype != torch.long:
        raise ValueError("prompt_ids must be an int64 tensor of shape (B, P) with P >= 1.")
    prompt_length = prompt_ids.shape[1]
    if max_length < prompt_length:
        raise ValueError(f"max_length ({max_length}) must be >= prompt length ({prompt_length}).")

    rows = prompt_ids.shape[0]
    finished = torch.zeros(rows, dtype=torch.bool, device=prompt_ids.device)
    lengths = torch.full((rows,), prompt_length, dtype=torch.long, device=prompt_ids.device)
    fill_id = pad_id if pad_id is not None else eos_id
    state = initial_state
    if max_length == prompt_length:
        return GenerationOutput(prompt_ids, lengths, finished, state)
    tokens = [prompt_ids[:, i] for i in range(prompt_length)]

    with torch.no_grad():
        logits = None
        for position in range(prompt_length):
            logits, state = step_fn(prompt_ids[:, position], state)
        while len(tokens) < max_length:
            next_tokens = select_fn(logits).to(torch.long)
            if next_tokens.shape != (rows,):
                raise ValueError(f"select_fn must return shape ({rows},), got {tuple(next_tokens.shape)}.")
            if eos_id is not None:
                next_tokens = torch.where(finished, torch.full_like(next_tokens, fill_id), next_tokens)
            lengths = lengths + (~finished).long()
            if eos_id is not None:
                finished = finished | (next_tokens == eos_id)
            tokens.append(next_tokens)
            if len(tokens) == max_length or (eos_id is not None and bool(finished.all())):
                break
            logits, state = step_fn(next_tokens, state)

    return GenerationOutput(torch.stack(tokens, dim=1), lengths, finished, state)
