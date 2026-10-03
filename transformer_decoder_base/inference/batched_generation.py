# Purpose: Runs batched decoder-LM generation with explicit left padding, attention masks, finished-sequence handling and preserved output order.
"""Batched generation for prompts of different lengths.

Left padding (required for decoder-only batched generation): prompt ``i`` occupies the LAST
``len(prompt_i)`` columns; ``attention_mask`` is 1 on real tokens. The model callable
(``model_fn(input_ids, past, attention_mask) -> (logits, past)``) must

* accept ``attention_mask`` of shape (B, T_past + T_new) covering cached + new tokens, and
* derive position ids from it (``cumsum(mask) - 1``) so that padding does not shift positions
  (``TransformerDecoderLM`` does this when ``position_ids`` is None).

Every generated column gets mask 1 (also for finished rows, whose new tokens are ``pad_id``
and are discarded). Rows finish at ``eos_id`` (EOS excluded from the output) or after
``max_new_tokens``. Output order equals input order; each output is the list of generated
ids for that prompt (prompt not included).

Equivalence: with greedy selection and a correctly masked model, each row's output equals
running that prompt alone (up to floating-point noise from padding-dependent kernels).

Requires: torch.
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import torch

ModelFn = Callable[[torch.Tensor, Any, Optional[torch.Tensor]], Tuple[torch.Tensor, Any]]


def left_pad(
    prompts: Sequence[Sequence[int]], pad_id: int, device: Union[str, torch.device] = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Left-pad prompts into (B, T) ``input_ids`` and ``attention_mask`` (LongTensor, 1 = real)."""
    if len(prompts) == 0:
        raise ValueError("prompts must not be empty")
    if any(len(p) == 0 for p in prompts):
        raise ValueError("every prompt must contain at least one token")
    longest = max(len(p) for p in prompts)
    input_ids = torch.full((len(prompts), longest), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(prompts), longest), dtype=torch.long)
    for i, p in enumerate(prompts):
        input_ids[i, longest - len(p):] = torch.tensor(list(p), dtype=torch.long)
        attention_mask[i, longest - len(p):] = 1
    return input_ids.to(device), attention_mask.to(device)


def _greedy(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1)


@torch.no_grad()
def batched_generate(
    model_fn: ModelFn,
    prompts: Sequence[Sequence[int]],
    pad_id: int,
    max_new_tokens: int,
    eos_id: Optional[int] = None,
    select_fn: Callable[[torch.Tensor], torch.Tensor] = _greedy,
    device: Union[str, torch.device] = "cpu",
    use_cache: bool = True,
) -> List[List[int]]:
    """Generate for each prompt; returns generated ids per prompt in input order.

    ``select_fn(logits (B, V)) -> (B,)`` chooses the next token (default greedy).
    """
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")
    input_ids, mask = left_pad(prompts, pad_id, device)
    batch = input_ids.shape[0]
    finished = torch.zeros(batch, dtype=torch.bool, device=input_ids.device)
    outputs: List[List[int]] = [[] for _ in range(batch)]
    sequences = input_ids
    step_input = input_ids
    past: Any = None
    for _ in range(max_new_tokens):
        logits, new_past = model_fn(step_input, past if use_cache else None, mask)
        next_token = select_fn(logits[:, -1, :].float()).to(torch.long)
        if eos_id is not None:
            just_finished = (~finished) & (next_token == eos_id)
        else:
            just_finished = torch.zeros_like(finished)
        for i in (~finished & ~just_finished).nonzero(as_tuple=True)[0].tolist():
            outputs[i].append(int(next_token[i]))
        finished = finished | just_finished
        next_token = torch.where(finished, torch.full_like(next_token, pad_id), next_token)
        sequences = torch.cat((sequences, next_token[:, None]), dim=1)
        mask = torch.cat((mask, torch.ones((batch, 1), dtype=mask.dtype, device=mask.device)), dim=1)
        if bool(finished.all()):
            break
        if use_cache:
            past = new_past
            step_input = next_token[:, None]
        else:
            step_input = sequences
    return outputs
