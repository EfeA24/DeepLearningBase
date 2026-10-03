# Purpose: Produces a text completion from caller-prepared prompt ids using an already initialized decoder LM callable and optional tokenizer decode.
"""Single-prompt completion.

Inputs are prepared by the caller: ``prompt_ids`` (list[int] or LongTensor (T,) / (1, T)) already
tokenized (BOS included if the model expects it), a model callable already on its device
(``model_fn(input_ids, past, attention_mask) -> (logits, past)``, e.g.
``TransformerDecoderLM.generation_step``), and the ``device`` where ``input_ids`` should be created.
The model is never moved or loaded here.

Token selection: ``select_fn(logits (1, V)) -> LongTensor (1,)``; default greedy argmax. Compose
sampling yourself, e.g. ``lambda l: sample_top_p(apply_temperature(l, 0.8), 0.9, generator=g)``.

Stopping: ``eos_token_id`` (excluded from ``completion_ids``), ``stop_sequences`` (lists of ids;
the matched stop sequence is removed from the output), or ``max_new_tokens``.

Returns :class:`CompletionResult` (prompt ids, completion ids, decoded text if a ``decode``
callable is supplied, and the stop reason "eos" | "stop_sequence" | "length").

Requires: torch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import torch

ModelFn = Callable[[torch.Tensor, Any, Optional[torch.Tensor]], Tuple[torch.Tensor, Any]]


@dataclass
class CompletionResult:
    prompt_ids: List[int]
    completion_ids: List[int]
    text: Optional[str]
    stop_reason: str


def _greedy(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1)


@torch.no_grad()
def complete(
    model_fn: ModelFn,
    prompt_ids: Union[Sequence[int], torch.Tensor],
    max_new_tokens: int,
    device: Union[str, torch.device] = "cpu",
    select_fn: Callable[[torch.Tensor], torch.Tensor] = _greedy,
    eos_token_id: Optional[int] = None,
    stop_sequences: Sequence[Sequence[int]] = (),
    decode: Optional[Callable[[List[int]], str]] = None,
    use_cache: bool = True,
) -> CompletionResult:
    """Generate a completion for one prompt (see module docstring)."""
    if isinstance(prompt_ids, torch.Tensor):
        if prompt_ids.dim() == 2 and prompt_ids.shape[0] == 1:
            prompt_ids = prompt_ids[0]
        if prompt_ids.dim() != 1:
            raise ValueError("prompt_ids tensor must be (T,) or (1, T)")
        prompt = [int(t) for t in prompt_ids.tolist()]
    else:
        prompt = [int(t) for t in prompt_ids]
    if not prompt:
        raise ValueError("prompt_ids must not be empty")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")
    stops = [list(s) for s in stop_sequences if len(s) > 0]

    sequence = torch.tensor([prompt], dtype=torch.long, device=device)
    step_input = sequence
    past: Any = None
    generated: List[int] = []
    stop_reason = "length"
    for _ in range(max_new_tokens):
        logits, new_past = model_fn(step_input, past if use_cache else None, None)
        token = select_fn(logits[:, -1, :].float()).reshape(1).to(torch.long)
        token_id = int(token.item())
        if eos_token_id is not None and token_id == eos_token_id:
            stop_reason = "eos"
            break
        generated.append(token_id)
        matched = next((s for s in stops if generated[-len(s):] == s), None)
        if matched is not None:
            del generated[-len(matched):]
            stop_reason = "stop_sequence"
            break
        sequence = torch.cat((sequence, token.view(1, 1).to(sequence.device)), dim=1)
        if use_cache:
            past = new_past
            step_input = sequence[:, -1:]
        else:
            step_input = sequence
    text = decode(generated) if decode is not None else None
    return CompletionResult(prompt_ids=prompt, completion_ids=generated, text=text, stop_reason=stop_reason)
