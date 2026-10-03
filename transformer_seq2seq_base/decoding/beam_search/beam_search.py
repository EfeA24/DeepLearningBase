# Purpose: Implements a seq2seq beam-search loop over a supplied log-prob step function, plus an explicit encode/decode model adapter with cache reordering.
"""Beam search.

Generic loop (no model assumptions):
    step_fn(sequences (B*K, t), state) -> (log_probs (B*K, V), new_state)
        ``log_probs`` must be normalized log-probabilities of the NEXT token; ``state`` is
        anything (e.g. a KV cache) and is reordered with ``reorder_state_fn(state, beam_idx)``
        where beam_idx (B*K,) selects the parent row of each surviving beam.
Rows are laid out batch-major: row b*K + j is beam j of batch element b.

Output: BeamSearchOutput(sequences (B*R, L) int64 starting with the decoder start token,
right-padded with ``pad_token_id``; sequence_scores (B*R,) length-normalized), R =
num_return_sequences, rows b*R .. b*R+R-1 belong to input b, best first.
With beam_size=1 and length_penalty=0 the result equals greedy decoding.

``seq2seq_beam_search`` adapts a model with the encode()/decode() contract of
models/seq2seq/transformer.py (memory repeated per beam; caches reordered).

Requires: torch; sibling decoding/beam_search/beam.py (imported lazily inside beam_search
with the template root on sys.path).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

StepFn = Callable[[torch.Tensor, Any], Tuple[torch.Tensor, Any]]


@dataclass
class BeamSearchOutput:
    sequences: torch.Tensor
    sequence_scores: torch.Tensor


def reorder_nested(state: Any, beam_idx: torch.Tensor) -> Any:
    """index_select dim 0 of every tensor inside nested tuples/lists/dicts."""
    if isinstance(state, torch.Tensor):
        return state.index_select(0, beam_idx.to(state.device))
    if isinstance(state, dict):
        return {k: reorder_nested(v, beam_idx) for k, v in state.items()}
    if isinstance(state, (list, tuple)):
        return type(state)(reorder_nested(v, beam_idx) for v in state)
    return state


@torch.no_grad()
def beam_search(
    step_fn: StepFn,
    *,
    batch_size: int,
    beam_size: int,
    decoder_start_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    max_new_tokens: int,
    device: Union[str, torch.device],
    initial_state: Any = None,
    reorder_state_fn: Optional[Callable[[Any, torch.Tensor], Any]] = None,
    length_penalty: float = 1.0,
    length_penalty_mode: str = "average",
    early_stopping: bool = False,
    num_return_sequences: int = 1,
) -> BeamSearchOutput:
    """Run beam search with a caller-supplied step function (see module docstring)."""
    from decoding.beam_search.beam import BeamHypotheses

    if batch_size < 1 or beam_size < 1 or max_new_tokens < 1:
        raise ValueError("batch_size, beam_size and max_new_tokens must be >= 1")
    if not 1 <= num_return_sequences <= beam_size:
        raise ValueError("num_return_sequences must be in [1, beam_size]")

    k = beam_size
    sequences = torch.full((batch_size * k, 1), decoder_start_token_id, dtype=torch.long, device=device)
    beam_scores = torch.zeros((batch_size, k), dtype=torch.float32, device=device)
    beam_scores[:, 1:] = float("-inf")
    beam_scores = beam_scores.view(-1)
    hypotheses = [
        BeamHypotheses(k, length_penalty, early_stopping, length_penalty_mode) for _ in range(batch_size)
    ]
    done = [False] * batch_size
    state = initial_state

    for step in range(max_new_tokens):
        log_probs, state = step_fn(sequences, state)
        if log_probs.dim() != 2 or log_probs.size(0) != batch_size * k:
            raise ValueError(f"step_fn must return (B*K, V) log-probs, got {tuple(log_probs.shape)}")
        vocab_size = log_probs.size(1)
        next_scores = (log_probs.float() + beam_scores.unsqueeze(1)).view(batch_size, k * vocab_size)
        num_candidates = min(2 * k, k * vocab_size)
        top_scores, top_indices = next_scores.topk(num_candidates, dim=1)
        top_beams = (top_indices // vocab_size).tolist()
        top_tokens = (top_indices % vocab_size).tolist()
        top_scores_list = top_scores.tolist()
        generated_length = step + 1

        new_scores = torch.full((batch_size, k), float("-inf"), dtype=torch.float32)
        new_tokens = torch.full((batch_size, k), pad_token_id, dtype=torch.long)
        new_parents = torch.zeros((batch_size, k), dtype=torch.long)
        for b in range(batch_size):
            new_parents[b] = torch.arange(b * k, (b + 1) * k)
            if done[b]:
                new_scores[b] = 0.0
                continue
            slot = 0
            for rank, (score, beam, token) in enumerate(zip(top_scores_list[b], top_beams[b], top_tokens[b])):
                if score == float("-inf"):
                    break
                parent = b * k + beam
                if token == eos_token_id:
                    if rank < k:
                        finished_ids = torch.cat([sequences[parent], sequences.new_tensor([token])])
                        hypotheses[b].add(finished_ids, score, generated_length)
                else:
                    new_scores[b, slot] = score
                    new_tokens[b, slot] = token
                    new_parents[b, slot] = parent
                    slot += 1
                if slot == k:
                    break
            done[b] = hypotheses[b].is_done(float(new_scores[b].max()), generated_length)

        beam_scores = new_scores.view(-1).to(device)
        parents = new_parents.view(-1).to(device)
        sequences = torch.cat([sequences.index_select(0, parents), new_tokens.view(-1, 1).to(device)], dim=1)
        if reorder_state_fn is not None and state is not None:
            state = reorder_state_fn(state, parents)
        if all(done):
            break

    final_length = sequences.size(1) - 1
    for b in range(batch_size):
        if done[b]:
            continue
        for j in range(k):
            row = b * k + j
            score = float(beam_scores[row])
            if score != float("-inf"):
                hypotheses[b].add(sequences[row], score, final_length)

    selected: List[torch.Tensor] = []
    selected_scores: List[float] = []
    for b in range(batch_size):
        best = hypotheses[b].best(num_return_sequences)
        if len(best) < num_return_sequences:
            raise RuntimeError(f"batch element {b} produced fewer than {num_return_sequences} hypotheses")
        for score, tokens in best:
            selected.append(tokens)
            selected_scores.append(score)
    max_len = max(t.numel() for t in selected)
    out = torch.full((len(selected), max_len), pad_token_id, dtype=torch.long, device=device)
    for row, tokens in enumerate(selected):
        out[row, : tokens.numel()] = tokens
    return BeamSearchOutput(sequences=out, sequence_scores=torch.tensor(selected_scores, device=device))


@torch.no_grad()
def seq2seq_beam_search(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    *,
    beam_size: int,
    decoder_start_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    max_new_tokens: int,
    length_penalty: float = 1.0,
    early_stopping: bool = False,
    num_return_sequences: int = 1,
    use_cache: bool = True,
) -> BeamSearchOutput:
    """Beam search for a model with ``encode(ids, mask)`` / ``decode(...) -> (logits, past)``.

    The caller is responsible for ``model.eval()`` and device placement.
    """
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must be (B, S), got {tuple(input_ids.shape)}")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    memory = model.encode(input_ids, attention_mask).repeat_interleave(beam_size, dim=0)
    memory_mask = attention_mask.repeat_interleave(beam_size, dim=0)

    def step_fn(sequences: torch.Tensor, past: Any) -> Tuple[torch.Tensor, Any]:
        step_input = sequences[:, -1:] if (use_cache and past is not None) else sequences
        logits, present = model.decode(
            step_input, memory, memory_attention_mask=memory_mask,
            past_key_values=past, use_cache=use_cache,
        )
        return F.log_softmax(logits[:, -1, :].float(), dim=-1), (present if use_cache else None)

    return beam_search(
        step_fn,
        batch_size=input_ids.size(0),
        beam_size=beam_size,
        decoder_start_token_id=decoder_start_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        max_new_tokens=max_new_tokens,
        device=input_ids.device,
        reorder_state_fn=reorder_nested if use_cache else None,
        length_penalty=length_penalty,
        early_stopping=early_stopping,
        num_return_sequences=num_return_sequences,
    )
