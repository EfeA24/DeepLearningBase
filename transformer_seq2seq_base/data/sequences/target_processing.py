# Purpose: Processes seq2seq target token ids into EOS-terminated labels and decoder-start-prefixed decoder inputs under a max_length contract.
"""Target-side processing for decoder training data.

For target body ``y1..yn`` (encode_fn output WITHOUT special tokens):
    labels            = truncate(y, max_length - 1) + [EOS]
    decoder_input_ids = [decoder_start] + labels[:-1]
Both lists have equal length <= ``max_length``. Padding / -100 replacement is done by the
collator (data/collators/seq2seq_collator.py).

Requires: pure Python (no third-party packages).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class TargetFeatures:
    """Unpadded decoder features for one target sequence."""

    labels: List[int]
    decoder_input_ids: List[int]


def process_target_ids(
    token_ids: Sequence[int],
    *,
    decoder_start_token_id: int,
    eos_token_id: Optional[int],
    max_length: Optional[int] = None,
    add_eos: bool = True,
    truncation_side: str = "right",
) -> TargetFeatures:
    """Build labels and decoder inputs for one target.

    Args:
        token_ids: target ids without decoder start; a trailing EOS is not duplicated.
        decoder_start_token_id: first decoder input token (BOS or pad-as-start).
        eos_token_id: EOS id appended to labels when ``add_eos``.
        max_length: maximum label length including EOS (None = no limit).
        add_eos: append EOS to labels.
        truncation_side: "right" keeps the start of the target (usual), "left" keeps the end.
    """
    if add_eos and eos_token_id is None:
        raise ValueError("add_eos=True requires eos_token_id")
    if truncation_side not in ("right", "left"):
        raise ValueError(f"truncation_side must be 'right' or 'left', got {truncation_side!r}")
    body = [int(t) for t in token_ids]
    if add_eos and body and body[-1] == eos_token_id:
        body = body[:-1]
    if max_length is not None:
        budget = max_length - (1 if add_eos else 0)
        if budget < 0:
            raise ValueError(f"max_length={max_length} too small to hold EOS")
        if len(body) > budget:
            body = body[:budget] if truncation_side == "right" else body[len(body) - budget:]
    labels = body + ([int(eos_token_id)] if add_eos else [])
    if not labels:
        raise ValueError("target is empty after processing; cannot build decoder features")
    return TargetFeatures(labels=labels, decoder_input_ids=[int(decoder_start_token_id)] + labels[:-1])


def process_target_text(
    text: str,
    encode_fn: Callable[[str], Sequence[int]],
    **kwargs,
) -> TargetFeatures:
    """Encode ``text`` and apply :func:`process_target_ids`."""
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    return process_target_ids(encode_fn(text), **kwargs)
