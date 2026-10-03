# Purpose: Processes seq2seq source token ids (optional prefix, truncation side, EOS appending) under an explicit max_length contract.
"""Source-side processing for encoder inputs.

Output layout of one example: ``prefix_ids + body (truncated) + [EOS]`` with total
length <= ``max_length``. The supplied ``encode_fn`` must return plain token ids WITHOUT
special tokens; EOS is added here so the policy lives in one place.

Requires: pure Python (no third-party packages).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import List, Optional


def truncate_ids(token_ids: Sequence[int], max_length: int, side: str = "right") -> List[int]:
    """Truncate a token list to ``max_length``.

    ``side="right"`` keeps the beginning, ``side="left"`` keeps the end.
    """
    if max_length < 0:
        raise ValueError(f"max_length must be >= 0, got {max_length}")
    if side not in ("right", "left"):
        raise ValueError(f"side must be 'right' or 'left', got {side!r}")
    ids = [int(t) for t in token_ids]
    if len(ids) <= max_length:
        return ids
    return ids[:max_length] if side == "right" else ids[len(ids) - max_length:]


def process_source_ids(
    token_ids: Sequence[int],
    *,
    max_length: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    add_eos: bool = True,
    prefix_ids: Sequence[int] = (),
    truncation_side: str = "right",
) -> List[int]:
    """Apply prefix, truncation and EOS to one source sequence.

    Args:
        token_ids: body token ids without special tokens. A trailing EOS is tolerated
            and not duplicated.
        max_length: maximum total length including prefix and EOS (None = no limit).
        eos_token_id: EOS id; required when ``add_eos``.
        add_eos: append EOS after the (truncated) body.
        prefix_ids: ids always kept at the front (e.g. an encoded task prefix).
        truncation_side: which side of the BODY to cut ("right" keeps the start).
    """
    if add_eos and eos_token_id is None:
        raise ValueError("add_eos=True requires eos_token_id")
    body = [int(t) for t in token_ids]
    if add_eos and body and body[-1] == eos_token_id:
        body = body[:-1]
    prefix = [int(t) for t in prefix_ids]
    reserved = len(prefix) + (1 if add_eos else 0)
    if max_length is not None:
        if max_length < reserved:
            raise ValueError(
                f"max_length={max_length} cannot hold prefix ({len(prefix)}) + EOS ({int(add_eos)})"
            )
        body = truncate_ids(body, max_length - reserved, truncation_side)
    out = prefix + body
    if add_eos:
        out.append(int(eos_token_id))
    return out


def process_source_text(
    text: str,
    encode_fn: Callable[[str], Sequence[int]],
    **kwargs,
) -> List[int]:
    """Encode ``text`` with ``encode_fn`` and apply :func:`process_source_ids` options."""
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    return process_source_ids(encode_fn(text), **kwargs)


def process_source_batch(
    texts: Sequence[str],
    encode_fn: Callable[[str], Sequence[int]],
    **kwargs,
) -> List[List[int]]:
    """Process several source texts; order is preserved, padding is the collator's job."""
    return [process_source_text(t, encode_fn, **kwargs) for t in texts]
