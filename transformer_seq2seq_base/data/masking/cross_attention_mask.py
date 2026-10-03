# Purpose: Builds seq2seq cross-attention masks that hide padded encoder memory positions from decoder queries, shaped (B,1,T,S) or (B,1,1,S).
"""Cross-attention (decoder query -> encoder memory) masks.

Polarity: True = decoder query may attend to this memory position.
    memory padding mask (B, S): True/1 = real source token, False/0 = pad.
    cross mask          (B, 1, T, S) when ``target_length`` is given, else (B, 1, 1, S)
                        (broadcasts over heads and query positions).
No causal structure applies: every decoder position may see every real source token.

Requires: torch only.
"""
from __future__ import annotations

from typing import Optional

import torch


def make_cross_attention_mask(
    memory_attention_mask: torch.Tensor, target_length: Optional[int] = None
) -> torch.Tensor:
    """(B, S) memory padding mask -> (B, 1, T, S) or (B, 1, 1, S) bool, True = may attend."""
    if memory_attention_mask.dim() != 2:
        raise ValueError(
            f"memory_attention_mask must be (B, S), got {tuple(memory_attention_mask.shape)}"
        )
    mask = memory_attention_mask.to(torch.bool)[:, None, None, :]
    if target_length is None:
        return mask
    if target_length < 1:
        raise ValueError(f"target_length must be >= 1, got {target_length}")
    return mask.expand(-1, 1, target_length, -1)


def expand_memory_mask_for_beams(memory_attention_mask: torch.Tensor, beam_size: int) -> torch.Tensor:
    """Repeat each batch row ``beam_size`` times: (B, S) -> (B * beam_size, S).

    Row order matches ``tensor.repeat_interleave(beam_size, dim=0)`` used for memory.
    """
    if beam_size < 1:
        raise ValueError(f"beam_size must be >= 1, got {beam_size}")
    return memory_attention_mask.repeat_interleave(beam_size, dim=0)
