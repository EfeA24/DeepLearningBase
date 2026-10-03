# Purpose: Applies rotary position embeddings (RoPE) to decoder query/key heads with KV-cache-aware positions.
"""Apply RoPE to q/k tensors using the ``rotate_half`` convention.

Shapes:
    q: (B, H_q, T, D)    k: (B, H_kv, T, D)    (H_kv may differ from H_q for GQA)
    cos, sin: (T, D) or (B, T, D) built with the duplicated-halves layout
              (see ``rope_utils.py``: emb = cat([freqs, freqs], -1)).

    rotate_half(x) = cat([-x[..., D/2:], x[..., :D/2]], -1)
    x_rot          = x * cos + rotate_half(x) * sin

Positions are ABSOLUTE token positions. During incremental decoding with a KV cache
the new tokens must use positions ``past_len .. past_len + T - 1`` (``default_position_ids``
builds exactly that); for left-padded batches derive positions from the attention mask
instead (``cumsum(mask) - 1``). Keys are rotated before they are written to the cache,
so cached keys never need re-rotation.

Properties (used in tests): rotation preserves per-head vector norms, and
``<rope(q, m), rope(k, n)>`` depends only on ``m - n``.

``RoPE`` is the callable injected into the attention modules of this template:
    rope(q, k, position_ids) -> (q_rot, k_rot)
It wraps any provider ``cos_sin_provider(position_ids, dtype) -> (cos, sin)``, e.g.
``models.embeddings.rotary.rope_utils.RotaryEmbedding`` (dependency injection; this file
does not import it).

Requires: torch.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
from torch import nn

CosSinProvider = Callable[[torch.Tensor, Optional[torch.dtype]], Tuple[torch.Tensor, torch.Tensor]]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of the last dimension: (x1, x2) -> (-x2, x1)."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _broadcast_cos_sin(cos: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if cos.dim() == 2:
        return cos.unsqueeze(0).unsqueeze(0)
    if cos.dim() == 3:
        return cos.unsqueeze(1)
    raise ValueError(f"cos/sin must be (T, D) or (B, T, D), got shape {tuple(cos.shape)}")


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate one tensor ``x`` of shape (B, H, T, D); returns the same shape and dtype."""
    if x.dim() != 4:
        raise ValueError(f"x must be (B, H, T, D), got shape {tuple(x.shape)}")
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {x.shape[-1]}")
    if cos.shape[-1] != x.shape[-1] or cos.shape[-2] != x.shape[-2]:
        raise ValueError(
            f"cos/sin (..., T={cos.shape[-2]}, D={cos.shape[-1]}) do not match x T={x.shape[-2]}, D={x.shape[-1]}"
        )
    cos_b = _broadcast_cos_sin(cos, x).to(x.dtype)
    sin_b = _broadcast_cos_sin(sin, x).to(x.dtype)
    return x * cos_b + rotate_half(x) * sin_b


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rotate queries and keys with the same cos/sin tables."""
    return apply_rope(q, cos, sin), apply_rope(k, cos, sin)


def default_position_ids(
    batch_size: int, seq_len: int, past_len: int = 0, device: Optional[torch.device] = None
) -> torch.Tensor:
    """Return (B, T) positions ``past_len .. past_len + seq_len - 1`` (no padding assumed)."""
    if past_len < 0:
        raise ValueError(f"past_len must be >= 0, got {past_len}")
    positions = torch.arange(past_len, past_len + seq_len, dtype=torch.long, device=device)
    return positions.unsqueeze(0).expand(batch_size, seq_len)


class RoPE(nn.Module):
    """Callable ``(q, k, position_ids) -> (q_rot, k_rot)`` for injection into attention modules.

    Args:
        cos_sin_provider: Module/callable mapping ``(position_ids, dtype)`` to ``(cos, sin)``.
            If it is an ``nn.Module`` it is registered as a submodule (moves with ``.to()``).
    """

    def __init__(self, cos_sin_provider: CosSinProvider) -> None:
        super().__init__()
        if not callable(cos_sin_provider):
            raise TypeError("cos_sin_provider must be callable")
        self.cos_sin_provider = cos_sin_provider

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if position_ids.shape[-1] != q.shape[-2]:
            raise ValueError(
                f"position_ids length {position_ids.shape[-1]} does not match query length {q.shape[-2]}"
            )
        cos, sin = self.cos_sin_provider(position_ids, q.dtype)
        return apply_rotary_pos_emb(q, k, cos.to(q.device), sin.to(q.device))
