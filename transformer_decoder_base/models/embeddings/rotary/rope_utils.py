# Purpose: Builds rotary-embedding (RoPE) inverse frequencies and cos/sin tables for causal decoder attention.
"""RoPE frequency and cos/sin table construction.

Convention (matches ``rope.py`` / ``rotate_half``, i.e. the GPT-NeoX / Llama-HF layout):
    inv_freq[i] = base ** (-2i / head_dim),  i = 0 .. head_dim/2 - 1      shape (head_dim/2,)
    freqs[p, i] = p * inv_freq[i]                                         shape (P, head_dim/2)
    emb         = cat([freqs, freqs], dim=-1)                             shape (P, head_dim)
    cos, sin    = emb.cos(), emb.sin()

So dimension ``j`` and ``j + head_dim/2`` share the same frequency. This is NOT the
interleaved (GPT-J / original-paper even/odd pair) layout; weights converted from an
interleaved implementation need a permutation of q/k projection rows.

``RotaryEmbedding`` holds its table as non-persistent buffers on the instance (no global
cache). It is precomputed up to ``max_positions`` at construction and grown on demand if
larger positions are requested. Tables are computed in float32 and cast to the requested
dtype on lookup.

Lookup contract:
    position_ids: LongTensor (T,) or (B, T); absolute positions (include the KV-cache offset)
    returns cos, sin: (T, head_dim) or (B, T, head_dim)

Requires: torch.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn


def build_inv_freq(
    head_dim: int, base: float = 10000.0, device: Optional[torch.device] = None
) -> torch.Tensor:
    """Return float32 inverse frequencies of shape ``(head_dim // 2,)``."""
    if head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError(f"head_dim must be a positive even integer, got {head_dim}")
    if base <= 0:
        raise ValueError(f"base (theta) must be positive, got {base}")
    exponents = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim
    return 1.0 / (base**exponents)


def build_cos_sin(
    positions: torch.Tensor, inv_freq: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute float32 cos/sin tables for arbitrary positions.

    Args:
        positions: (...,) integer or float positions.
        inv_freq: (head_dim // 2,) from :func:`build_inv_freq`.

    Returns:
        cos, sin of shape (..., head_dim) using the duplicated-halves layout.
    """
    freqs = positions.to(torch.float32).unsqueeze(-1) * inv_freq.to(torch.float32)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


class RotaryEmbedding(nn.Module):
    """Instance-held cos/sin cache for RoPE.

    Args:
        head_dim: Per-head dimension (even).
        max_positions: Positions precomputed at construction (table grows if exceeded).
        base: RoPE theta (10000 in the original paper; larger values for long context).
    """

    def __init__(self, head_dim: int, max_positions: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        if max_positions <= 0:
            raise ValueError(f"max_positions must be positive, got {max_positions}")
        self.head_dim = head_dim
        self.base = float(base)
        self.register_buffer("inv_freq", build_inv_freq(head_dim, base), persistent=False)
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        self._build_cache(max_positions)

    @property
    def cached_positions(self) -> int:
        return int(self.cos_cached.shape[0])

    def _build_cache(self, num_positions: int) -> None:
        positions = torch.arange(num_positions, device=self.inv_freq.device)
        cos, sin = build_cos_sin(positions, self.inv_freq)
        self.cos_cached = cos
        self.sin_cached = sin

    def forward(
        self, position_ids: torch.Tensor, dtype: Optional[torch.dtype] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Look up cos/sin for ``position_ids`` ((T,) or (B, T)); returns matching (..., head_dim)."""
        if position_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"position_ids must be an integer tensor, got {position_ids.dtype}")
        if position_ids.numel() and int(position_ids.min()) < 0:
            raise ValueError("position_ids must be non-negative")
        needed = int(position_ids.max()) + 1 if position_ids.numel() else 0
        if needed > self.cached_positions or self.cos_cached.device != self.inv_freq.device:
            self._build_cache(max(needed, self.cached_positions))
        position_ids = position_ids.to(self.cos_cached.device)
        cos = self.cos_cached[position_ids]
        sin = self.sin_cached[position_ids]
        if dtype is not None:
            cos, sin = cos.to(dtype), sin.to(dtype)
        return cos, sin

    def extra_repr(self) -> str:
        return f"head_dim={self.head_dim}, base={self.base}, cached_positions={self.cached_positions}"
