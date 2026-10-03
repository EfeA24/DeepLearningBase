# Purpose: Provides per-layer key/value cache containers (concatenating and preallocated) for incremental causal decoding.
"""Per-layer KV caches for decoder self-attention.

Layout (all caches in this template):
    keys, values: (B, H_kv, T_cached, D)      sequence axis = dim 2
H_kv is the number of KEY/VALUE heads (``n_kv_heads`` for GQA), i.e. keys are cached
BEFORE ``repeat_kv``; keys are stored AFTER RoPE has been applied.

Cache-object protocol (duck-typed by the attention modules of this template):
    cache.seq_len                    -> int, tokens already cached (read BEFORE update)
    cache.update(k_new, v_new)       -> (k_all, v_all) including the new tokens
    cache.reset() / cache.trim(n) / cache.reorder(beam_idx)

Attention modules also accept a plain tuple ``(k_past, v_past)`` instead of an object;
then they return a NEW tuple (functional style). Objects are updated IN PLACE and
returned as the "present" value.

``KVCache``        grows by ``torch.cat`` (simple, reallocates every step).
``StaticKVCache``  preallocates ``max_seq_len`` slots and writes in place (no reallocation;
                   returns views ``[:, :, :seq_len]``).

Requires: torch.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

KVPair = Tuple[torch.Tensor, torch.Tensor]


def _validate_new_kv(k_new: torch.Tensor, v_new: torch.Tensor) -> None:
    if k_new.dim() != 4 or v_new.dim() != 4:
        raise ValueError(
            f"k/v must be (B, H_kv, T, D); got {tuple(k_new.shape)} and {tuple(v_new.shape)}"
        )
    if k_new.shape[:3] != v_new.shape[:3]:
        raise ValueError(f"k/v (B, H, T) mismatch: {tuple(k_new.shape)} vs {tuple(v_new.shape)}")


class KVCache:
    """Concatenating cache for one attention layer."""

    def __init__(self) -> None:
        self.keys: Optional[torch.Tensor] = None
        self.values: Optional[torch.Tensor] = None

    @property
    def seq_len(self) -> int:
        return 0 if self.keys is None else int(self.keys.shape[2])

    def update(self, k_new: torch.Tensor, v_new: torch.Tensor) -> KVPair:
        """Append new keys/values along dim 2 and return the full cached tensors."""
        _validate_new_kv(k_new, v_new)
        if self.keys is None:
            self.keys, self.values = k_new, v_new
        else:
            if k_new.shape[0] != self.keys.shape[0] or k_new.shape[1] != self.keys.shape[1]:
                raise ValueError(
                    f"New k {tuple(k_new.shape)} incompatible with cached {tuple(self.keys.shape)}"
                )
            self.keys = torch.cat((self.keys, k_new), dim=2)
            self.values = torch.cat((self.values, v_new), dim=2)
        return self.keys, self.values

    def get(self) -> Optional[KVPair]:
        return None if self.keys is None else (self.keys, self.values)

    def reset(self) -> None:
        self.keys = None
        self.values = None

    def trim(self, length: int) -> None:
        """Keep only the first ``length`` cached positions (e.g. after rejecting draft tokens)."""
        if length < 0 or length > self.seq_len:
            raise ValueError(f"trim length must be in [0, {self.seq_len}], got {length}")
        if self.keys is None:
            return
        if length == 0:
            self.reset()
            return
        self.keys = self.keys[:, :, :length]
        self.values = self.values[:, :, :length]

    def reorder(self, beam_idx: torch.Tensor) -> None:
        """Select/reorder batch rows (beam search): new_row[i] = old_row[beam_idx[i]]."""
        if self.keys is None:
            return
        beam_idx = beam_idx.to(self.keys.device)
        self.keys = self.keys.index_select(0, beam_idx)
        self.values = self.values.index_select(0, beam_idx)

    @classmethod
    def from_tuple(cls, kv: Optional[KVPair]) -> "KVCache":
        cache = cls()
        if kv is not None:
            cache.update(kv[0], kv[1])
        return cache


class StaticKVCache:
    """Preallocated cache for one attention layer.

    Args:
        batch_size, n_kv_heads, max_seq_len, head_dim: Storage shape (B, H_kv, max_seq_len, D).
        dtype, device: Explicit storage dtype/device (no implicit device selection).
    """

    def __init__(
        self,
        batch_size: int,
        n_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        for name, value in (
            ("batch_size", batch_size),
            ("n_kv_heads", n_kv_heads),
            ("max_seq_len", max_seq_len),
            ("head_dim", head_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        shape = (batch_size, n_kv_heads, max_seq_len, head_dim)
        self.keys = torch.zeros(shape, dtype=dtype, device=device)
        self.values = torch.zeros(shape, dtype=dtype, device=device)
        self.max_seq_len = max_seq_len
        self._length = 0

    @property
    def seq_len(self) -> int:
        return self._length

    def update(self, k_new: torch.Tensor, v_new: torch.Tensor) -> KVPair:
        """Write new keys/values at ``[seq_len, seq_len + T)`` and return views of the filled part."""
        _validate_new_kv(k_new, v_new)
        b, h, t, d = k_new.shape
        if (b, h, d) != (self.keys.shape[0], self.keys.shape[1], self.keys.shape[3]):
            raise ValueError(
                f"New k {tuple(k_new.shape)} incompatible with storage {tuple(self.keys.shape)}"
            )
        end = self._length + t
        if end > self.max_seq_len:
            raise ValueError(f"StaticKVCache overflow: {end} > max_seq_len={self.max_seq_len}")
        self.keys[:, :, self._length : end] = k_new.to(self.keys.dtype)
        self.values[:, :, self._length : end] = v_new.to(self.values.dtype)
        self._length = end
        return self.keys[:, :, :end], self.values[:, :, :end]

    def get(self) -> Optional[KVPair]:
        if self._length == 0:
            return None
        return self.keys[:, :, : self._length], self.values[:, :, : self._length]

    def reset(self) -> None:
        """Mark the cache empty (storage is reused; stale values are never read)."""
        self._length = 0

    def trim(self, length: int) -> None:
        if length < 0 or length > self._length:
            raise ValueError(f"trim length must be in [0, {self._length}], got {length}")
        self._length = length

    def reorder(self, beam_idx: torch.Tensor) -> None:
        """Reorder batch rows; ``beam_idx`` must have length B (storage batch size is fixed)."""
        if beam_idx.numel() != self.keys.shape[0]:
            raise ValueError(
                f"beam_idx must have {self.keys.shape[0]} entries for a static cache, got {beam_idx.numel()}"
            )
        beam_idx = beam_idx.to(self.keys.device)
        self.keys = self.keys.index_select(0, beam_idx)
        self.values = self.values.index_select(0, beam_idx)
