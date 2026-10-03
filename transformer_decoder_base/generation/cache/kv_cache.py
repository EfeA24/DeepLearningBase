# Purpose: Manages the per-request KV-cache lifecycle (init, update, trim, beam reorder, reset) for decoder LM generation loops.
"""Generation-facing cache lifecycle.

Holds the model's per-layer caches between decoding steps; it never computes attention.
Two storage modes, matching what ``models/attention/*`` accept as ``past_kv``:

* Tuple mode (default): the model returns ``present`` as a list of ``(k, v)`` tuples
  (B, H_kv, T, D); call ``update(present)`` after each step and pass ``cache.past`` back.
* Object mode: give ``layer_cache_factory(layer_idx) -> cache`` (e.g. a lambda building
  ``StaticKVCache``/``KVCache`` from models/attention/kv_cache.py). ``start_request()`` creates
  one object per layer; the model updates them IN PLACE, and ``update(present)`` just checks
  that the same objects came back.

Operations delegate to the layer caches: ``trim`` slices tuples along dim 2 or calls
``cache.trim``; ``reorder`` index-selects dim 0 or calls ``cache.reorder`` (beam search);
``reset`` drops tuples or calls ``cache.reset``.

Typical loop:
    cache = GenerationCache(num_layers)
    cache.start_request()
    logits, present = model_fn(prompt_ids, cache.past, mask); cache.update(present)
    logits, present = model_fn(next_ids, cache.past, mask);   cache.update(present)

Requires: torch.
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional

import torch


class GenerationCache:
    """Per-request container for the per-layer KV caches.

    Args:
        num_layers: Number of decoder layers (length of the past list).
        layer_cache_factory: Optional ``layer_idx -> cache object`` for object mode.
    """

    def __init__(
        self, num_layers: int, layer_cache_factory: Optional[Callable[[int], Any]] = None
    ) -> None:
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.num_layers = num_layers
        self.layer_cache_factory = layer_cache_factory
        self._layers: Optional[List[Any]] = None

    @property
    def object_mode(self) -> bool:
        return self.layer_cache_factory is not None

    def start_request(self) -> None:
        """Begin a new request: clears tuples or builds fresh layer cache objects."""
        if self.object_mode:
            self._layers = [self.layer_cache_factory(i) for i in range(self.num_layers)]
        else:
            self._layers = None

    @property
    def past(self) -> Optional[List[Any]]:
        """Value to pass as ``past``/``past_key_values`` to the model (None when empty in tuple mode)."""
        return self._layers

    @property
    def seq_len(self) -> int:
        if not self._layers or self._layers[0] is None:
            return 0
        first = self._layers[0]
        return int(first.seq_len) if hasattr(first, "update") else int(first[0].shape[2])

    def update(self, present: Optional[List[Any]]) -> None:
        """Store the model's returned per-layer caches."""
        if present is None:
            raise ValueError("Model returned no cache; call it with use_cache=True")
        if len(present) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} layer caches, got {len(present)}")
        if self.object_mode and self._layers is not None:
            if any(p is not c for p, c in zip(present, self._layers)):
                raise ValueError("Object-mode cache: the model returned different cache objects")
        self._layers = list(present)

    def trim(self, length: int) -> None:
        """Keep only the first ``length`` cached positions in every layer."""
        if self._layers is None:
            if length != 0:
                raise ValueError("Cannot trim an empty cache to a positive length")
            return
        if not 0 <= length <= self.seq_len:
            raise ValueError(f"length must be in [0, {self.seq_len}], got {length}")
        new_layers: List[Any] = []
        for layer in self._layers:
            if hasattr(layer, "trim"):
                layer.trim(length)
                new_layers.append(layer)
            else:
                new_layers.append((layer[0][:, :, :length], layer[1][:, :, :length]))
        self._layers = new_layers

    def reorder(self, beam_idx: torch.Tensor) -> None:
        """Reorder batch rows: new row i = old row ``beam_idx[i]`` (beam search)."""
        if self._layers is None:
            return
        new_layers: List[Any] = []
        for layer in self._layers:
            if hasattr(layer, "reorder"):
                layer.reorder(beam_idx)
                new_layers.append(layer)
            else:
                idx = beam_idx.to(layer[0].device)
                new_layers.append((layer[0].index_select(0, idx), layer[1].index_select(0, idx)))
        self._layers = new_layers

    def reset(self) -> None:
        """Release tuple caches or reset cache objects (storage reused)."""
        if self._layers is not None and self.object_mode:
            for layer in self._layers:
                layer.reset()
        else:
            self._layers = None
