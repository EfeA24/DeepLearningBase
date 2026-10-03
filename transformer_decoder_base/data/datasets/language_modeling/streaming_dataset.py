# Purpose: Provides an IterableDataset scaffold for streaming causal-LM data from a caller-provided iterator factory with worker/rank sharding.
"""Streaming causal-LM dataset (no network/file access of its own).

The caller supplies ``iterator_factory() -> Iterable[example]`` (e.g. a generator reading local
files or a ``datasets`` streaming split the caller created). A NEW iterator is requested on every
``__iter__`` call (every epoch and every DataLoader worker).

Sharding: example number ``i`` of the stream is kept by the consumer whose global id is
``i % num_shards`` where ``num_shards = world_size * num_workers`` and
``global id = rank * num_workers + worker_id``. ``rank``/``world_size`` are explicit constructor
arguments (no torch.distributed queries). Each consumer still READS the whole stream and
skips others' examples; for large data shard at the source instead (build a factory that
closes over the shard assignment and reads only its own files) and set ``shard=False``.

Optional processing, applied in order to each kept example:
    tokenize(example) -> list[int]  then, if ``context_length`` is set, tokens are buffered
    (with optional ``separator_id`` between examples) and emitted as fixed windows
    {"input_ids": (L,), "labels": (L,)} (aligned labels). Without ``context_length`` each example
    is yielded as ``{"input_ids": LongTensor}`` (tokenized) or unchanged (no tokenize).
The incomplete tail buffer is dropped at the end of the stream. No shuffling is done
(add a shuffle buffer in your iterator if needed).

Requires: torch.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import IterableDataset, get_worker_info


class StreamingLMDataset(IterableDataset):
    """Iterable causal-LM dataset over a caller-provided stream.

    Args:
        iterator_factory: Zero-argument callable returning a fresh iterable of examples.
        tokenize: Optional ``example -> list[int]``.
        context_length: Optional window length; requires ``tokenize``.
        separator_id: Optional id appended after each tokenized example (e.g. EOS).
        shard: Shard examples across workers/ranks (see module docstring).
        rank, world_size: Explicit distributed position (defaults: single process).
    """

    def __init__(
        self,
        iterator_factory: Callable[[], Iterable[Any]],
        tokenize: Optional[Callable[[Any], List[int]]] = None,
        context_length: Optional[int] = None,
        separator_id: Optional[int] = None,
        shard: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        super().__init__()
        if not callable(iterator_factory):
            raise TypeError("iterator_factory must be callable")
        if context_length is not None and (context_length < 2 or tokenize is None):
            raise ValueError("context_length must be >= 2 and requires tokenize")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError(f"Invalid rank/world_size: {rank}/{world_size}")
        self.iterator_factory = iterator_factory
        self.tokenize = tokenize
        self.context_length = context_length
        self.separator_id = separator_id
        self.shard = shard
        self.rank = rank
        self.world_size = world_size

    def _shard_info(self) -> Tuple[int, int]:
        info = get_worker_info()
        num_workers = info.num_workers if info is not None else 1
        worker_id = info.id if info is not None else 0
        return self.rank * num_workers + worker_id, self.world_size * num_workers

    def _kept_examples(self) -> Iterator[Any]:
        consumer, num_shards = self._shard_info()
        for i, example in enumerate(self.iterator_factory()):
            if not self.shard or i % num_shards == consumer:
                yield example

    def __iter__(self) -> Iterator[Any]:
        if self.tokenize is None:
            yield from self._kept_examples()
            return
        if self.context_length is None:
            for example in self._kept_examples():
                yield {"input_ids": torch.tensor(self.tokenize(example), dtype=torch.long)}
            return
        buffer: List[int] = []
        length = self.context_length
        for example in self._kept_examples():
            buffer.extend(int(t) for t in self.tokenize(example))
            if self.separator_id is not None:
                buffer.append(self.separator_id)
            while len(buffer) >= length:
                window = torch.tensor(buffer[:length], dtype=torch.long)
                del buffer[:length]
                yield {"input_ids": window, "labels": window.clone()}
