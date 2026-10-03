# Purpose: Splits dataset indices randomly into explicit sizes or fractions with a seeded torch.Generator, without copying data.
"""Random index splitting.

``random_split_indices(num_samples, lengths, seed)``:
    lengths: absolute ints summing to ``num_samples``, or fractions summing to 1.0. Fractions use
        ``floor(f * N)`` then distribute the remainder one-by-one round-robin from the first split
        (same rule as ``torch.utils.data.random_split``).
    Returns a list of index lists (disjoint, covering all indices).
Wrap with ``torch.utils.data.Subset(dataset, indices)`` (``make_subsets``) - no data copying.

For class-imbalanced data prefer stratified_split.py. Split BEFORE fitting any statistics
(normalization, class weights) so val/test do not leak into training choices.

Requires: torch.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Union

import torch
from torch.utils.data import Dataset, Subset


def resolve_lengths(num_samples: int, lengths: Sequence[Union[int, float]]) -> List[int]:
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")
    if len(lengths) == 0:
        raise ValueError("lengths must not be empty")
    if all(isinstance(x, int) for x in lengths):
        if any(x < 0 for x in lengths) or sum(lengths) != num_samples:
            raise ValueError(f"integer lengths must be >= 0 and sum to {num_samples}, got {list(lengths)}")
        return list(lengths)
    fractions = [float(x) for x in lengths]
    if any(f < 0 for f in fractions) or not math.isclose(sum(fractions), 1.0, abs_tol=1e-6):
        raise ValueError(f"fractional lengths must be >= 0 and sum to 1, got {fractions}")
    sizes = [int(math.floor(f * num_samples)) for f in fractions]
    remainder = num_samples - sum(sizes)
    for i in range(remainder):
        sizes[i % len(sizes)] += 1
    return sizes


def random_split_indices(
    num_samples: int,
    lengths: Sequence[Union[int, float]],
    seed: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> List[List[int]]:
    """Return disjoint shuffled index lists; pass exactly one of ``seed`` or ``generator``."""
    if (seed is None) == (generator is None):
        raise ValueError("pass exactly one of seed or generator for an explicit, reproducible split")
    if generator is None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    sizes = resolve_lengths(num_samples, lengths)
    perm = torch.randperm(num_samples, generator=generator).tolist()
    splits, start = [], 0
    for size in sizes:
        splits.append(perm[start : start + size])
        start += size
    return splits


def make_subsets(dataset: Dataset, index_lists: Sequence[Sequence[int]]) -> List[Subset]:
    """Wrap each index list in a ``Subset`` view of ``dataset``."""
    return [Subset(dataset, list(indices)) for indices in index_lists]
