# Purpose: Splits single-label dataset indices into stratified partitions preserving class proportions using a seeded pure-torch implementation (optional sklearn variant).
"""Stratified index splitting for single-label classification.

Algorithm (pure torch): for each class, shuffle its indices with a seeded ``torch.Generator`` and
allocate them to splits proportionally to ``fractions`` using largest-remainder rounding, so every
split gets ``floor(f_k * n_c)`` or one more sample of class ``c``. Ties in remainders go to the
earlier split. Results are shuffled within each split.

Class-count constraints: a class with fewer samples than the number of non-zero fractions cannot
appear in every split. With ``require_all_classes_in_each_split=True`` (default) this raises;
otherwise such classes are still allocated by rounding and some splits will miss them.

``stratified_split_sklearn`` is an optional two-way variant via scikit-learn (lazy import).

Requires: torch; scikit-learn only for ``stratified_split_sklearn``.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple, Union

import torch


def _allocate(n: int, fractions: Sequence[float]) -> List[int]:
    raw = [f * n for f in fractions]
    sizes = [int(math.floor(r)) for r in raw]
    remainder = n - sum(sizes)
    order = sorted(range(len(fractions)), key=lambda k: (-(raw[k] - sizes[k]), k))
    for k in order[:remainder]:
        sizes[k] += 1
    return sizes


def stratified_split_indices(
    labels: Union[Sequence[int], torch.Tensor],
    fractions: Sequence[float],
    seed: int,
    require_all_classes_in_each_split: bool = True,
) -> List[List[int]]:
    """Return one index list per fraction (e.g. ``(0.8, 0.1, 0.1)``), disjoint and exhaustive."""
    y = torch.as_tensor(labels, dtype=torch.long)
    if y.dim() != 1 or y.numel() == 0:
        raise ValueError("labels must be a non-empty 1D sequence of int class ids")
    fractions = [float(f) for f in fractions]
    if len(fractions) < 2 or any(f < 0 for f in fractions) or not math.isclose(sum(fractions), 1.0, abs_tol=1e-6):
        raise ValueError(f"fractions must be >= 2 non-negative values summing to 1, got {fractions}")

    generator = torch.Generator()
    generator.manual_seed(seed)
    by_class: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(y.tolist()):
        by_class[label].append(idx)

    needed = sum(1 for f in fractions if f > 0)
    splits: List[List[int]] = [[] for _ in fractions]
    for label in sorted(by_class):
        indices = by_class[label]
        if require_all_classes_in_each_split and len(indices) < needed:
            raise ValueError(
                f"class {label} has {len(indices)} samples but {needed} non-empty splits need one each"
            )
        perm = torch.randperm(len(indices), generator=generator).tolist()
        shuffled = [indices[i] for i in perm]
        sizes = _allocate(len(shuffled), fractions)
        if require_all_classes_in_each_split:
            for k, f in enumerate(fractions):
                if f > 0 and sizes[k] == 0:
                    donor = max(range(len(sizes)), key=lambda j: sizes[j])
                    sizes[donor] -= 1
                    sizes[k] += 1
        start = 0
        for k, size in enumerate(sizes):
            splits[k].extend(shuffled[start : start + size])
            start += size

    result = []
    for split in splits:
        order = torch.randperm(len(split), generator=generator).tolist()
        result.append([split[i] for i in order])
    return result


def stratified_split_sklearn(
    labels: Sequence[int], test_fraction: float, seed: int
) -> Tuple[List[int], List[int]]:
    """Two-way stratified split with ``sklearn.model_selection.train_test_split``."""
    try:
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise ImportError("Install scikit-learn for this variant: pip install scikit-learn") from exc
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    indices = list(range(len(labels)))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_fraction, stratify=list(labels), random_state=seed
    )
    return list(train_idx), list(test_idx)
