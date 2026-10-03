# Purpose: Collates fixed-length sequence examples (dicts or tuples) into batch-first tensors while preserving every field.
"""Fixed-length sequence collator.

Examples must all be dicts with the same keys, or all tuples/lists of the same length.
Per field:
- tensors with identical shapes -> ``torch.stack`` -> (B, *shape) (batch-first);
  mismatched shapes raise ValueError (use padded_collator.py for variable lengths);
- Python numbers/bools, or equal-length lists of numbers -> ``torch.tensor``;
- anything else (strings, ids, metadata) is kept as a Python list of length B.
For dict examples, ``stack_keys`` optionally restricts which keys are tensorized; other
keys are kept as lists.

Requires: torch only. No sibling files.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence, Union

import torch


def _is_number(value: Any) -> bool:
    return isinstance(value, (bool, int, float))


def collate_field(values: List[Any], name: str = "field") -> Any:
    """Collate one field across a batch according to the module rules."""
    if all(isinstance(v, torch.Tensor) for v in values):
        shapes = {tuple(v.shape) for v in values}
        if len(shapes) != 1:
            raise ValueError(
                f"{name}: tensors have different shapes {sorted(shapes)}; use a padding collator."
            )
        return torch.stack(values, dim=0)
    if all(_is_number(v) for v in values):
        return torch.tensor(values)
    if all(isinstance(v, (list, tuple)) and all(_is_number(x) for x in v) for v in values):
        lengths = {len(v) for v in values}
        if len(lengths) != 1:
            raise ValueError(f"{name}: lists have different lengths {sorted(lengths)}; use a padding collator.")
        return torch.tensor([list(v) for v in values])
    return list(values)


class SequenceCollator:
    """Callable collate_fn for fixed-length examples; see module docstring."""

    def __init__(self, stack_keys: Optional[Sequence[str]] = None) -> None:
        self.stack_keys = None if stack_keys is None else set(stack_keys)

    def __call__(self, batch: Sequence[Any]) -> Union[Dict[str, Any], tuple, Any]:
        if len(batch) == 0:
            raise ValueError("Cannot collate an empty batch.")
        first = batch[0]
        if isinstance(first, Mapping):
            keys = list(first.keys())
            for index, example in enumerate(batch):
                if not isinstance(example, Mapping) or set(example.keys()) != set(keys):
                    raise ValueError(f"Example {index} does not have the same keys as example 0: {keys}.")
            result: Dict[str, Any] = {}
            for key in keys:
                values = [example[key] for example in batch]
                if self.stack_keys is not None and key not in self.stack_keys:
                    result[key] = values
                else:
                    result[key] = collate_field(values, name=str(key))
            return result
        if isinstance(first, (tuple, list)) and not all(_is_number(x) for x in first):
            width = len(first)
            for index, example in enumerate(batch):
                if not isinstance(example, (tuple, list)) or len(example) != width:
                    raise ValueError(f"Example {index} does not have {width} fields like example 0.")
            return tuple(collate_field([example[i] for example in batch], name=f"field {i}") for i in range(width))
        return collate_field(list(batch))


def collate_fixed_length(batch: Sequence[Any]) -> Any:
    """Functional form of ``SequenceCollator()(batch)``."""
    return SequenceCollator()(batch)
