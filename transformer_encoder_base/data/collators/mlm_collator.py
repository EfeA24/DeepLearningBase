# Purpose: Collates tokenized texts for masked-language-model pre-training: pads the batch, then applies injected MLM corruption so labels exist only at selected positions.
"""MLM collator.

Input features: list of dicts with "input_ids" (list[int] / 1-D tensor, special tokens included),
optional "special_tokens_mask" (same length; 1 = special) and optional "token_type_ids".
Output:
    input_ids (B, T) long - corrupted ids (padded with ``pad_id``);
    attention_mask (B, T) long (1 = valid token, 0 = pad);
    labels (B, T) long - original ids at selected positions, -100 elsewhere (incl. padding/specials);
    token_type_ids (B, T) long when present.

Masking is pluggable: ``masking_fn(input_ids, attention_mask, special_tokens_mask) -> (corrupted, labels)``.
When not given, the default uses ``mask_tokens`` from data/masking/mlm_mask.py (imported lazily in
__init__) with the explicit probabilities, ``special_token_ids`` and ``generator`` supplied here.

Requires: torch; data/masking/mlm_mask.py when ``masking_fn`` is not injected (template root on sys.path).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

__all__ = ["MLMCollator"]

MaskingFn = Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor, torch.Tensor]]


def _pad(rows: Sequence[List[int]], value: int, length: int) -> torch.Tensor:
    out = torch.full((len(rows), length), value, dtype=torch.long)
    for i, row in enumerate(rows):
        if row:
            out[i, : len(row)] = torch.as_tensor(row, dtype=torch.long)
    return out


class MLMCollator:
    """Right-pads then corrupts a batch for MLM."""

    def __init__(
        self,
        pad_id: int,
        mask_token_id: Optional[int] = None,
        vocab_size: Optional[int] = None,
        special_token_ids: Optional[Sequence[int]] = None,
        mlm_probability: float = 0.15,
        mask_replace_prob: float = 0.8,
        random_replace_prob: float = 0.1,
        generator: Optional[torch.Generator] = None,
        masking_fn: Optional[MaskingFn] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> None:
        self.pad_id = pad_id
        self.pad_to_multiple_of = pad_to_multiple_of
        if masking_fn is not None:
            self.masking_fn = masking_fn
            return
        if mask_token_id is None or vocab_size is None:
            raise ValueError("mask_token_id and vocab_size are required when masking_fn is not injected")
        from data.masking.mlm_mask import MLMMaskingConfig, mask_tokens

        config = MLMMaskingConfig(
            mlm_probability=mlm_probability,
            mask_replace_prob=mask_replace_prob,
            random_replace_prob=random_replace_prob,
        )
        excluded = list(special_token_ids or [])

        def default_masking(
            input_ids: torch.Tensor, attention_mask: torch.Tensor, special_tokens_mask: Optional[torch.Tensor]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            return mask_tokens(
                input_ids,
                mask_token_id=mask_token_id,
                vocab_size=vocab_size,
                config=config,
                attention_mask=attention_mask,
                special_tokens_mask=special_tokens_mask,
                special_token_ids=excluded,
                generator=generator,
            )

        self.masking_fn = default_masking

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
        if not features:
            raise ValueError("cannot collate an empty batch")

        def as_list(values: Any) -> List[int]:
            return values.tolist() if isinstance(values, torch.Tensor) else list(values)

        ids = [as_list(f["input_ids"]) for f in features]
        length = max(len(s) for s in ids)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            length = ((length + m - 1) // m) * m
        input_ids = _pad(ids, self.pad_id, length)
        attention_mask = _pad([[1] * len(s) for s in ids], 0, length)
        special_tokens_mask = None
        if all("special_tokens_mask" in f for f in features):
            special_tokens_mask = _pad([as_list(f["special_tokens_mask"]) for f in features], 1, length)
        corrupted, labels = self.masking_fn(input_ids, attention_mask, special_tokens_mask)
        batch = {"input_ids": corrupted, "attention_mask": attention_mask, "labels": labels}
        if all("token_type_ids" in f for f in features):
            batch["token_type_ids"] = _pad([as_list(f["token_type_ids"]) for f in features], 0, length)
        return batch
