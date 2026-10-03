# Purpose: Selects masked-language-modeling positions and applies BERT 80/10/10 corruption with explicit probabilities and torch.Generator RNG.
"""MLM corruption for encoder pre-training.

Contract:
    input_ids: (B, T) long, already padded.
    attention_mask (optional): (B, T) bool or 0/1, True/1 = valid token. Padding is never selected.
    special_tokens_mask (optional): (B, T) bool or 0/1, True/1 = special token ([CLS], [SEP], ...),
        never selected. ``special_token_ids`` is an alternative way to exclude ids.
    Returns (corrupted_input_ids, labels):
        corrupted_input_ids: (B, T) long, a new tensor (input is not modified in place).
        labels: (B, T) long, original token id at selected positions, ``ignore_index`` (-100) elsewhere.

Of the selected positions: ``mask_replace_prob`` -> [MASK], ``random_replace_prob`` -> random token,
the remainder keep the original token (still predicted). Defaults 0.8 / 0.1 / 0.1 (BERT).
Random tokens are drawn uniformly from ``random_token_ids`` if supplied, else from [0, vocab_size).

RNG: pass ``generator`` (torch.Generator on the same device as ``input_ids``) for reproducibility.

Requires: torch only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

__all__ = ["MLMMaskingConfig", "mask_tokens"]


@dataclass(frozen=True)
class MLMMaskingConfig:
    """Explicit MLM masking hyperparameters."""

    mlm_probability: float = 0.15
    mask_replace_prob: float = 0.8
    random_replace_prob: float = 0.1
    ignore_index: int = -100

    def __post_init__(self) -> None:
        if not 0.0 < self.mlm_probability < 1.0:
            raise ValueError(f"mlm_probability must be in (0, 1), got {self.mlm_probability}")
        if self.mask_replace_prob < 0 or self.random_replace_prob < 0:
            raise ValueError("replacement probabilities must be non-negative")
        if self.mask_replace_prob + self.random_replace_prob > 1.0 + 1e-9:
            raise ValueError("mask_replace_prob + random_replace_prob must be <= 1")


def _bool(mask: torch.Tensor, name: str) -> torch.Tensor:
    if mask.dtype == torch.bool:
        return mask
    if mask.dtype.is_floating_point:
        raise TypeError(f"{name} must be bool or integer 0/1, got {mask.dtype}")
    return mask != 0


def mask_tokens(
    input_ids: torch.Tensor,
    mask_token_id: int,
    vocab_size: int,
    config: MLMMaskingConfig = MLMMaskingConfig(),
    attention_mask: Optional[torch.Tensor] = None,
    special_tokens_mask: Optional[torch.Tensor] = None,
    special_token_ids: Optional[Sequence[int]] = None,
    random_token_ids: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply MLM corruption; see the module docstring for the full contract."""
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must be (B, T), got shape {tuple(input_ids.shape)}")
    if input_ids.dtype.is_floating_point or input_ids.dtype == torch.bool:
        raise TypeError(f"input_ids must be an integer tensor, got {input_ids.dtype}")
    if not 0 <= mask_token_id < vocab_size:
        raise ValueError(f"mask_token_id={mask_token_id} outside [0, {vocab_size})")

    device = input_ids.device
    candidates = torch.ones_like(input_ids, dtype=torch.bool)
    if attention_mask is not None:
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")
        candidates &= _bool(attention_mask, "attention_mask")
    if special_tokens_mask is not None:
        if special_tokens_mask.shape != input_ids.shape:
            raise ValueError("special_tokens_mask must have the same shape as input_ids")
        candidates &= ~_bool(special_tokens_mask, "special_tokens_mask")
    if special_token_ids:
        ids = torch.as_tensor(list(special_token_ids), dtype=input_ids.dtype, device=device)
        candidates &= ~torch.isin(input_ids, ids)

    def uniform() -> torch.Tensor:
        return torch.rand(input_ids.shape, generator=generator, device=device)

    selected = candidates & (uniform() < config.mlm_probability)
    labels = torch.where(selected, input_ids, torch.full_like(input_ids, config.ignore_index))

    replace_with_mask = selected & (uniform() < config.mask_replace_prob)
    remaining = 1.0 - config.mask_replace_prob
    random_conditional = config.random_replace_prob / remaining if remaining > 0 else 0.0
    replace_with_random = selected & ~replace_with_mask & (uniform() < random_conditional)

    corrupted = input_ids.clone()
    corrupted[replace_with_mask] = mask_token_id

    num_random = int(replace_with_random.sum())
    if num_random > 0:
        if random_token_ids is not None:
            pool = random_token_ids.to(device=device, dtype=input_ids.dtype).flatten()
            if pool.numel() == 0:
                raise ValueError("random_token_ids must not be empty")
            picks = torch.randint(pool.numel(), (num_random,), generator=generator, device=device)
            random_tokens = pool[picks]
        else:
            random_tokens = torch.randint(
                vocab_size, (num_random,), generator=generator, device=device, dtype=input_ids.dtype
            )
        corrupted[replace_with_random] = random_tokens
    return corrupted, labels
