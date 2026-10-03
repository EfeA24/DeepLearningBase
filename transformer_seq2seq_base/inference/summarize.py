# Purpose: Provides a summarization facade with explicit source-length truncation and summary-length decoding limits over a supplied seq2seq model and decoding callable.
"""Summarization facade.

    Summarizer(model, encode_batch, decode_fn, decoding_fn, device=..., eos/pad ids,
               max_source_length=..., max_summary_tokens=...)
    .summarize(documents, batch_size=None) -> list[str] in input order

Source limit: encoded right-padded batches longer than ``max_source_length`` are cut from
the right (document start kept); rows that were cut get ``eos_token_id`` written at their
last kept position when ``restore_eos=True``. Target limit: ``max_summary_tokens`` is
passed to the decoding callable as ``max_new_tokens``. ``decoding_fn`` follows the
inference/generate.py contract; length penalties etc. are configured inside it (e.g. a
beam-search partial with ``length_penalty``).

Requires: torch; sibling inference/generate.py (imported lazily inside ``summarize``).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Dict, List, Optional, Union

import torch


def truncate_source_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    max_source_length: int,
    eos_token_id: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Cut a right-padded (B, S) batch to ``max_source_length`` columns.

    If ``eos_token_id`` is given, rows whose real length exceeded the limit end with EOS.
    """
    if input_ids.shape != attention_mask.shape or input_ids.dim() != 2:
        raise ValueError("input_ids and attention_mask must both be (B, S)")
    if max_source_length < 1:
        raise ValueError("max_source_length must be >= 1")
    if input_ids.size(1) <= max_source_length:
        return {"input_ids": input_ids, "attention_mask": attention_mask}
    lengths = attention_mask.to(torch.long).sum(dim=1)
    ids = input_ids[:, :max_source_length].clone()
    mask = attention_mask[:, :max_source_length].clone()
    if eos_token_id is not None:
        overflow = lengths > max_source_length
        ids[overflow, max_source_length - 1] = eos_token_id
    return {"input_ids": ids, "attention_mask": mask}


class Summarizer:
    """Holds explicit encode/decode/decoding pieces and length limits for summarization."""

    def __init__(
        self,
        model: Any,
        *,
        encode_batch: Callable[[List[str]], Mapping[str, torch.Tensor]],
        decode_fn: Callable[[List[int]], str],
        decoding_fn: Callable[..., Any],
        device: Union[str, torch.device],
        eos_token_id: int,
        pad_token_id: int,
        max_source_length: int,
        max_summary_tokens: int,
        restore_eos: bool = True,
        source_prefix: str = "",
        num_prefix_tokens: int = 1,
    ) -> None:
        if max_source_length < 1 or max_summary_tokens < 1:
            raise ValueError("max_source_length and max_summary_tokens must be >= 1")
        self.model = model
        self.encode_batch = encode_batch
        self.decode_fn = decode_fn
        self.decoding_fn = decoding_fn
        self.device = device
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.max_source_length = max_source_length
        self.max_summary_tokens = max_summary_tokens
        self.restore_eos = restore_eos
        self.source_prefix = source_prefix
        self.num_prefix_tokens = num_prefix_tokens

    def _encode(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        encoded = self.encode_batch(texts)
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        return truncate_source_batch(
            input_ids, attention_mask, max_source_length=self.max_source_length,
            eos_token_id=self.eos_token_id if self.restore_eos else None,
        )

    def summarize(self, documents: Sequence[str], *, batch_size: Optional[int] = None) -> List[str]:
        from inference.generate import generate

        if isinstance(documents, str):
            raise TypeError("documents must be a sequence of strings, not a single string")
        texts = [self.source_prefix + d for d in documents]
        if not texts:
            return []
        step = batch_size or len(texts)
        if step < 1:
            raise ValueError("batch_size must be >= 1")
        outputs: List[str] = []
        for start in range(0, len(texts), step):
            result = generate(
                self.model,
                self.decoding_fn,
                device=self.device,
                max_new_tokens=self.max_summary_tokens,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.pad_token_id,
                texts=texts[start : start + step],
                encode_batch=self._encode,
                decode_fn=self.decode_fn,
                num_prefix_tokens=self.num_prefix_tokens,
            )
            outputs.extend(result.texts or [])
        return outputs
