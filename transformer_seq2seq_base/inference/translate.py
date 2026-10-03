# Purpose: Provides a translation facade over a supplied seq2seq model, source encoder/target decoder pair and decoding strategy callable.
"""Translation facade.

    Translator(model, encode_batch, decode_fn, decoding_fn, device=..., eos/pad ids)
    .translate(texts, max_new_tokens=..., batch_size=None) -> list[str] in input order

``encode_batch(texts) -> {"input_ids", "attention_mask"}`` encodes SOURCE language text
(e.g. ``SourceTokenizer.batch_encode``), ``decode_fn(ids) -> str`` decodes TARGET ids
(e.g. ``TargetTokenizer.decode``), ``decoding_fn`` follows the inference/generate.py
contract (greedy or beam search partial). An optional ``source_prefix`` (e.g.
"translate English to German: ") is prepended for text-to-text models trained that way.
The model is not moved, loaded or switched to eval mode here.

Requires: torch; sibling inference/generate.py (imported lazily inside ``translate`` with
the template root on sys.path).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, List, Optional, Union

import torch


class Translator:
    """Holds the explicit pieces needed to translate batches of text."""

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
        source_prefix: str = "",
        num_prefix_tokens: int = 1,
    ) -> None:
        for name, fn in (("encode_batch", encode_batch), ("decode_fn", decode_fn), ("decoding_fn", decoding_fn)):
            if not callable(fn):
                raise TypeError(f"{name} must be callable")
        self.model = model
        self.encode_batch = encode_batch
        self.decode_fn = decode_fn
        self.decoding_fn = decoding_fn
        self.device = device
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.source_prefix = source_prefix
        self.num_prefix_tokens = num_prefix_tokens

    def translate(
        self,
        texts: Sequence[str],
        *,
        max_new_tokens: int,
        batch_size: Optional[int] = None,
    ) -> List[str]:
        """Translate ``texts``; ``batch_size=None`` processes everything in one batch."""
        from inference.generate import generate

        if isinstance(texts, str):
            raise TypeError("texts must be a sequence of strings, not a single string")
        texts = [self.source_prefix + t for t in texts]
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
                max_new_tokens=max_new_tokens,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.pad_token_id,
                texts=texts[start : start + step],
                encode_batch=self.encode_batch,
                decode_fn=self.decode_fn,
                num_prefix_tokens=self.num_prefix_tokens,
            )
            outputs.extend(result.texts or [])
        return outputs
