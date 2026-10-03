# Purpose: Collates variable-length seq2seq examples by padding source and target independently into a fixed encoder-decoder batch key contract.
"""Seq2seq collator.

Input: a list of examples (mappings) each holding
    ``source_key`` (default "input_ids"): 1-D list/tensor of source ids (EOS already added).
    ``target_key`` (default "labels"):    1-D list/tensor of target ids ending with EOS,
                                          WITHOUT the decoder start token. Optional when
                                          ``require_targets=False`` (inference batches).

Output dict (right padding, int64 unless noted):
    input_ids              (B, S)  pad = pad_token_id
    attention_mask         (B, S)  1 = real source token, 0 = pad
    decoder_input_ids      (B, T)  shift_right(labels): [start, y1, ...]; -100 -> pad_token_id
                                   (the first pad slot holds the shifted EOS; its label is -100)
    decoder_attention_mask (B, T)  1 = real decoder position, 0 = pad
    labels                 (B, T)  pad = label_pad_token_id (-100)
    source_lengths         (B,)
    target_lengths         (B,)
Target keys are omitted when no example has a target and ``require_targets=False``.
Fields listed in ``keep_fields`` are returned as Python lists in input order.

Requires: torch only.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional

import torch


def _to_id_list(value: Any, name: str) -> List[int]:
    if isinstance(value, torch.Tensor):
        if value.dim() != 1:
            raise ValueError(f"{name} tensor must be 1-D, got shape {tuple(value.shape)}")
        return [int(v) for v in value.tolist()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [int(v) for v in value]
    raise TypeError(f"{name} must be a 1-D tensor or a sequence of ints, got {type(value).__name__}")


def _round_up(length: int, multiple: Optional[int]) -> int:
    if not multiple:
        return length
    return ((length + multiple - 1) // multiple) * multiple


class Seq2SeqCollator:
    """Pads source and target sequences independently and builds decoder inputs.

    Args:
        pad_token_id: id used to pad ``input_ids`` and ``decoder_input_ids``.
        decoder_start_token_id: first decoder input token.
        label_pad_token_id: label padding value ignored by the loss.
        source_key / target_key: example field names.
        pad_to_multiple_of: optionally round padded lengths up (e.g. 8 for tensor cores).
        require_targets: raise if an example lacks ``target_key``.
        keep_fields: extra example fields copied into the batch as lists.
    """

    def __init__(
        self,
        *,
        pad_token_id: int,
        decoder_start_token_id: int,
        label_pad_token_id: int = -100,
        source_key: str = "input_ids",
        target_key: str = "labels",
        pad_to_multiple_of: Optional[int] = None,
        require_targets: bool = True,
        keep_fields: Sequence[str] = (),
    ) -> None:
        if pad_to_multiple_of is not None and pad_to_multiple_of < 1:
            raise ValueError("pad_to_multiple_of must be a positive integer or None")
        self.pad_token_id = int(pad_token_id)
        self.decoder_start_token_id = int(decoder_start_token_id)
        self.label_pad_token_id = int(label_pad_token_id)
        self.source_key = source_key
        self.target_key = target_key
        self.pad_to_multiple_of = pad_to_multiple_of
        self.require_targets = require_targets
        self.keep_fields = tuple(keep_fields)

    def _pad(self, sequences: List[List[int]], pad_value: int) -> torch.Tensor:
        max_len = _round_up(max(len(s) for s in sequences), self.pad_to_multiple_of)
        out = torch.full((len(sequences), max_len), pad_value, dtype=torch.long)
        for row, seq in enumerate(sequences):
            if seq:
                out[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        return out

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        if len(examples) == 0:
            raise ValueError("cannot collate an empty list of examples")
        sources: List[List[int]] = []
        targets: List[Optional[List[int]]] = []
        for i, example in enumerate(examples):
            if not isinstance(example, Mapping):
                raise TypeError(f"example {i} must be a mapping, got {type(example).__name__}")
            if self.source_key not in example:
                raise KeyError(f"example {i} is missing source field {self.source_key!r}")
            src = _to_id_list(example[self.source_key], self.source_key)
            if not src:
                raise ValueError(f"example {i} has an empty source sequence")
            sources.append(src)
            tgt = example.get(self.target_key)
            if tgt is None:
                if self.require_targets:
                    raise KeyError(f"example {i} is missing target field {self.target_key!r}")
                targets.append(None)
            else:
                tgt_ids = _to_id_list(tgt, self.target_key)
                if not tgt_ids:
                    raise ValueError(f"example {i} has an empty target sequence")
                targets.append(tgt_ids)

        input_ids = self._pad(sources, self.pad_token_id)
        source_lengths = torch.tensor([len(s) for s in sources], dtype=torch.long)
        attention_mask = (
            torch.arange(input_ids.size(1)).unsqueeze(0) < source_lengths.unsqueeze(1)
        ).long()
        batch: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "source_lengths": source_lengths,
        }

        present = [t is not None for t in targets]
        if any(present):
            if not all(present):
                raise ValueError("either all or none of the examples must provide targets")
            target_lists = [t for t in targets if t is not None]
            labels = self._pad(target_lists, self.label_pad_token_id)
            target_lengths = torch.tensor([len(t) for t in target_lists], dtype=torch.long)
            decoder_input_ids = labels.new_empty(labels.shape)
            decoder_input_ids[:, 1:] = labels[:, :-1]
            decoder_input_ids[:, 0] = self.decoder_start_token_id
            decoder_input_ids = decoder_input_ids.masked_fill(
                decoder_input_ids == self.label_pad_token_id, self.pad_token_id
            )
            decoder_attention_mask = (
                torch.arange(labels.size(1)).unsqueeze(0) < target_lengths.unsqueeze(1)
            ).long()
            batch.update(
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
                target_lengths=target_lengths,
            )

        for field in self.keep_fields:
            batch[field] = [example.get(field) for example in examples]
        return batch
