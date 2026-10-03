# Purpose: Formats instruction/response records with an explicit prompt template into input_ids plus a loss mask that excludes prompt tokens.
"""Instruction-tuning dataset (single turn).

Formatting (explicit template, ``str.format`` fields from the record):
    prompt   = template.prompt_with_input.format(**fields)   if the input field is non-empty
             = template.prompt_no_input.format(**fields)     otherwise
    response = template.response.format(**fields)

Tokenization: prompt and response are tokenized SEPARATELY and concatenated:
    input_ids = [bos] + tokenize(prompt) + tokenize(response) + [eos]
    loss_mask =  0    +   0 ... 0        +   1 ... 1          +  1
So only response tokens (and EOS, which teaches stopping) are trained. Separate tokenization
can differ from tokenizing the joined string at the boundary for some BPE vocabularies; end
the prompt template with a newline/space convention your tokenizer handles consistently.

Item contract (aligned labels; loss compares logits[:, :-1] with labels[:, 1:]):
    input_ids LongTensor (T,), loss_mask LongTensor (T,) 0/1,
    labels LongTensor (T,) = input_ids where loss_mask == 1 else -100.

Overflow (> max_length): "truncate_response" keeps the whole prompt and cuts the response
from the right (raises if no response token survives); "error" raises.

Requires: torch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class InstructionTemplate:
    """Prompt/response format strings; field names refer to record keys."""

    prompt_with_input: str = (
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    )
    prompt_no_input: str = "### Instruction:\n{instruction}\n\n### Response:\n"
    response: str = "{response}"


class InstructionDataset(Dataset):
    """Instruction/response records -> token ids + response-only loss mask.

    Args:
        records: Sequence of mappings containing the template fields.
        tokenize: Callable ``str -> list[int]`` WITHOUT automatic special tokens.
        template: :class:`InstructionTemplate`.
        input_field: Record key whose emptiness selects ``prompt_no_input`` (None -> always
            ``prompt_with_input``).
        bos_id, eos_id: Optional special ids.
        max_length: Optional maximum total length.
        overflow: "truncate_response" or "error".
        ignore_index: Label value for prompt tokens.
    """

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        tokenize: Callable[[str], List[int]],
        template: InstructionTemplate = InstructionTemplate(),
        input_field: Optional[str] = "input",
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        max_length: Optional[int] = None,
        overflow: str = "truncate_response",
        ignore_index: int = -100,
    ) -> None:
        if overflow not in ("truncate_response", "error"):
            raise ValueError("overflow must be 'truncate_response' or 'error'")
        if max_length is not None and max_length < 2:
            raise ValueError("max_length must be >= 2")
        self.records = records
        self.tokenize = tokenize
        self.template = template
        self.input_field = input_field
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.max_length = max_length
        self.overflow = overflow
        self.ignore_index = ignore_index

    def __len__(self) -> int:
        return len(self.records)

    def format_prompt(self, record: Mapping[str, Any]) -> str:
        has_input = self.input_field is None or bool(str(record.get(self.input_field, "") or "").strip())
        fmt = self.template.prompt_with_input if has_input else self.template.prompt_no_input
        try:
            return fmt.format(**record)
        except KeyError as exc:
            raise KeyError(f"Record is missing template field {exc}") from exc

    def format_response(self, record: Mapping[str, Any]) -> str:
        try:
            return self.template.response.format(**record)
        except KeyError as exc:
            raise KeyError(f"Record is missing template field {exc}") from exc

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]
        prompt_ids = ([self.bos_id] if self.bos_id is not None else []) + list(self.tokenize(self.format_prompt(record)))
        response_ids = list(self.tokenize(self.format_response(record)))
        if self.eos_id is not None:
            response_ids.append(self.eos_id)
        if self.max_length is not None and len(prompt_ids) + len(response_ids) > self.max_length:
            if self.overflow == "error":
                raise ValueError(f"Record {index} has {len(prompt_ids) + len(response_ids)} tokens > {self.max_length}")
            keep = self.max_length - len(prompt_ids)
            if keep <= 0:
                raise ValueError(f"Record {index}: prompt alone ({len(prompt_ids)} tokens) exceeds max_length")
            response_ids = response_ids[:keep]
        input_ids = torch.tensor(prompt_ids + response_ids, dtype=torch.long)
        loss_mask = torch.tensor([0] * len(prompt_ids) + [1] * len(response_ids), dtype=torch.long)
        labels = input_ids.masked_fill(loss_mask == 0, self.ignore_index)
        return {"input_ids": input_ids, "loss_mask": loss_mask, "labels": labels}
