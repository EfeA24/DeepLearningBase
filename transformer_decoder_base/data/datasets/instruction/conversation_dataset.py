# Purpose: Formats multi-turn chat conversations with an explicit role template and marks only assistant tokens for the causal-LM loss.
"""Multi-turn conversation dataset.

Conversation format: a sequence of ``{"role": str, "content": str}`` messages.

Role template (explicit, no hidden chat format): for each role a ``RoleFormat(prefix, suffix)``;
a message is rendered as ``prefix + content + suffix``. The three pieces are tokenized
separately and concatenated:
    prefix tokens   -> loss 0 (role header such as "<|assistant|>\\n")
    content tokens  -> loss 1 if role in ``train_roles`` else 0
    suffix tokens   -> loss 1 if role in ``train_roles`` else 0 (so the model learns to emit
                       its end-of-turn marker)
Optional ``bos_id`` (loss 0) is prepended and ``eos_id`` appended (loss 1 only if the last
message is a trained role).

Item contract (aligned labels; loss compares logits[:, :-1] with labels[:, 1:]):
    input_ids LongTensor (T,), loss_mask LongTensor (T,) 0/1,
    labels = input_ids where loss_mask == 1 else -100.
Truncation (``max_length``) cuts from the right; a conversation with no trained token left
raises ValueError (it would contribute nothing and usually signals a data problem).

If your tokenizer offers ``apply_chat_template``, verify that the templates here reproduce it
exactly before mixing the two.

Requires: torch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class RoleFormat:
    prefix: str
    suffix: str


def _default_roles() -> Dict[str, RoleFormat]:
    return {
        "system": RoleFormat("<|system|>\n", "<|end|>\n"),
        "user": RoleFormat("<|user|>\n", "<|end|>\n"),
        "assistant": RoleFormat("<|assistant|>\n", "<|end|>\n"),
    }


@dataclass(frozen=True)
class ChatTemplate:
    """Per-role prefix/suffix strings. The defaults are illustrative, not any model's official format."""

    roles: Dict[str, RoleFormat] = field(default_factory=_default_roles)


class ConversationDataset(Dataset):
    """Conversations -> token ids + assistant-only loss mask.

    Args:
        conversations: Sequence of message lists.
        tokenize: Callable ``str -> list[int]`` without automatic special tokens.
        template: :class:`ChatTemplate`.
        train_roles: Roles whose content/suffix tokens are trained (default assistant only).
        bos_id, eos_id: Optional special ids.
        max_length: Optional right truncation length.
        ignore_index: Label value for untrained positions.
    """

    def __init__(
        self,
        conversations: Sequence[Sequence[Mapping[str, Any]]],
        tokenize: Callable[[str], List[int]],
        template: ChatTemplate = ChatTemplate(),
        train_roles: Tuple[str, ...] = ("assistant",),
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        max_length: Optional[int] = None,
        ignore_index: int = -100,
    ) -> None:
        unknown = set(train_roles) - set(template.roles)
        if unknown:
            raise ValueError(f"train_roles {sorted(unknown)} are not defined in the template")
        if max_length is not None and max_length < 2:
            raise ValueError("max_length must be >= 2")
        self.conversations = conversations
        self.tokenize = tokenize
        self.template = template
        self.train_roles = tuple(train_roles)
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.max_length = max_length
        self.ignore_index = ignore_index

    def __len__(self) -> int:
        return len(self.conversations)

    def encode_conversation(self, messages: Sequence[Mapping[str, Any]]) -> Tuple[List[int], List[int]]:
        """Return (token ids, 0/1 loss mask) before truncation."""
        if not messages:
            raise ValueError("Conversation is empty")
        ids: List[int] = []
        mask: List[int] = []
        if self.bos_id is not None:
            ids.append(self.bos_id)
            mask.append(0)
        trained = False
        for i, message in enumerate(messages):
            role = message.get("role")
            if role not in self.template.roles:
                raise ValueError(f"Message {i} has unknown role {role!r}")
            fmt = self.template.roles[role]
            trained = role in self.train_roles
            for text, flag in ((fmt.prefix, 0), (str(message.get("content", "")), int(trained)), (fmt.suffix, int(trained))):
                if text:
                    piece = [int(t) for t in self.tokenize(text)]
                    ids.extend(piece)
                    mask.extend([flag] * len(piece))
        if self.eos_id is not None:
            ids.append(self.eos_id)
            mask.append(int(trained))
        return ids, mask

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        ids, mask = self.encode_conversation(self.conversations[index])
        if self.max_length is not None:
            ids, mask = ids[: self.max_length], mask[: self.max_length]
        if not any(mask[1:]):
            raise ValueError(f"Conversation {index} has no trained tokens (after truncation)")
        input_ids = torch.tensor(ids, dtype=torch.long)
        loss_mask = torch.tensor(mask, dtype=torch.long)
        labels = input_ids.masked_fill(loss_mask == 0, self.ignore_index)
        return {"input_ids": input_ids, "loss_mask": loss_mask, "labels": labels}
