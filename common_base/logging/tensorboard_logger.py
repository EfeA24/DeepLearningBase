# Purpose: Adapts torch.utils.tensorboard SummaryWriter for logging scalars and histograms in PyTorch projects, with lazy import and explicit close.
"""TensorBoard adapter.

``SummaryWriter`` is imported inside the constructor, so importing this module does not require
the ``tensorboard`` package. The log directory is passed explicitly; nothing is created until
the logger is constructed. Use as a context manager or call ``close()``. In distributed runs
create it on the main process only.

Requires: torch, tensorboard (``pip install tensorboard``) at construction time.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

import torch


class TensorBoardLogger:
    """Thin wrapper around ``SummaryWriter``.

    Args:
        log_dir: Directory for event files (str or PathLike), chosen by the caller.
        flush_secs: Flush interval in seconds.
        **writer_kwargs: Forwarded to ``SummaryWriter``.
    """

    def __init__(self, log_dir: Any, flush_secs: int = 120, **writer_kwargs: Any) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise ImportError("Install tensorboard to use TensorBoardLogger: pip install tensorboard") from exc
        self.writer = SummaryWriter(log_dir=str(log_dir), flush_secs=flush_secs, **writer_kwargs)
        self._closed = False

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("TensorBoardLogger is closed")

    def log_scalar(self, tag: str, value: Any, step: int) -> None:
        self._check_open()
        if isinstance(value, torch.Tensor):
            value = value.detach().item()
        self.writer.add_scalar(tag, float(value), global_step=step)

    def log_scalars(self, metrics: Mapping[str, Any], step: int, prefix: str = "") -> None:
        """Log each metric as its own scalar ``{prefix}/{name}`` (or ``name`` without prefix)."""
        for name, value in metrics.items():
            self.log_scalar(f"{prefix}/{name}" if prefix else name, value, step)

    def log_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        self._check_open()
        self.writer.add_histogram(tag, values.detach().float().cpu(), global_step=step)

    def log_text(self, tag: str, text: str, step: Optional[int] = None) -> None:
        self._check_open()
        self.writer.add_text(tag, text, global_step=step)

    def flush(self) -> None:
        if not self._closed:
            self.writer.flush()

    def close(self) -> None:
        if not self._closed:
            self.writer.close()
            self._closed = True

    def __enter__(self) -> "TensorBoardLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
