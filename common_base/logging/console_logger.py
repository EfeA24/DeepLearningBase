# Purpose: Writes formatted training metrics to a caller-supplied text stream as an optional console logger for any PyTorch project.
"""Minimal console logger.

Writes one line per call to ``stream`` (default ``sys.stdout``, imported lazily at construction).
No handlers, global configuration or stdlib ``logging`` usage (this folder name would shadow the
stdlib module if ``common_base/`` itself were placed on ``sys.path``). Printing only happens on
explicit ``log``/``log_metrics`` calls. In distributed runs, construct with ``enabled=is_main_process()``.

Requires: torch is not needed (stdlib only).
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional, TextIO


class ConsoleLogger:
    """Formats ``key=value`` metric lines.

    Args:
        stream: Writable text stream; None -> ``sys.stdout``.
        prefix: Optional string prepended to every line (e.g. "[rank0]").
        precision: Decimal places for float values.
        enabled: When False, all calls are no-ops.
        show_time: Prepend a local ``HH:MM:SS`` timestamp.
    """

    def __init__(
        self,
        stream: Optional[TextIO] = None,
        prefix: str = "",
        precision: int = 4,
        enabled: bool = True,
        show_time: bool = False,
    ) -> None:
        if stream is None:
            import sys

            stream = sys.stdout
        if precision < 0:
            raise ValueError(f"precision must be >= 0, got {precision}")
        self.stream = stream
        self.prefix = prefix
        self.precision = precision
        self.enabled = enabled
        self.show_time = show_time

    def _format_value(self, value: Any) -> str:
        if hasattr(value, "item") and callable(value.item):
            try:
                value = value.item()
            except (RuntimeError, ValueError):
                pass
        if isinstance(value, float):
            return f"{value:.{self.precision}f}"
        return str(value)

    def _head(self) -> str:
        parts = []
        if self.show_time:
            import time

            parts.append(time.strftime("%H:%M:%S"))
        if self.prefix:
            parts.append(self.prefix)
        return " ".join(parts)

    def log(self, message: str) -> None:
        if not self.enabled:
            return
        head = self._head()
        self.stream.write(f"{head} {message}\n" if head else f"{message}\n")
        self.stream.flush()

    def log_metrics(self, metrics: Mapping[str, Any], step: Optional[int] = None, tag: str = "") -> None:
        """Write e.g. ``train step=100 loss=0.1234 acc=0.9000``."""
        fields = [tag] if tag else []
        if step is not None:
            fields.append(f"step={step}")
        fields.extend(f"{k}={self._format_value(v)}" for k, v in metrics.items())
        self.log(" ".join(fields))
