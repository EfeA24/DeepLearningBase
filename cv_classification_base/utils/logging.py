# Purpose: Provides a small named-logger adapter for image-classification runs with explicit, opt-in handler setup and metric formatting.
"""Task logging adapter.

Stdlib ``logging`` is imported inside functions only, and nothing is configured at import: no
``basicConfig``, no handlers, no levels. ``get_logger`` returns a named logger; handlers are added
only when the caller calls ``add_console_handler`` / ``add_file_handler`` explicitly.

Note: this file is named ``logging.py``. Import it as ``utils.logging`` from the template root;
do NOT put the ``utils/`` folder itself on ``sys.path`` (it would shadow the stdlib module).

Requires: Python standard library only.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def get_logger(name: str = "cv_classification", level: Optional[int] = None) -> Any:
    """Return ``logging.getLogger(name)``; sets the level only if ``level`` is given."""
    import logging

    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger


def add_console_handler(
    logger: Any, level: Optional[int] = None, fmt: str = DEFAULT_FORMAT, stream: Optional[Any] = None
) -> Any:
    """Attach a StreamHandler (default stream: sys.stderr) and return it."""
    import logging

    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(fmt))
    if level is not None:
        handler.setLevel(level)
    logger.addHandler(handler)
    return handler


def add_file_handler(
    logger: Any, path: str, level: Optional[int] = None, fmt: str = DEFAULT_FORMAT, mode: str = "a"
) -> Any:
    """Attach a FileHandler writing to the caller-supplied ``path`` and return it."""
    import logging

    handler = logging.FileHandler(path, mode=mode, encoding="utf-8")
    handler.setFormatter(logging.Formatter(fmt))
    if level is not None:
        handler.setLevel(level)
    logger.addHandler(handler)
    return handler


def format_metrics(metrics: Mapping[str, Any], precision: int = 4) -> str:
    """Format ``{"loss": 0.1234, ...}`` as ``"loss=0.1234 ..."`` (floats rounded)."""
    parts = []
    for key, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.{precision}f}")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def log_epoch(logger: Any, epoch: int, metrics: Mapping[str, Any], level: Optional[int] = None) -> None:
    """Log one epoch summary line at INFO (or the given level)."""
    import logging

    logger.log(level if level is not None else logging.INFO, "epoch %d | %s", epoch, format_metrics(metrics))
