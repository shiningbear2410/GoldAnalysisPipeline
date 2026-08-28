"""Logging configuration.

Deliberately plain: a stderr handler and a format that carries enough to trace
one Run. Log records are written by the pipeline as
``run=<id> stage=<stage> status=<status>`` so a grep on a run id reconstructs
its history.

Nothing here ever logs a payload. Secrets (bot tokens, API keys) are read from
the environment by later rounds and must never reach a log line; the pipeline
logs identifiers and counts, not content.
"""

from __future__ import annotations

import logging
import os
import sys

DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def configure_logging(level: str | int | None = None) -> None:
    """Install a stderr handler once.

    Args:
        level: Log level name or number. Falls back to ``GOLDPIPELINE_LOG_LEVEL``
            and then to ``INFO``.
    """
    resolved = level or os.environ.get("GOLDPIPELINE_LOG_LEVEL", "INFO")
    root = logging.getLogger("goldpipeline")
    root.setLevel(resolved)
    if root.handlers:
        return
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(DEFAULT_FORMAT))
    root.addHandler(handler)
    root.propagate = False


__all__ = ["configure_logging"]
