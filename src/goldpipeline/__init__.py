"""Gold / XAUUSD analysis pipeline.

Round 1 scope: ingest raw analysis + OHLC, normalize, and persist an
immutable Run containing a machine-readable ``context.json``.

No AI provider, publisher, or technical-analysis logic lives here.
"""

from __future__ import annotations

PIPELINE_VERSION = "0.1.0"
"""Version of the pipeline implementation that produced a Run."""

CONTEXT_SCHEMA_VERSION = "1.0.0"
"""Version of the AnalysisContext contract consumed by downstream agents."""

__all__ = ["CONTEXT_SCHEMA_VERSION", "PIPELINE_VERSION"]
