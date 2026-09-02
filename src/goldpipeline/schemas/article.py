"""What kind of article a Run is meant to produce.

Three product modes, and the enum is the whole vocabulary. A producer may say
which of these it wants; it may not say anything else about how the pipeline
behaves - not a prompt, not a model, not a destination.

Kept in its own module because both ends of the pipeline need it - the inbox
payload that carries it and the Run manifest that records it - and neither
should have to import the other to name a product mode.
"""

from __future__ import annotations

from enum import StrEnum


class ArticleType(StrEnum):
    """The article modes this pipeline knows about.

    ``ANALYSIS`` is the mode every Run before this enum existed was written in,
    which is why it is the default: an event that says nothing means what events
    have always meant.

    The other two are declared before they are executable, deliberately. A
    schema that cannot express ``TRADE_PLAN`` forces the day it arrives to be a
    schema migration *and* a feature launch at once; a schema that can express
    it, paired with a routing table that refuses to run it, makes activation a
    single reviewable change.
    """

    ANALYSIS = "ANALYSIS"
    """Market commentary from candles and the analyst's note. Production today."""

    TRADE_PLAN = "TRADE_PLAN"
    """Directional plan with entry, invalidation and targets. Not executable yet."""

    NEWS_DIGEST = "NEWS_DIGEST"
    """Ranked, deduplicated news with impact on gold. Not executable yet."""


__all__ = ["ArticleType"]
