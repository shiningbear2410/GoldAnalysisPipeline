"""Turning a catalog selection into a client, for the writer and the finalizer.

The seam a later round will call once preferences actually drive production. It
exists now, separately, so that wiring round is a small change to *what selects*
rather than a large change to *how a client is built*.

**Generation only.** There is no ``build_reviewer_client`` here and there will
not be one. The review exists to disagree with the draft, and it cannot do that
from the same account, the same model and the same setting the draft came from.
The reviewer keeps its own independent Anthropic configuration, and the cheapest
way to guarantee that is for this module to have no way to touch it - a test
asserts the absence rather than trusting the intention.

**Nothing is resolved until it is needed.** Building a Claude client never looks
for a DeepSeek key, and building a DeepSeek client never looks for an Anthropic
one. An operator who has never selected DeepSeek must never be asked for a
DeepSeek credential, and preflight must not fail for the want of one.
"""

from __future__ import annotations

import logging

from goldpipeline.adapters.finalizer_client import FinalizerClient
from goldpipeline.adapters.writer_client import WriterClient
from goldpipeline.schemas.preferences import ModelSpec, Provider, resolve_model

logger = logging.getLogger(__name__)


def build_writer_client(
    provider: Provider,
    selection_id: str,
    *,
    transport: object | None = None,
) -> WriterClient:
    """The writer client for one catalog selection.

    Args:
        provider: Which vendor. A closed enum, never a string from a payload.
        selection_id: The choice the operator made. Validated against the
            catalog before anything is constructed, so an unknown selection
            costs no credential lookup and no client.
        transport: Injected HTTP client for tests. Ignored for Claude, whose
            adapter takes an SDK client instead.

    Raises:
        ValueError: The catalog does not offer that pairing.
        WriterConfigurationError: The chosen provider has no usable credential.
    """
    model = resolve_model(provider, selection_id)
    logger.info("generation.writer provider=%s selection=%s", provider, selection_id)

    if provider is Provider.DEEPSEEK:
        from goldpipeline.adapters.deepseek_writer import build_deepseek_writer

        return build_deepseek_writer(selection_id, transport=transport)

    return _claude_writer(model)


def build_finalizer_client(
    provider: Provider,
    selection_id: str,
    *,
    transport: object | None = None,
) -> FinalizerClient:
    """The finalizer client for one catalog selection.

    Raises:
        ValueError: The catalog does not offer that pairing.
        FinalizeConfigurationError: The chosen provider has no usable credential.
    """
    model = resolve_model(provider, selection_id)
    logger.info("generation.finalizer provider=%s selection=%s", provider, selection_id)

    if provider is Provider.DEEPSEEK:
        from goldpipeline.adapters.deepseek_finalizer import build_deepseek_finalizer

        return build_deepseek_finalizer(selection_id, transport=transport)

    return _claude_finalizer(model)


def _claude_writer(model: ModelSpec) -> WriterClient:
    """Claude, with the selection as its model override.

    For Claude the selection *is* the vendor id, so the override is exact. The
    catalog is still consulted rather than the string passed through: it is the
    one place that says which ids exist.
    """
    from goldpipeline.adapters.anthropic_writer import AnthropicWriterClient
    from goldpipeline.config import WriterSettings

    return AnthropicWriterClient(WriterSettings.from_env(model_override=model.api_model_id))


def _claude_finalizer(model: ModelSpec) -> FinalizerClient:
    """Claude, with the selection as its model override."""
    from goldpipeline.adapters.anthropic_finalizer import AnthropicFinalizerClient
    from goldpipeline.config import FinalizerSettings

    return AnthropicFinalizerClient(FinalizerSettings.from_env(model_override=model.api_model_id))


__all__ = ["build_finalizer_client", "build_writer_client"]
