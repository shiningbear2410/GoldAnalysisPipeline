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

**This module resolves nothing itself.** It has no credential store, no
environment reading and no idea where a key lives; ``env`` and ``secrets``
arrive from the caller that already owns them, and are handed to the settings
loader unchanged. That is the whole reason the defect this signature fixes was
possible: the seam was added without them, so a Run carrying a selection took
the settings loader's *default* - the process environment alone - while the
path beside it passed the composite provider that reaches the operating
system's credential store. A scheduled task inherits no session, so the two
paths agreed on every developer's machine and disagreed in production.

Both parameters are optional and both default to exactly what this module did
before: nothing supplied, environment only. A caller that has no provider is
therefore not made to invent one, and no behaviour changes for anyone who was
already passing nothing.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

from goldpipeline.adapters.finalizer_client import FinalizerClient
from goldpipeline.adapters.secrets import SecretProvider
from goldpipeline.adapters.writer_client import WriterClient
from goldpipeline.schemas.preferences import ModelSpec, Provider, resolve_model

logger = logging.getLogger(__name__)


def build_writer_client(
    provider: Provider,
    selection_id: str,
    *,
    env: Mapping[str, str] | None = None,
    secrets: SecretProvider | None = None,
    transport: object | None = None,
) -> WriterClient:
    """The writer client for one catalog selection.

    Args:
        provider: Which vendor. A closed enum, never a string from a payload.
        selection_id: The choice the operator made. Validated against the
            catalog before anything is constructed, so an unknown selection
            costs no credential lookup and no client.
        env: Mapping for non-secret settings - timeouts, retries, token
            budgets. Defaults to the process environment. Production passes the
            layered view so a scheduled task, which inherits no session, still
            finds the values a person configured once.
        secrets: Where the credential comes from. Defaults to the environment
            alone, so importing this module never reaches a vault. Production
            passes the composite provider that falls back to the operating
            system's credential store.
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

        return build_deepseek_writer(selection_id, env=env, secrets=secrets, transport=transport)

    return _claude_writer(model, env=env, secrets=secrets)


def build_finalizer_client(
    provider: Provider,
    selection_id: str,
    *,
    env: Mapping[str, str] | None = None,
    secrets: SecretProvider | None = None,
    transport: object | None = None,
) -> FinalizerClient:
    """The finalizer client for one catalog selection.

    Takes *env* and *secrets* on the same terms as :func:`build_writer_client`.
    A Run that reaches revision has already been drafted, so a finalizer that
    resolved its credential differently from the writer would fail *after* the
    expensive stage succeeded - the worst place to discover a wiring defect.

    Raises:
        ValueError: The catalog does not offer that pairing.
        FinalizeConfigurationError: The chosen provider has no usable credential.
    """
    model = resolve_model(provider, selection_id)
    logger.info("generation.finalizer provider=%s selection=%s", provider, selection_id)

    if provider is Provider.DEEPSEEK:
        from goldpipeline.adapters.deepseek_finalizer import build_deepseek_finalizer

        return build_deepseek_finalizer(selection_id, env=env, secrets=secrets, transport=transport)

    return _claude_finalizer(model, env=env, secrets=secrets)


def _claude_writer(
    model: ModelSpec,
    *,
    env: Mapping[str, str] | None,
    secrets: SecretProvider | None,
) -> WriterClient:
    """Claude, with the selection as its model override.

    For Claude the selection *is* the vendor id, so the override is exact. The
    catalog is still consulted rather than the string passed through: it is the
    one place that says which ids exist.

    The credential is resolved through *secrets* and never read here. This
    function knows the model; it does not know where a key lives.
    """
    from goldpipeline.adapters.anthropic_writer import AnthropicWriterClient
    from goldpipeline.config import WriterSettings

    return AnthropicWriterClient(
        WriterSettings.from_env(env, model_override=model.api_model_id, secrets=secrets)
    )


def _claude_finalizer(
    model: ModelSpec,
    *,
    env: Mapping[str, str] | None,
    secrets: SecretProvider | None,
) -> FinalizerClient:
    """Claude, with the selection as its model override."""
    from goldpipeline.adapters.anthropic_finalizer import AnthropicFinalizerClient
    from goldpipeline.config import FinalizerSettings

    return AnthropicFinalizerClient(
        FinalizerSettings.from_env(env, model_override=model.api_model_id, secrets=secrets)
    )


__all__ = ["build_finalizer_client", "build_writer_client"]
