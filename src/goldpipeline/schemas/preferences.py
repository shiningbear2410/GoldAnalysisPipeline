"""What the operator has chosen, and the closed catalog they may choose from.

**Preferences are not configuration, and the distinction is load-bearing.**
``config.json`` holds infrastructure that must be right or the worker must not
run: a symbol, a chat id, a staleness limit. It is validated by
``REQUIRED_PRODUCTION_KEYS = frozenset(ConfigKey)``, so *every* member of that
enum is mandatory in scheduled mode. A trader's choice of model has no business
in that set - adding it would mean an operator who never expressed a preference
has a worker that refuses to start. So these live in their own file, beside the
automation worker's other mutable state, and their absence is a valid answer.

**Every value is code-defined.** ``provider`` and ``article_type`` are closed
enums; ``model_id`` must name a model this catalog declares *for that provider*;
``news_lookback_seconds`` is bounded by the same constants the collector
enforces. There is no free-text field anywhere in this module, which is what
makes a future callback payload unable to become a file key.

**Nothing here can name a secret, a path, a chat, or a publish behaviour.** Not
because such values are filtered - because no field for them exists, and
``extra="forbid"`` means one cannot be added by whatever writes the file.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from goldpipeline.schemas.article import ArticleType
from goldpipeline.schemas.common import StrictModel
from goldpipeline.schemas.news import DEFAULT_LOOKBACK, MAX_LOOKBACK, MIN_LOOKBACK

PREFERENCES_SCHEMA_VERSION = "1"
"""Version of the preferences document.

A document announcing a version this code does not know is refused rather than
read leniently: the fields it does recognise might mean something different
under the newer contract.
"""

PREFERENCES_FILENAME = "preferences.json"
"""Beside ``state.json`` in the automation directory.

The same neighbourhood as the worker's other mutable state, which is the honest
place for it: overwritten in normal operation, deletable without losing an
article, and never consulted to decide whether something is safe to publish.
"""


class Provider(StrEnum):
    """User-facing generation providers. A closed set, not a string."""

    CLAUDE = "CLAUDE"
    DEEPSEEK = "DEEPSEEK"


class RuntimeReadiness(StrEnum):
    """Whether a provider can actually be called today.

    Deliberately separate from whether it may be *selected*. The catalog is the
    menu; this says which dishes the kitchen can currently cook. Conflating them
    would mean either hiding a planned option or pretending an unimplemented one
    works, and the second is how a bot ends up reporting success for a call it
    never made.
    """

    AVAILABLE = "AVAILABLE"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"


@dataclass(frozen=True)
class ModelSpec:
    """One model a user may pick, and what to call it on screen."""

    model_id: str
    """The provider's own id, e.g. ``claude-opus-5``. Never shown as the label."""

    label: str
    """What a person sees, e.g. ``Opus 5``."""


@dataclass(frozen=True)
class ProviderSpec:
    """One provider, its models, and whether it can be called yet."""

    provider: Provider
    label: str
    runtime: RuntimeReadiness
    models: tuple[ModelSpec, ...]
    requires: str = ""
    """What is still missing, when ``runtime`` is not ``AVAILABLE``."""

    @property
    def available(self) -> bool:
        return self.runtime is RuntimeReadiness.AVAILABLE

    def model(self, model_id: str) -> ModelSpec | None:
        return next((m for m in self.models if m.model_id == model_id), None)


CATALOG: tuple[ProviderSpec, ...] = (
    ProviderSpec(
        provider=Provider.CLAUDE,
        label="Claude API",
        runtime=RuntimeReadiness.AVAILABLE,
        models=(
            ModelSpec(model_id="claude-haiku-4-5", label="Haiku 4.5"),
            ModelSpec(model_id="claude-sonnet-5", label="Sonnet 5"),
            ModelSpec(model_id="claude-opus-5", label="Opus 5"),
        ),
    ),
    ProviderSpec(
        provider=Provider.DEEPSEEK,
        label="DeepSeek API",
        runtime=RuntimeReadiness.NOT_IMPLEMENTED,
        models=(
            ModelSpec(model_id="deepseek-v4-pro", label="DeepSeek-V4 Pro"),
            ModelSpec(model_id="deepseek-v4-flash", label="DeepSeek-V4 Flash"),
            ModelSpec(model_id="deepseek-chat", label="DeepSeek Chat"),
            ModelSpec(model_id="deepseek-reasoner", label="DeepSeek Reasoner"),
        ),
        requires="a DeepSeek writer and finalizer client, and a stored API key",
    ),
)
"""Every provider/model combination a user may select, in menu order.

One table, so a future bot renders a keyboard from it rather than carrying its
own copy of the model list - which is how a menu comes to offer something the
server refuses.

A tuple rather than a dict of sets: the order is what a person sees, and it
should be a decision made here rather than whatever a hash happens to produce.

DeepSeek is listed and *not* runnable. Declaring it before it works is the same
choice the article-routing table makes for ``TRADE_PLAN``: the day it arrives is
one reviewable change to this table plus the client it points at, rather than a
schema migration under pressure.
"""

DEFAULT_PROVIDER = Provider.CLAUDE
DEFAULT_MODEL_ID = "claude-opus-5"
"""What an operator who has never expressed a preference gets.

Chosen to match what production already does, not to improve on it. The writer
and finalizer both resolve their model from ``config.DEFAULT_MODEL``, which is
``claude-opus-5``; anything else here would quietly change the model that writes
production articles as a side effect of adding a preferences file, which is
exactly the kind of change nobody would think to look for. A test pins the two
together so they cannot drift apart silently.
"""

DEFAULT_ARTICLE_TYPE = ArticleType.ANALYSIS
"""The mode every Run before article types existed was written in."""


def provider_spec(provider: Provider) -> ProviderSpec:
    """The catalog entry for *provider*."""
    return next(spec for spec in CATALOG if spec.provider is provider)


def resolve_model(provider: Provider, model_id: str) -> ModelSpec:
    """The model *provider* offers under *model_id*.

    Raises:
        ValueError: The provider does not offer that model. Compatibility is
            decided here, server-side, from the table - never inferred from the
            shape of the id. ``claude-`` is a naming convention, and a rule that
            read a provider out of a prefix would be a rule an id could lie to.
    """
    spec = provider_spec(provider)
    model = spec.model(model_id)
    if model is None:
        offered = ", ".join(m.model_id for m in spec.models)
        raise ValueError(f"{spec.label} does not offer {model_id!r}; it offers: {offered}")
    return model


class UserPreferences(StrictModel):
    """One operator's current selections.

    Four fields, and every one of them is a choice about *what to produce* -
    never about where it goes, who reviews it, or what it costs. There is
    deliberately no reviewer model here: the review exists to disagree with the
    writer, and a setting that let one person move both would quietly remove
    the disagreement.
    """

    schema_version: Literal["1"] = "1"
    provider: Provider = DEFAULT_PROVIDER
    model_id: str = Field(
        default=DEFAULT_MODEL_ID,
        description="A model id this catalog declares for `provider`. Never free text.",
    )
    article_type: ArticleType = Field(
        default=DEFAULT_ARTICLE_TYPE,
        description=(
            "Which product mode to produce. Storing an unfinished one is allowed; "
            "it does not make it runnable - readiness lives in article_routing."
        ),
    )
    news_lookback_seconds: int = Field(
        default=int(DEFAULT_LOOKBACK.total_seconds()),
        ge=int(MIN_LOOKBACK.total_seconds()),
        le=int(MAX_LOOKBACK.total_seconds()),
        description="How far back a generated article's news window reaches.",
    )

    @model_validator(mode="after")
    def _model_belongs_to_provider(self) -> UserPreferences:
        """Refuse a pair the catalog does not offer.

        Checked on the whole object rather than field by field, because neither
        field is wrong on its own: ``DEEPSEEK`` is a real provider and
        ``claude-opus-5`` is a real model. It is the pairing that is not a thing.
        """
        resolve_model(self.provider, self.model_id)
        return self

    @property
    def news_lookback(self) -> timedelta:
        return timedelta(seconds=self.news_lookback_seconds)


DEFAULT_PREFERENCES = UserPreferences()
"""The absent-file state, as a value rather than as a special case."""


# --------------------------------------------------------------------------
# the read model
# --------------------------------------------------------------------------


class PreferencesSource(StrEnum):
    """Where an answer came from.

    A reader has to be able to tell "the operator chose Opus 5" from "nobody has
    chosen anything and Opus 5 is the default", because only one of those is a
    decision somebody made.
    """

    DEFAULT = "DEFAULT"
    FILE = "FILE"


class PreferencesHealth(StrEnum):
    """Whether the stored document could be used."""

    OK = "OK"

    UNREADABLE = "UNREADABLE"
    """Not valid UTF-8, not JSON, not an object, or the file could not be read."""

    UNSUPPORTED_VERSION = "UNSUPPORTED_VERSION"
    """A document from a newer contract. Refused rather than read leniently."""

    INVALID = "INVALID"
    """Parsed, but not a usable selection - an unknown model, a pairing the
    catalog does not offer, an out-of-range window, or an unknown key."""


class PreferencesStatus(StrictModel):
    """The safe view of current selections, for a future ``/status`` to render.

    A read model, and a deliberately narrow one. Every field is either a
    selection the operator made or a label from a code-defined table; nothing
    here is derived from a credential, an environment variable, a path or the
    production configuration, because none of those is reachable from the
    document this is built out of.

    The selection fields are optional so that a damaged file still produces a
    renderable status: an operator whose preferences will not parse needs to be
    told that, not handed a screen full of defaults that look like their
    choices.
    """

    source: PreferencesSource
    health: PreferencesHealth
    detail: str | None = Field(
        default=None, max_length=400, description="Why the stored document is unusable."
    )

    provider: Provider | None = None
    provider_label: str | None = None
    provider_runtime: RuntimeReadiness | None = Field(
        default=None,
        description=(
            "Whether this provider can actually be called yet - a different "
            "question from whether it may be picked."
        ),
    )
    provider_requires: str | None = Field(
        default=None, description="What the provider still needs, when it is not runnable."
    )

    model_id: str | None = None
    model_label: str | None = None

    article_type: ArticleType | None = None
    article_type_ready: bool | None = Field(
        default=None,
        description="Asked of article_routing, never remembered here.",
    )
    article_type_requires: str | None = Field(
        default=None, description="What the mode still needs, when it is not ready."
    )

    news_lookback_seconds: int | None = Field(default=None, ge=1)

    @property
    def usable(self) -> bool:
        """Whether the selections could be read at all."""
        return self.health is PreferencesHealth.OK

    @property
    def generation_ready(self) -> bool:
        """Whether the current selections could actually produce an article.

        Both halves must hold: a provider that can be called, and a mode that has
        an implementation. Reported rather than enforced - what to do about a
        selection that cannot run is the caller's decision, and this view exists
        so the caller can make it with the facts in front of it.
        """
        return bool(
            self.usable
            and self.provider_runtime is RuntimeReadiness.AVAILABLE
            and self.article_type_ready
        )


__all__ = [
    "CATALOG",
    "DEFAULT_ARTICLE_TYPE",
    "DEFAULT_MODEL_ID",
    "DEFAULT_PREFERENCES",
    "DEFAULT_PROVIDER",
    "PREFERENCES_FILENAME",
    "PREFERENCES_SCHEMA_VERSION",
    "ModelSpec",
    "PreferencesHealth",
    "PreferencesSource",
    "PreferencesStatus",
    "Provider",
    "ProviderSpec",
    "RuntimeReadiness",
    "UserPreferences",
    "provider_spec",
    "resolve_model",
]
