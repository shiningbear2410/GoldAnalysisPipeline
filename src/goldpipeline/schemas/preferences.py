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
from goldpipeline.schemas.secrets import SecretName

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
    """How close a provider is to actually being callable.

    Four states rather than two, because "the code exists" and "it will work"
    are different facts and a status line that conflates them is a status line
    that shows green for a call nobody can make.
    """

    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    """No adapter exists. Selectable so a menu can show what is coming."""

    IMPLEMENTED = "IMPLEMENTED"
    """An adapter exists; whether its credential is present was not checked.

    The honest answer when nobody asked the credential store. Deliberately not
    ``AVAILABLE``: unchecked is not the same as present, and only a real probe
    may upgrade this.
    """

    IMPLEMENTED_NOT_CONFIGURED = "IMPLEMENTED_NOT_CONFIGURED"
    """An adapter exists and its credential was looked for and not found."""

    AVAILABLE = "AVAILABLE"
    """An adapter exists and its credential is present. The only green one."""


class ThinkingMode(StrEnum):
    """Whether a selection asks the vendor to think before answering.

    Carried per *selection* rather than per vendor model, because that is where
    the distinction now lives: two of the four DeepSeek choices point at the
    same vendor model and differ only in this.
    """

    ENABLED = "ENABLED"
    DISABLED = "DISABLED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    """The provider has no such control. Claude, today."""


@dataclass(frozen=True)
class ModelSpec:
    """One choice a user may pick, and what it means at the vendor.

    **Three ids, deliberately.** Until DeepSeek retired its aliases these were
    one string, and collapsing them was fine. They are not one string any more:

    * ``selection_id`` is what a person picked and what gets stored. It must
      stay stable across vendor changes, because it is the thing an operator
      recognises and the thing a preferences file remembers.
    * ``api_model_id`` is what goes on the wire. It changes when the vendor
      changes it, and nothing outside the adapter should care.
    * ``label`` is what appears on a button.

    Keeping a retired alias as the wire value would send the vendor something it
    no longer accepts; dropping the alias as a *choice* would take away an option
    the operator asked for. Separating the two costs one field and keeps both.
    """

    selection_id: str
    label: str
    api_model_id: str
    thinking: ThinkingMode = ThinkingMode.NOT_APPLICABLE


@dataclass(frozen=True)
class ProviderSpec:
    """One provider, its choices, and what it needs before it can be called."""

    provider: Provider
    label: str
    implemented: bool
    secret: SecretName
    models: tuple[ModelSpec, ...]
    requires: str = ""
    """What is still missing, in words an operator can act on."""

    def model(self, selection_id: str) -> ModelSpec | None:
        return next((m for m in self.models if m.selection_id == selection_id), None)

    def readiness(self, *, secret_present: bool | None) -> RuntimeReadiness:
        """How runnable this provider is, given what is known about its key.

        ``None`` means nobody looked. That is its own answer rather than a
        pessimistic guess, and it is what this round reports: no credential
        store is consulted here, so nothing may claim to be configured.
        """
        if not self.implemented:
            return RuntimeReadiness.NOT_IMPLEMENTED
        if secret_present is None:
            return RuntimeReadiness.IMPLEMENTED
        return (
            RuntimeReadiness.AVAILABLE
            if secret_present
            else RuntimeReadiness.IMPLEMENTED_NOT_CONFIGURED
        )


CATALOG: tuple[ProviderSpec, ...] = (
    ProviderSpec(
        provider=Provider.CLAUDE,
        label="Claude API",
        implemented=True,
        secret=SecretName.ANTHROPIC_API_KEY,
        models=(
            ModelSpec(
                selection_id="claude-haiku-4-5",
                label="Haiku 4.5",
                api_model_id="claude-haiku-4-5",
            ),
            ModelSpec(
                selection_id="claude-sonnet-5",
                label="Sonnet 5",
                api_model_id="claude-sonnet-5",
            ),
            ModelSpec(
                selection_id="claude-opus-5",
                label="Opus 5",
                api_model_id="claude-opus-5",
            ),
        ),
    ),
    ProviderSpec(
        provider=Provider.DEEPSEEK,
        label="DeepSeek API",
        implemented=True,
        secret=SecretName.DEEPSEEK_API_KEY,
        models=(
            ModelSpec(
                selection_id="deepseek-v4-pro",
                label="DeepSeek-V4 Pro",
                api_model_id="deepseek-v4-pro",
                thinking=ThinkingMode.ENABLED,
            ),
            ModelSpec(
                selection_id="deepseek-v4-flash",
                label="DeepSeek-V4 Flash",
                api_model_id="deepseek-v4-flash",
                thinking=ThinkingMode.ENABLED,
            ),
            ModelSpec(
                selection_id="deepseek-chat",
                label="DeepSeek Chat",
                api_model_id="deepseek-v4-flash",
                thinking=ThinkingMode.DISABLED,
            ),
            ModelSpec(
                selection_id="deepseek-reasoner",
                label="DeepSeek Reasoner",
                api_model_id="deepseek-v4-flash",
                thinking=ThinkingMode.ENABLED,
            ),
        ),
        requires="a DeepSeek API key stored in the credential manager",
    ),
)
"""Every choice a user may select, in menu order.

One table, so a future bot renders a keyboard from it rather than carrying its
own copy - which is how a menu comes to offer something the server refuses. A
tuple rather than a mapping: the order is what a person sees, and that should be
a decision made here rather than whatever a hash produces.

**Two DeepSeek choices point at the same vendor model.** ``deepseek-chat`` and
``deepseek-reasoner`` were vendor aliases until they were retired on 2026-07-24;
the two current ids are ``deepseek-v4-pro`` and ``deepseek-v4-flash``, and both
support thinking and non-thinking modes. So the four choices survive as choices,
``DeepSeek Reasoner`` and ``DeepSeek-V4 Flash`` resolve to the same runtime
behaviour, and ``DeepSeek Chat`` is that model with thinking off.

That collision is deliberately not disguised. Inventing a difference the vendor
no longer offers would mean a menu that promises something the API cannot do,
and the first person to notice would be a reader wondering why two buttons
produce the same article.
"""

DEFAULT_PROVIDER = Provider.CLAUDE
DEFAULT_SELECTION_ID = "claude-opus-5"
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


def resolve_model(provider: Provider, selection_id: str) -> ModelSpec:
    """The choice *provider* offers under *selection_id*.

    Raises:
        ValueError: The provider does not offer that choice. Compatibility is
            decided here, server-side, from the table - never inferred from the
            shape of an id. ``claude-`` is a naming convention, and a rule that
            read a provider out of a prefix would be a rule an id could lie to.
    """
    spec = provider_spec(provider)
    model = spec.model(selection_id)
    if model is None:
        offered = ", ".join(m.selection_id for m in spec.models)
        raise ValueError(f"{spec.label} does not offer {selection_id!r}; it offers: {offered}")
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
    selection_id: str = Field(
        default=DEFAULT_SELECTION_ID,
        description=(
            "A choice this catalog declares for `provider`. Never free text, and "
            "never a vendor model id: what goes on the wire is the catalog's job."
        ),
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
        resolve_model(self.provider, self.selection_id)
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

    selection_id: str | None = Field(
        default=None, description="What the operator picked. Stable across vendor changes."
    )
    model_label: str | None = None
    api_model_id: str | None = Field(
        default=None, description="What this selection actually sends to the vendor."
    )
    thinking: ThinkingMode | None = None

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
    "DEFAULT_SELECTION_ID",
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
    "ThinkingMode",
    "RuntimeReadiness",
    "UserPreferences",
    "provider_spec",
    "resolve_model",
]
