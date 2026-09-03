"""Reading and changing the operator's preferences, safely.

Three properties, and each exists because of a specific way this could go wrong.

**A damaged file is never silently replaced.** The easy implementation returns
defaults whenever the file will not parse, and it is wrong in the one case that
matters: an operator who set Sonnet 5 last week, whose file was truncated by a
power cut, would be told their preference is Opus 5 and would have no way to
know it had changed. So an unreadable file is reported as unreadable, reads
carry where the answer came from, and a mutation on top of a damaged file is
refused rather than performed against invented state.

**Every write replaces the whole object atomically.** Preferences are small and
interdependent - a provider and a model only mean anything together - so there
is no partial write to reason about. A temporary file in the same directory,
fsynced, then ``os.replace``: a reader sees the old document or the new one.

**Mutation is typed, never keyed.** There is no ``set(key, value)``. A future
Telegram callback payload is attacker-influenced text, and the shortest path
from that text to a config file is a function that accepts a key name. The
operations here take enums and integers, so the payload can only ever select
among choices this code already declared.

**Single-writer, and that is a stated contract rather than an accident.** The
intended writer is a one-shot bot process, exactly like the automation tick that
already serialises itself with ``.worker.lock``. A ``RunLock`` here was
considered and rejected: that lock is deliberately never cleared automatically,
so a bot killed mid-write would leave preferences permanently unwritable until a
human intervened. Losing an unlucky concurrent preference update is a smaller
harm than that, and unlike a duplicate publish it is visible and repeatable. So
concurrent writers get documented last-writer-wins on the whole object, never a
corrupt file, and a test pins it.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from pydantic import ValidationError as PydanticValidationError

from goldpipeline.domain.errors import PreferencesUnavailableError
from goldpipeline.schemas.article import ArticleType
from goldpipeline.schemas.preferences import (
    DEFAULT_PREFERENCES,
    PREFERENCES_FILENAME,
    PREFERENCES_SCHEMA_VERSION,
    PreferencesHealth,
    PreferencesSource,
    PreferencesStatus,
    Provider,
    UserPreferences,
    provider_spec,
    resolve_model,
)
from goldpipeline.schemas.secrets import SecretName
from goldpipeline.services.article_routing import spec_for
from goldpipeline.storage.atomic import atomic_write_bytes, encode_json

logger = logging.getLogger(__name__)

SecretProbe = Callable[[SecretName], bool]
"""How a caller answers "is this provider's credential stored?".

Injected rather than imported, so this module never reaches a credential
store on its own and a status render costs nothing until somebody asks it to.
"""


@dataclass(frozen=True)
class PreferencesRead:
    """One read, and everything a caller needs to judge it."""

    preferences: UserPreferences | None
    source: PreferencesSource
    health: PreferencesHealth
    detail: str | None = None

    @property
    def usable(self) -> UserPreferences:
        """The selections, or a refusal naming what is wrong with the file.

        Raises:
            PreferencesUnavailableError: The stored document is unusable. There
                is deliberately no fallback to defaults: answering with a
                selection nobody made is how an operator ends up publishing with
                a model they did not choose.
        """
        if self.preferences is None:
            raise PreferencesUnavailableError(
                f"the stored preferences are {self.health}: {self.detail}",
                health=str(self.health),
            )
        return self.preferences

    @property
    def healthy(self) -> bool:
        return self.health is PreferencesHealth.OK


class PreferencesStore:
    """The operator's selections, in one small file."""

    def __init__(self, automation_dir: Path | str) -> None:
        self.root = Path(automation_dir)

    @property
    def path(self) -> Path:
        return self.root / PREFERENCES_FILENAME

    # -- reading -----------------------------------------------------------

    def read(self) -> PreferencesRead:
        """Load the stored selections, or say precisely why they cannot be.

        Never raises for a bad file and never rewrites one. Both would take the
        decision away from the caller, and the caller - a bot answering a person
        - is the one that can say "your saved preferences are damaged" instead
        of quietly acting on something else.
        """
        path = self.path
        if not path.is_file():
            return PreferencesRead(
                preferences=DEFAULT_PREFERENCES,
                source=PreferencesSource.DEFAULT,
                health=PreferencesHealth.OK,
            )

        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            return self._broken(PreferencesHealth.UNREADABLE, f"the file could not be read: {exc}")

        if not isinstance(document, dict):
            return self._broken(
                PreferencesHealth.UNREADABLE,
                f"the document is a {type(document).__name__}, not an object",
            )

        version = document.get("schema_version")
        if version != PREFERENCES_SCHEMA_VERSION:
            # Checked before validation so the message names the real problem. A
            # v2 document would otherwise be reported as a pile of unknown
            # fields, which sends the reader looking in the wrong direction.
            return self._broken(
                PreferencesHealth.UNSUPPORTED_VERSION,
                f"schema_version {version!r} is not {PREFERENCES_SCHEMA_VERSION!r}",
            )

        try:
            preferences = UserPreferences.model_validate(document)
        except PydanticValidationError as exc:
            return self._broken(PreferencesHealth.INVALID, _first_problem(exc))

        return PreferencesRead(
            preferences=preferences,
            source=PreferencesSource.FILE,
            health=PreferencesHealth.OK,
        )

    def _broken(self, health: PreferencesHealth, detail: str) -> PreferencesRead:
        logger.warning("preferences.%s path=%s", health.lower(), self.path)
        return PreferencesRead(
            preferences=None,
            source=PreferencesSource.FILE,
            health=health,
            detail=detail,
        )

    # -- writing -----------------------------------------------------------

    def write(self, preferences: UserPreferences) -> Path:
        """Replace the whole document atomically.

        Revalidated on the way in even though the argument is already typed:
        this is the last place the bytes are decided, and a model constructed by
        some future path that skipped validation should not be able to reach the
        disk through it.
        """
        checked = UserPreferences.model_validate(preferences.model_dump(mode="json"))
        self.root.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(self.path, encode_json(checked))
        logger.info(
            "preferences.write provider=%s model=%s type=%s lookback=%ds",
            checked.provider,
            checked.selection_id,
            checked.article_type,
            checked.news_lookback_seconds,
        )
        return self.path

    # -- typed mutation ----------------------------------------------------

    def set_provider_model(self, provider: Provider, selection_id: str) -> UserPreferences:
        """Choose a provider and a model together.

        One operation, because they are one decision. Setting them separately
        would mean passing through a state where the pair is not offered - and
        either that intermediate state gets written, or the second call has to
        repair the first, and both are worse than asking for both at once.

        Raises:
            ValueError: The catalog does not offer that pairing.
            PreferencesUnavailableError: The stored file is damaged.
        """
        resolve_model(provider, selection_id)
        return self._update(provider=provider, selection_id=selection_id)

    def set_article_type(self, article_type: ArticleType) -> UserPreferences:
        """Choose which product mode to produce.

        Any of the three may be stored, including one that cannot run yet.
        Readiness is not copied here - it is asked of the routing table when the
        status is rendered, so a preference can never be the thing that decides
        whether an unfinished mode executes.
        """
        return self._update(article_type=article_type)

    def set_news_lookback(self, lookback: timedelta) -> UserPreferences:
        """Choose how far back the news window reaches.

        Bounded by the constants the collector itself enforces, so a window that
        would be clamped later is refused now instead.

        Raises:
            ValueError: The duration is outside the permitted range.
            PreferencesUnavailableError: The stored file is damaged.
        """
        return self._update(news_lookback_seconds=int(lookback.total_seconds()))

    def _update(self, **change: object) -> UserPreferences:
        """Read, apply one typed change, revalidate everything, write.

        The read comes first and its health is honoured: a mutation on top of a
        document nobody could parse would be writing a mixture of one field the
        caller chose and three that were guessed. Repairing the file as a side
        effect of an unrelated change is exactly the silent overwrite this
        module exists to avoid.
        """
        current = self.read()
        if current.preferences is None:
            raise PreferencesUnavailableError(
                f"the stored preferences are {current.health} and were not modified: "
                f"{current.detail}. Fix or remove {self.path.name} first.",
                health=str(current.health),
            )

        try:
            updated = UserPreferences.model_validate(
                {**current.preferences.model_dump(mode="json"), **change}
            )
        except PydanticValidationError as exc:
            # Nothing has been written at this point, and nothing will be.
            raise ValueError(_first_problem(exc)) from exc

        self.write(updated)
        return updated

    # -- the read model ----------------------------------------------------

    def status(self, *, secret_present: SecretProbe | None = None) -> PreferencesStatus:
        """What a future ``/status`` command may show.

        Display labels where a person is reading, readiness asked of the routing
        table rather than remembered here, and nothing that touches a
        credential unless the caller hands over a way to look.

        Args:
            secret_present: An optional probe answering "is this provider's key
                stored?". Omitted here and in every test, so this round reports
                ``IMPLEMENTED`` - the honest answer when nobody looked - rather
                than claiming a provider is configured because its code exists.
                A bot that wants a green light passes a real probe.
        """
        current = self.read()
        preferences = current.preferences

        if preferences is None:
            return PreferencesStatus(
                source=current.source,
                health=current.health,
                detail=current.detail,
            )

        provider = provider_spec(preferences.provider)
        model = resolve_model(preferences.provider, preferences.selection_id)
        routing = spec_for(preferences.article_type)
        available = secret_present(provider.secret) if secret_present else None

        return PreferencesStatus(
            source=current.source,
            health=current.health,
            provider=preferences.provider,
            provider_label=provider.label,
            provider_runtime=provider.readiness(secret_present=available),
            provider_requires=provider.requires or None,
            selection_id=model.selection_id,
            model_label=model.label,
            api_model_id=model.api_model_id,
            thinking=model.thinking,
            article_type=preferences.article_type,
            article_type_ready=routing.ready,
            article_type_requires=routing.requires or None,
            news_lookback_seconds=preferences.news_lookback_seconds,
        )


def _first_problem(exc: PydanticValidationError) -> str:
    """One readable sentence from a validation error.

    The first problem, named by field. A pydantic error dump is accurate and
    unreadable, and this string ends up in front of a person.
    """
    errors = exc.errors()
    if not errors:  # pragma: no cover - pydantic always reports at least one
        return "the document is not a valid preferences object"
    first = errors[0]
    where = ".".join(str(part) for part in first["loc"]) or "document"
    return f"{where}: {first['msg']}"


__all__ = [
    "PreferencesHealth",
    "SecretProbe",
    "PreferencesRead",
    "PreferencesSource",
    "PreferencesStore",
]
