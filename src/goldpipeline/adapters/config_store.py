"""Reading and writing the persistent non-secret configuration.

Lives under ``%LOCALAPPDATA%\\GoldAnalysisPipeline\\config.json`` - the user's
own profile, never the repository, so a checkout can be deleted or re-cloned
without losing the machine's settings and a ``git add .`` can never capture them.

**Resolution is environment, then file, then default.** The same shape as the
credential provider and for the same reason: a temporary override must not
require editing a persisted file and remembering to change it back. The
direction is one-way here too - a value found in the environment is never
written to the file.

**Validation reuses the settings classes.** ``config-set`` does not re-implement
"bars must be between 10 and 500" or "the timeframe must be one of six". It
builds the candidate configuration, hands it to the real settings loaders, and
refuses the write if they refuse it. One set of rules, enforced in one place, and
a value that would break the next tick cannot be persisted now.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path

from pydantic import ValidationError as PydanticValidationError

from goldpipeline.domain.errors import RuntimeConfigError, SecretNotPersistableError
from goldpipeline.schemas.runtime_config import (
    APP_DIR_NAME,
    BOOLEAN_KEYS,
    CONFIG_FILENAME,
    FALSE_VALUES,
    FORBIDDEN_KEYS,
    TRUE_VALUES,
    ConfigEntry,
    ConfigKey,
    ConfigSource,
    RuntimeConfig,
)
from goldpipeline.storage.atomic import encode_json

logger = logging.getLogger(__name__)


def default_config_path(env: Mapping[str, str] | None = None) -> Path:
    """Where the persisted configuration lives.

    ``%LOCALAPPDATA%`` rather than roaming: this describes one machine's
    installation - which terminal, which task, which directories - and following
    a user to another computer would be wrong.
    """
    source = os.environ if env is None else env
    base = source.get("LOCALAPPDATA") or source.get("XDG_CONFIG_HOME")
    root = Path(base) if base else Path.home() / ".config"
    return root / APP_DIR_NAME / CONFIG_FILENAME


class RuntimeConfigStore:
    """The persisted settings file, read and written atomically."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path if path is not None else default_config_path()

    # -- reading -----------------------------------------------------------

    def load(self) -> RuntimeConfig:
        """Read the file, or an empty configuration when there is none.

        Raises:
            RuntimeConfigError: The file exists but cannot be parsed. Refusing
                is the point: a corrupt file silently read as "empty" would
                start a scheduled task with every default in place, including a
                symbol and a destination nobody chose.
        """
        if not self.path.is_file():
            return RuntimeConfig()
        try:
            return RuntimeConfig.model_validate_json(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, PydanticValidationError) as exc:
            raise RuntimeConfigError(
                f"the persisted configuration at {self.path} could not be read. "
                "Fix or delete the file; it is not treated as empty, because "
                "that would silently fall back to defaults.",
                path=str(self.path),
            ) from exc

    # -- writing -----------------------------------------------------------

    def set(self, key: ConfigKey, value: str) -> RuntimeConfig:
        """Persist one setting, refusing anything the pipeline could not use.

        Raises:
            SecretNotPersistableError: The name is a credential.
            RuntimeConfigError: The value would not load.
        """
        current = self.load()
        candidate = dict(current.values)
        candidate[key] = value.strip()
        validate(candidate)

        updated = current.model_copy(update={"values": candidate})
        self._write(updated)
        logger.info("config.set key=%s", key.value)
        return updated

    def delete(self, key: ConfigKey) -> RuntimeConfig:
        """Remove one setting, falling the pipeline back to its default."""
        current = self.load()
        if key not in current.values:
            return current
        candidate = {name: item for name, item in current.values.items() if name != key}
        validate(candidate)

        updated = current.model_copy(update={"values": candidate})
        self._write(updated)
        logger.info("config.delete key=%s", key.value)
        return updated

    def _write(self, config: RuntimeConfig) -> None:
        """Replace the file atomically, and only for this user.

        A half-written configuration is worse than none: the next tick would
        parse whatever survived and run with a partial picture of the machine.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = encode_json(config)
        handle, name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent)
        )
        temporary = Path(name)
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            # Owner-only where the platform honours it. Windows inherits the
            # profile directory's ACL, which already excludes other users; this
            # is the cheap belt-and-braces, not an ACL management system.
            os.chmod(temporary, 0o600)
            os.replace(temporary, self.path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise


class LayeredConfig(Mapping[str, str]):
    """Process environment first, then the persisted file.

    Presents itself as a plain mapping so every existing settings loader reads
    through it unchanged - the layering is invisible to code that only wants a
    value, and visible to :meth:`resolve` for code that wants to explain where
    the value came from.
    """

    def __init__(
        self, env: Mapping[str, str] | None = None, config: RuntimeConfig | None = None
    ) -> None:
        self._env: Mapping[str, str] = os.environ if env is None else env
        self._config = config if config is not None else RuntimeConfig()

    def __getitem__(self, key: str) -> str:
        value = self._env.get(key)
        if value is not None and value.strip():
            return value
        try:
            persisted = self._config.get(ConfigKey(key))
        except ValueError:
            persisted = None
        if persisted is None:
            raise KeyError(key)
        return persisted

    def __iter__(self) -> Iterator[str]:
        seen = {key.value for key in self._config.values}
        yield from seen
        for key in self._env:
            if key not in seen:
                yield key

    def __len__(self) -> int:
        return len(set(self._env) | {key.value for key in self._config.values})

    def resolve(self, key: ConfigKey) -> ConfigEntry:
        """The value and which layer supplied it.

        ``DEFAULT`` here means "neither layer has one", so whatever the settings
        class falls back to applies. That is a meaningfully different answer
        from a persisted value that happens to equal the default.
        """
        value = self._env.get(key.value)
        if value is not None and value.strip():
            return ConfigEntry(key=key, value=value.strip(), source=ConfigSource.PROCESS_ENV)
        persisted = self._config.get(key)
        if persisted is not None:
            return ConfigEntry(key=key, value=persisted, source=ConfigSource.PERSISTENT)
        return ConfigEntry(key=key, value=None, source=ConfigSource.DEFAULT)


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def parse_key(name: str) -> ConfigKey:
    """Turn an operator's argument into a key, refusing credentials by name.

    Raises:
        SecretNotPersistableError: The name is a credential.
        RuntimeConfigError: The name is not a setting this store holds.
    """
    cleaned = name.strip().upper()
    if cleaned in FORBIDDEN_KEYS:
        raise SecretNotPersistableError(
            f"{cleaned} is a credential and is never written to a configuration "
            "file. Store it in the operating system's credential manager with "
            f"`secrets-set`, which prompts invisibly.",
            setting=cleaned,
        )
    try:
        return ConfigKey(cleaned)
    except ValueError:
        raise RuntimeConfigError(
            f"{cleaned} is not a setting this store holds. "
            f"Allowed: {', '.join(sorted(key.value for key in ConfigKey))}",
            setting=cleaned,
        ) from None


def validate(values: Mapping[ConfigKey, str]) -> None:
    """Prove a candidate configuration would actually load.

    Deliberately implemented by *using* the real settings classes rather than
    restating their rules. Bar counts, timeframes, minute limits and destination
    shapes are all validated exactly once in this codebase, and a value that
    would break the next scheduled tick is refused now instead.

    Raises:
        RuntimeConfigError: The configuration would not load.
    """
    from goldpipeline.config import AutomationSettings, MarketDataSettings, validate_target_chat

    for key in BOOLEAN_KEYS:
        raw = values.get(key)
        if raw is not None and raw.strip().lower() not in TRUE_VALUES | FALSE_VALUES:
            raise RuntimeConfigError(
                f"{key.value} must be true or false, not {raw!r}. A persisted typo "
                "would quietly mean 'off' across every reboot.",
                setting=key.value,
            )

    mapping = {key.value: value for key, value in values.items()}
    try:
        MarketDataSettings.from_env(mapping)
        AutomationSettings.from_env(mapping)
        target = mapping.get(ConfigKey.TELEGRAM_TARGET_CHAT_ID.value)
        if target:
            validate_target_chat(target)
    except Exception as exc:
        message = getattr(exc, "message", str(exc))
        setting = getattr(exc, "details", {}).get("setting", "unknown")
        raise RuntimeConfigError(
            f"this configuration would not load: {message}", setting=setting
        ) from None


__all__ = [
    "LayeredConfig",
    "RuntimeConfigStore",
    "default_config_path",
    "parse_key",
    "validate",
]
