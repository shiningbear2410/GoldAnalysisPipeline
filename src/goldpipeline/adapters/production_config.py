"""Strict configuration loading for the unattended scheduled worker.

The companion to :mod:`goldpipeline.adapters.config_store`, and deliberately a
*different* contract rather than a flag on the same one.

**The defect this module exists to prevent.** The layered loader treats an
absent configuration file as an empty one, then fills every setting from a
built-in default. For a person running a command that is a convenience. For a
task firing every minute it is a silent failure: the worker read no
configuration, decided automation was off because ``false`` is the default,
reported ``exit 0``, and did that 420 times before anyone noticed. Task
Scheduler showed a green history the whole time. The state "an operator
switched automation off" and the state "the configuration is gone" produced
byte-identical evidence.

So the scheduled worker gets the opposite rule: **the file, in full, or
nothing**. Every approved key must be present explicitly, the two kill switches
most of all, because those are precisely the settings whose default is
indistinguishable from their absence.

**No second production path.** On Windows the file lives under
``%LOCALAPPDATA%`` and nowhere else. When that variable is missing the answer is
an error, not ``~/.config``: a fallback path is a second place for the truth to
hide, and the failure it produces is the quiet kind. Other platforms keep the
XDG behaviour, because there the profile directory is not the thing being
protected and development matters more.

**Why the process environment is not layered in here.** A scheduled task
inherits no session, so in production the layer would always be empty - but it
would also make the recorded fingerprint a lie, describing a file rather than
the settings the tick actually ran on. The fingerprint is the whole point, so
the file is the sole source and the SHA-256 always describes what was used.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from goldpipeline.domain.errors import (
    ConfigPathUnavailableError,
    PersistentConfigIncompleteError,
    PersistentConfigInvalidJsonError,
    PersistentConfigNotFoundError,
    PersistentConfigSchemaMismatchError,
    PersistentConfigSecretKeyError,
    PersistentConfigUnknownKeyError,
    PersistentConfigUnreadableError,
    ProductionConfigError,
)
from goldpipeline.schemas.runtime_config import (
    APP_DIR_NAME,
    BOOLEAN_KEYS,
    CONFIG_FILENAME,
    FALSE_VALUES,
    FORBIDDEN_KEYS,
    REQUIRED_PRODUCTION_KEYS,
    RUNTIME_CONFIG_SCHEMA_VERSION,
    TRUE_VALUES,
    ConfigKey,
    ProductionConfig,
    ProductionConfigReport,
    ProductionConfigStatus,
)

logger = logging.getLogger(__name__)

_ALLOWED_DOCUMENT_FIELDS = frozenset({"schema_version", "values"})


def production_config_path(
    env: Mapping[str, str] | None = None, *, windows: bool | None = None
) -> Path:
    """Where the unattended worker reads its configuration.

    Args:
        env: Environment to resolve against. Defaults to the process's own.
        windows: Whether to apply Windows rules. Defaults to the real platform;
            injected by tests so the no-fallback rule can be proven on any host.

    Returns:
        ``%LOCALAPPDATA%/GoldAnalysisPipeline/config.json`` on Windows.

    Raises:
        ConfigPathUnavailableError: Windows, and ``%LOCALAPPDATA%`` is unset.
            No ``~/.config`` fallback is attempted - a scheduled task reading a
            path nobody configured is how the original defect stayed invisible.
    """
    source = os.environ if env is None else env
    is_windows = os.name == "nt" if windows is None else windows

    local = source.get("LOCALAPPDATA")
    if local and local.strip():
        return Path(local.strip()) / APP_DIR_NAME / CONFIG_FILENAME

    if is_windows:
        raise ConfigPathUnavailableError(
            "%LOCALAPPDATA% is not set, so the production configuration path "
            "cannot be resolved. Refusing to guess a second location: a "
            "scheduled task reading the wrong file looks exactly like one "
            "reading the right file and finding nothing.",
            setting="LOCALAPPDATA",
        )

    xdg = source.get("XDG_CONFIG_HOME")
    root = Path(xdg.strip()) if xdg and xdg.strip() else Path.home() / ".config"
    return root / APP_DIR_NAME / CONFIG_FILENAME


def load_production_config(
    path: Path | str | None = None,
    *,
    env: Mapping[str, str] | None = None,
    windows: bool | None = None,
) -> ProductionConfig:
    """Read and fully validate the configuration an unattended tick will use.

    Every failure mode is a distinct exception because they need distinct
    human responses: a missing file wants ``config-set``, an unknown key wants a
    spelling correction, and a forbidden key wants ``secrets-set``.

    Args:
        path: Explicit file to read. Defaults to :func:`production_config_path`.
        env: Environment used to resolve a default path.
        windows: Platform override for path resolution, for tests.

    Returns:
        The validated settings plus the fingerprint of the exact bytes read.

    Raises:
        ProductionConfigError: Any reason the worker must not proceed.
    """
    target = Path(path) if path is not None else production_config_path(env, windows=windows)

    raw = _read_bytes(target)
    document = _parse(target, raw)
    values = _extract_values(target, document)
    _require_complete(target, values)
    _require_explicit_booleans(target, values)
    _require_loadable(target, values)

    config = ProductionConfig(
        path=str(target),
        sha256=hashlib.sha256(raw).hexdigest(),
        schema_version=str(document.get("schema_version", RUNTIME_CONFIG_SCHEMA_VERSION)),
        values=values,
    )
    logger.info(
        "config.production path=%s sha256=%s keys=%d",
        config.path,
        config.sha256[:12],
        len(config.values),
    )
    return config


def inspect_production_config(
    path: Path | str | None = None,
    *,
    env: Mapping[str, str] | None = None,
    windows: bool | None = None,
) -> ProductionConfigReport:
    """The same check, as a report rather than an exception.

    For diagnostics, which must describe an unhealthy machine rather than fail
    on it. The worker itself never uses this: a status command that survives a
    broken configuration and a worker that survives one are opposite
    requirements.
    """
    try:
        config = load_production_config(path, env=env, windows=windows)
    except PersistentConfigNotFoundError as exc:
        return ProductionConfigReport(
            status=ProductionConfigStatus.MISSING,
            path=str(exc.details.get("path")) if exc.details.get("path") else None,
            error_code=exc.code,
            error=exc.message,
        )
    except ProductionConfigError as exc:
        return ProductionConfigReport(
            status=ProductionConfigStatus.INVALID,
            path=str(exc.details.get("path")) if exc.details.get("path") else None,
            error_code=exc.code,
            error=exc.message,
            missing_keys=list(exc.details.get("missing", [])),
        )
    return ProductionConfigReport(
        status=ProductionConfigStatus.FOUND,
        path=config.path,
        schema_version=config.schema_version,
        sha256=config.sha256,
    )


# --------------------------------------------------------------------------
# steps
# --------------------------------------------------------------------------


def _read_bytes(path: Path) -> bytes:
    """The file's exact bytes, distinguishing "absent" from "unreadable"."""
    try:
        return path.read_bytes()
    except FileNotFoundError:
        raise PersistentConfigNotFoundError(
            f"no production configuration at {path}. An unattended worker will "
            "not invent one: run `config-set` for every required setting, "
            "including both kill switches, so that 'switched off' and "
            "'configuration missing' stay distinguishable.",
            path=str(path),
        ) from None
    except OSError as exc:
        # Also the shape a permission problem takes, which is worth separating
        # from "absent": one is a setup step, the other is an ACL to fix.
        raise PersistentConfigUnreadableError(
            f"the production configuration at {path} exists but could not be read.",
            path=str(path),
            reason=type(exc).__name__,
        ) from None


def _parse(path: Path, raw: bytes) -> dict[str, Any]:
    """Decode UTF-8 JSON into a document, or say precisely what is wrong."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise PersistentConfigInvalidJsonError(
            f"the production configuration at {path} is not valid UTF-8.",
            path=str(path),
        ) from None

    try:
        document = json.loads(text)
    except ValueError:
        raise PersistentConfigInvalidJsonError(
            f"the production configuration at {path} is not valid JSON. It is "
            "not treated as empty: an unattended worker running on defaults "
            "because a file was truncated is the failure this check exists for.",
            path=str(path),
        ) from None

    if not isinstance(document, dict):
        raise PersistentConfigInvalidJsonError(
            f"the production configuration at {path} must be a JSON object.",
            path=str(path),
        )

    unknown = sorted(set(document) - _ALLOWED_DOCUMENT_FIELDS)
    if unknown:
        raise PersistentConfigUnknownKeyError(
            f"the production configuration at {path} has unexpected top-level "
            f"fields: {', '.join(unknown)}.",
            path=str(path),
            unknown=unknown,
        )

    declared = document.get("schema_version", RUNTIME_CONFIG_SCHEMA_VERSION)
    if declared != RUNTIME_CONFIG_SCHEMA_VERSION:
        raise PersistentConfigSchemaMismatchError(
            f"the production configuration at {path} declares schema "
            f"{declared!r}, but this build reads {RUNTIME_CONFIG_SCHEMA_VERSION!r}.",
            path=str(path),
            declared=str(declared),
            expected=RUNTIME_CONFIG_SCHEMA_VERSION,
        )
    return document


def _extract_values(path: Path, document: Mapping[str, Any]) -> dict[ConfigKey, str]:
    """Turn the raw ``values`` mapping into typed keys, refusing surprises."""
    values = document.get("values", {})
    if not isinstance(values, dict):
        raise PersistentConfigInvalidJsonError(
            f"the production configuration at {path} must hold `values` as an object.",
            path=str(path),
        )

    resolved: dict[ConfigKey, str] = {}
    for name, value in values.items():
        cleaned = str(name).strip().upper()
        if cleaned in FORBIDDEN_KEYS:
            # Never echoed, only named. The value beside it is a credential.
            raise PersistentConfigSecretKeyError(
                f"{cleaned} appears in {path}. Credentials are never read from "
                "a configuration file; move it to the operating system's "
                "credential manager with `secrets-set` and delete the file entry.",
                path=str(path),
                setting=cleaned,
            )
        try:
            key = ConfigKey(cleaned)
        except ValueError:
            raise PersistentConfigUnknownKeyError(
                f"{cleaned} in {path} is not a setting this build knows. It is "
                "refused rather than ignored, because an unknown key is usually "
                "a misspelt one and dropping it silently leaves the setting the "
                "operator meant to configure sitting at its default.",
                path=str(path),
                setting=cleaned,
            ) from None
        if not isinstance(value, str):
            raise PersistentConfigInvalidJsonError(
                f"{cleaned} in {path} must be a string.",
                path=str(path),
                setting=cleaned,
            )
        resolved[key] = value.strip()
    return resolved


def _require_complete(path: Path, values: Mapping[ConfigKey, str]) -> None:
    """Every approved setting must be present and non-empty."""
    missing = sorted(
        key.value for key in REQUIRED_PRODUCTION_KEYS if not values.get(key, "").strip()
    )
    if missing:
        raise PersistentConfigIncompleteError(
            f"the production configuration at {path} is missing "
            f"{len(missing)} required setting(s): {', '.join(missing)}. "
            "Nothing is filled in from a built-in default in scheduled mode.",
            path=str(path),
            missing=missing,
        )


def _require_explicit_booleans(path: Path, values: Mapping[ConfigKey, str]) -> None:
    """The kill switches must read as an unambiguous true or false.

    Checked separately from completeness so the message can say *why* these two
    matter more than the rest: their default is ``false``, which is also what a
    missing file looks like. An explicit value is the only thing that
    distinguishes a deliberate decision from an absence.
    """
    for key in sorted(BOOLEAN_KEYS, key=lambda item: item.value):
        raw = values[key].strip().lower()
        if raw not in TRUE_VALUES | FALSE_VALUES:
            raise PersistentConfigIncompleteError(
                f"{key.value} in {path} is {values[key]!r}, which is neither "
                "true nor false. A kill switch that cannot be read is not a "
                "kill switch.",
                path=str(path),
                setting=key.value,
            )


def _require_loadable(path: Path, values: Mapping[ConfigKey, str]) -> None:
    """Prove the settings would actually build, reusing the real loaders.

    Same delegation as the interactive writer: bar counts, timeframes and
    destination shapes are validated in exactly one place in this codebase, and
    a scheduled worker must not accept anything ``config-set`` would refuse.
    """
    from goldpipeline.adapters.config_store import validate
    from goldpipeline.domain.errors import RuntimeConfigError

    try:
        validate(values)
    except RuntimeConfigError as exc:
        raise PersistentConfigIncompleteError(
            f"the production configuration at {path} would not load: {exc.message}",
            path=str(path),
            setting=exc.details.get("setting", "unknown"),
        ) from None


__all__ = [
    "inspect_production_config",
    "load_production_config",
    "production_config_path",
]
