"""JSON-file implementations of the source protocols.

Used by the Round 1 CLI and by tests. A future Telegram or MT5 adapter drops in
beside these without touching the domain or service layers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from goldpipeline.adapters.base import LoadedSource
from goldpipeline.domain.errors import InputValidationError
from goldpipeline.schemas.market import MarketDataInput
from goldpipeline.schemas.telegram import TelegramAnalysisInput


def _read_payload(path: Path) -> dict[str, Any]:
    """Read a UTF-8 JSON object from *path*.

    Raises:
        InputValidationError: If the file is missing, malformed, or not an object.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise InputValidationError(f"source file not found: {path}", path=str(path)) from exc
    except UnicodeDecodeError as exc:
        raise InputValidationError(
            f"source file is not valid UTF-8: {path}", path=str(path)
        ) from exc

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InputValidationError(
            f"source file is not valid JSON: {path} ({exc.msg} at line {exc.lineno})",
            path=str(path),
        ) from exc

    if not isinstance(payload, dict):
        raise InputValidationError(
            f"source file must contain a JSON object, got {type(payload).__name__}: {path}",
            path=str(path),
        )
    return payload


def _describe(exc: PydanticValidationError) -> list[dict[str, Any]]:
    """Flatten pydantic errors into something safe to store in a manifest."""
    return [
        {"field": ".".join(str(part) for part in error["loc"]), "error": error["msg"]}
        for error in exc.errors()
    ]


class JsonFileAnalysisSource:
    """Loads a :class:`TelegramAnalysisInput` from a JSON file."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def load(self) -> LoadedSource[TelegramAnalysisInput]:
        payload = _read_payload(self.path)
        try:
            model = TelegramAnalysisInput.model_validate(payload)
        except PydanticValidationError as exc:
            raise InputValidationError(
                f"analysis payload failed schema validation: {self.path}",
                path=str(self.path),
                errors=_describe(exc),
            ) from exc
        return LoadedSource(model=model, raw_payload=payload, origin=str(self.path))


class JsonFileMarketDataSource:
    """Loads a :class:`MarketDataInput` from a JSON file."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def load(self) -> LoadedSource[MarketDataInput]:
        payload = _read_payload(self.path)
        try:
            model = MarketDataInput.model_validate(payload)
        except PydanticValidationError as exc:
            raise InputValidationError(
                f"market data payload failed schema validation: {self.path}",
                path=str(self.path),
                errors=_describe(exc),
            ) from exc
        return LoadedSource(model=model, raw_payload=payload, origin=str(self.path))


__all__ = ["JsonFileAnalysisSource", "JsonFileMarketDataSource"]
