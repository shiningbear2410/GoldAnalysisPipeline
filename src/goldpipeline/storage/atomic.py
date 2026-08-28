"""Atomic, UTF-8 JSON writes.

A half-written ``context.json`` is worse than no context at all: a later stage
would happily parse a truncated document. Every write therefore goes to a
temporary file in the *same* directory, is flushed and fsync'd, and only then
replaced into place with :func:`os.replace` - atomic on both NTFS and POSIX.

JSON is written with ``ensure_ascii=False`` so Vietnamese text stays readable
in the stored artifacts rather than turning into ``\u1ea1`` escapes.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel

JSON_INDENT = 2


def sha256_bytes(payload: bytes) -> str:
    """Return the hex SHA-256 digest of *payload*."""
    return hashlib.sha256(payload).hexdigest()


def encode_json(data: Any) -> bytes:
    """Serialize *data* to pretty UTF-8 JSON bytes.

    Pydantic models are dumped through their own JSON serializers first, so
    ``Decimal`` and datetime rendering match the schema definitions exactly.
    """
    if isinstance(data, BaseModel):
        text = data.model_dump_json(indent=JSON_INDENT)
    else:
        text = json.dumps(data, ensure_ascii=False, indent=JSON_INDENT, default=str)
    return text.encode("utf-8") + b"\n"


def atomic_write_bytes(path: Path, payload: bytes) -> str:
    """Write *payload* to *path* atomically. Returns the SHA-256 of the bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return sha256_bytes(payload)


def atomic_write_json(path: Path, data: Any) -> tuple[str, int]:
    """Serialize *data* and write it atomically.

    Returns:
        ``(sha256, size_bytes)`` of the bytes written.
    """
    payload = encode_json(data)
    digest = atomic_write_bytes(path, payload)
    return digest, len(payload)


def encode_text(text: str) -> bytes:
    """Encode markdown or plain text for storage.

    UTF-8 with a single trailing newline, so a Vietnamese article reads
    correctly in any editor and diffs cleanly.
    """
    return (text.rstrip("\n") + "\n").encode("utf-8")


def read_json(path: Path) -> Any:
    """Read UTF-8 JSON from *path*."""
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "JSON_INDENT",
    "atomic_write_bytes",
    "atomic_write_json",
    "encode_json",
    "encode_text",
    "read_json",
    "sha256_bytes",
]
