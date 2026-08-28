"""Nonce-delimited fences for untrusted text.

Any text that did not come from a template on disk is untrusted: the analyst's
note, and - from Round 3 onward - the article a model wrote. Both are fenced
between markers carrying a token generated fresh for each request.

The nonce is the point. Text that wants to close the block early and start
issuing orders would have to guess it; a fixed delimiter like ``<source>`` can
simply be typed by the source itself.

Shared by the writer and reviewer prompts so the two cannot drift apart.
"""

from __future__ import annotations

import secrets

NONCE_BYTES = 8
"""64 bits. Long enough that guessing is hopeless, short enough to stay readable."""


def make_nonce() -> str:
    """Generate a fresh fence token."""
    return secrets.token_hex(NONCE_BYTES)


def fence_marker(nonce: str, position: str, label: str) -> str:
    """Render one fence marker.

    Args:
        nonce: Per-request token.
        position: ``BEGIN`` or ``END``.
        label: What the block holds, e.g. ``UNTRUSTED_SOURCE``.
    """
    return f"<<<{position}_{label}_{nonce}>>>"


def fenced_block(nonce: str, label: str, text: str) -> str:
    """Wrap *text* between BEGIN/END markers."""
    return "\n".join(
        (
            fence_marker(nonce, "BEGIN", label),
            text,
            fence_marker(nonce, "END", label),
        )
    )


def extract_fenced(rendered: str, nonce: str, label: str) -> str:
    """Recover the text inside a fence.

    Splitting from the right on BEGIN matters: a prompt names its markers in
    prose *before* the block, so a plain search finds the explanation instead.
    """
    begin = fence_marker(nonce, "BEGIN", label)
    end = fence_marker(nonce, "END", label)
    return rendered.rsplit(begin, 1)[1].split(end, 1)[0]


__all__ = [
    "NONCE_BYTES",
    "extract_fenced",
    "fence_marker",
    "fenced_block",
    "make_nonce",
]
