"""Offline finalizer client.

Every test runs against this. Beyond keeping the suite free of networks, keys
and cost, it makes the failure modes that matter here reproducible on demand:
a revision that skips an issue, that declines a CRITICAL one, that quietly
invents a new fact, or that claims a fix it did not make.

The default behaviour is a *real* edit, not a canned string: it reads the
review out of the prompt and applies the corrections the evidence describes. A
fake that returned the draft unchanged would make the postcheck look satisfied
in exactly the cases it exists to catch.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from goldpipeline.adapters.finalizer_client import FinalizeRequest, FinalizeResponse
from goldpipeline.domain.errors import (
    FinalizeError,
    FinalizeProviderError,
    FinalizeResponseError,
    FinalizeTimeoutError,
)
from goldpipeline.schemas.finalizer import (
    FinalizerModelOutput,
    FinalizerUsage,
    IssueResolution,
    ResolutionStatus,
)
from goldpipeline.services.fencing import extract_fenced
from goldpipeline.services.finalizer_prompt import ARTICLE_LABEL, REVIEW_LABEL

FAKE_PROVIDER = "fake"
FAKE_MODEL = "fake-finalizer-v1"

_INDICATORS = (
    "RSI",
    "MACD",
    "EMA",
    "SMA",
    "Bollinger",
    "Fibonacci",
    "Stochastic",
    "Ichimoku",
    "ATR",
    "ADX",
)


def read_prompt_article(request: FinalizeRequest) -> str:
    """Recover the original article from the rendered prompt."""
    return extract_fenced(request.prompt.user, request.prompt.nonce, ARTICLE_LABEL).strip()


def read_prompt_review(request: FinalizeRequest) -> dict[str, Any]:
    """Recover the review payload from the rendered prompt.

    Reading it back keeps the fake honest: it edits the article it was actually
    handed, against the issues it was actually given, so a plumbing mistake
    surfaces as a wrong revision rather than passing silently.
    """
    raw = extract_fenced(request.prompt.user, request.prompt.nonce, REVIEW_LABEL).strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {"issues": []}
    return payload if isinstance(payload, dict) else {"issues": []}


def _drop_sentences_mentioning(article: str, needle: str) -> str:
    """Remove any line that mentions *needle*, then tidy blank runs.

    A blunt instrument, and deliberately so: this is a test double, and removing
    the offending sentence is what a real editor would be asked to do for an
    unsupported claim.
    """
    kept = [
        line
        for line in article.splitlines()
        if needle.casefold() not in line.casefold() or not line.strip()
    ]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def apply_review(article: str, review: dict[str, Any]) -> tuple[str, list[IssueResolution]]:
    """Apply each issue's correction to *article*, as an editor would.

    Uses the issue's own evidence: a `DATA_MISMATCH` swaps the wrong value for
    the expected one; anything else drops the sentence carrying the offending
    token.
    """
    revised = article
    resolutions: list[IssueResolution] = []

    for issue in review.get("issues", []):
        issue_id = str(issue.get("issue_id", ""))
        evidence = issue.get("evidence") or {}
        actual = (evidence.get("actual") or "").strip()
        expected = (evidence.get("expected") or "").strip()
        severity = str(issue.get("severity", "LOW"))
        category = str(issue.get("category", "OTHER"))

        description = "No change was needed."
        resolution = ResolutionStatus.APPLIED

        if actual and expected and actual in revised and category == "DATA_MISMATCH":
            revised = revised.replace(actual, expected)
            description = f"Corrected {actual} to {expected}."
        elif actual and actual in revised:
            revised = _drop_sentences_mentioning(revised, actual)
            description = f"Removed the unsupported reference to {actual}."
        else:
            revised, description, resolution = _fallback_edit(revised, issue, severity)

        resolutions.append(
            IssueResolution(issue_id=issue_id, resolution=resolution, description=description)
        )

    return revised, resolutions


def _fallback_edit(
    article: str, issue: dict[str, Any], severity: str
) -> tuple[str, str, ResolutionStatus]:
    """Handle an issue whose evidence does not name a token to remove."""
    category = str(issue.get("category", "OTHER"))

    if category == "UNSUPPORTED_CLAIM":
        for name in _INDICATORS:
            if re.search(rf"\b{name}\d*\b", article, re.IGNORECASE):
                return (
                    _drop_sentences_mentioning(article, name),
                    f"Removed the unsupported {name} reference.",
                    ResolutionStatus.APPLIED,
                )

    if severity in {"HIGH", "CRITICAL"}:
        return (article, "Applied the correction the review requested.", ResolutionStatus.APPLIED)

    return (
        article,
        "Reviewed the article and found this does not apply to the current text.",
        ResolutionStatus.NOT_APPLICABLE,
    )


@dataclass
class FakeFinalizerClient:
    """Deterministic, offline implementation of :class:`FinalizerClient`.

    Configure at most one behaviour:

    * default - apply the review's corrections to the prompt's article;
    * ``raises`` - raise that error instead of answering;
    * ``output`` - return a specific :class:`FinalizerModelOutput`, for contract
      violations such as a missing resolution or a declined CRITICAL issue;
    * ``output_factory`` - compute the output from the request.
    """

    output: FinalizerModelOutput | None = None
    output_factory: Callable[[FinalizeRequest], FinalizerModelOutput] | None = None
    raises: FinalizeError | None = None
    usage: FinalizerUsage = field(
        default_factory=lambda: FinalizerUsage(input_tokens=3100, output_tokens=520)
    )
    model_name: str = FAKE_MODEL
    calls: list[FinalizeRequest] = field(default_factory=list)
    """Every request seen, so tests can assert on what was actually sent."""

    @property
    def provider(self) -> str:
        return FAKE_PROVIDER

    @property
    def model(self) -> str:
        return self.model_name

    def finalize(self, request: FinalizeRequest) -> FinalizeResponse:
        """Return the configured response, or an edit derived from the prompt."""
        self.calls.append(request)

        if self.raises is not None:
            raise self.raises

        if self.output is not None:
            output = self.output
        elif self.output_factory is not None:
            output = self.output_factory(request)
        else:
            output = self._build_default(request)

        return FinalizeResponse(
            output=output, model=self.model, provider=self.provider, usage=self.usage
        )

    def _build_default(self, request: FinalizeRequest) -> FinalizerModelOutput:
        article = read_prompt_article(request)
        review = read_prompt_review(request)
        revised, resolutions = apply_review(article, review)

        return FinalizerModelOutput(
            run_id=request.run_id,
            article=revised,
            issue_resolutions=resolutions,
            warnings=[],
        )


def lazy_client() -> FakeFinalizerClient:
    """A finalizer that reports every issue fixed but changes nothing.

    The most dangerous failure mode there is: a plausible account of work that
    never happened. Used to prove the postcheck catches it.
    """

    def build(request: FinalizeRequest) -> FinalizerModelOutput:
        review = read_prompt_review(request)
        return FinalizerModelOutput(
            run_id=request.run_id,
            article=read_prompt_article(request),
            issue_resolutions=[
                IssueResolution(
                    issue_id=str(issue.get("issue_id", "")),
                    resolution=ResolutionStatus.APPLIED,
                    description="Fixed as requested.",
                )
                for issue in review.get("issues", [])
            ],
        )

    return FakeFinalizerClient(output_factory=build)


def careless_client(addition: str) -> FakeFinalizerClient:
    """A finalizer that fixes what it was asked to and breaks something else.

    Models do this: told to remove an invented indicator, they remove it and
    add a different one in the same breath.
    """

    def build(request: FinalizeRequest) -> FinalizerModelOutput:
        article, resolutions = apply_review(
            read_prompt_article(request), read_prompt_review(request)
        )
        return FinalizerModelOutput(
            run_id=request.run_id,
            article=f"{article}\n\n{addition}",
            issue_resolutions=resolutions,
        )

    return FakeFinalizerClient(output_factory=build)


def failing_client(error: FinalizeError) -> FakeFinalizerClient:
    """A client that always raises *error*."""
    return FakeFinalizerClient(raises=error)


def timing_out_client(seconds: float = 120.0) -> FakeFinalizerClient:
    """A client that always times out."""
    return failing_client(
        FinalizeTimeoutError(
            f"provider did not respond within {seconds:g}s", timeout_seconds=seconds
        )
    )


def erroring_client(message: str = "provider returned HTTP 500") -> FakeFinalizerClient:
    """A client that always reports a provider failure."""
    return failing_client(FinalizeProviderError(message, status_code=500))


def malformed_client(message: str = "response was not valid JSON") -> FakeFinalizerClient:
    """A client that always reports an unparseable answer."""
    return failing_client(FinalizeResponseError(message))


__all__ = [
    "FAKE_MODEL",
    "FAKE_PROVIDER",
    "FakeFinalizerClient",
    "apply_review",
    "careless_client",
    "erroring_client",
    "failing_client",
    "lazy_client",
    "malformed_client",
    "read_prompt_article",
    "read_prompt_review",
    "timing_out_client",
]
