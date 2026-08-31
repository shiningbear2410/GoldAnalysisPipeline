"""Error taxonomy for the pipeline.

Two distinct concepts, deliberately not merged:

* :class:`PipelineError` - a *validation error*. The data is factually wrong or
  self-contradictory; the Run must not reach ``NORMALIZED``.
* A quality *warning* (see :mod:`goldpipeline.schemas.quality`) - the data is
  usable but degraded. It is recorded in ``context.data_quality`` and the Run
  still completes.
"""

from __future__ import annotations

from typing import Any


class PipelineError(Exception):
    """Base class for every deterministic failure raised by the pipeline."""

    code: str = "PIPELINE_ERROR"

    def __init__(self, message: str, /, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details

    def to_dict(self) -> dict[str, Any]:
        """Serializable form, safe to store in a manifest."""
        return {"code": self.code, "message": self.message, "details": self.details}

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"[{self.code}] {self.message}"


# --- input / schema level -------------------------------------------------


class InputValidationError(PipelineError):
    """A source payload does not satisfy its schema."""

    code = "INPUT_VALIDATION_ERROR"


class EmptyAnalysisTextError(PipelineError):
    """The raw analysis text is missing or blank after sanitisation."""

    code = "EMPTY_ANALYSIS_TEXT"


class AnalysisTextTooLargeError(PipelineError):
    """The raw analysis text exceeds the hard size limit."""

    code = "ANALYSIS_TEXT_TOO_LARGE"


# --- market data normalization -------------------------------------------


class NormalizationError(PipelineError):
    """Base class for failures raised while normalizing market data."""

    code = "NORMALIZATION_ERROR"


class EmptyBarsError(NormalizationError):
    """A market data payload contains no bars."""

    code = "EMPTY_BARS"


class DuplicateTimestampError(NormalizationError):
    """Two or more bars share the same timestamp.

    Policy: duplicates are a hard failure. They are never silently de-duplicated
    because the pipeline cannot know which bar is authoritative.
    """

    code = "DUPLICATE_TIMESTAMP"


class NaiveTimestampError(NormalizationError):
    """A naive timestamp was supplied without a declaring source timezone."""

    code = "NAIVE_TIMESTAMP"


class UnknownTimezoneError(NormalizationError):
    """The declared source timezone cannot be resolved."""

    code = "UNKNOWN_TIMEZONE"


class SymbolMismatchError(NormalizationError):
    """Two sources disagree about which instrument the data describes."""

    code = "SYMBOL_MISMATCH"


class LatestBarMismatchError(NormalizationError):
    """A supplied ``latest_bar`` disagrees with the last normalized bar."""

    code = "LATEST_BAR_MISMATCH"


class InvalidBarError(NormalizationError):
    """A bar violates the OHLC ordering invariants."""

    code = "INVALID_BAR"


# --- storage --------------------------------------------------------------


class StorageError(PipelineError):
    """Base class for filesystem-level failures."""

    code = "STORAGE_ERROR"


class RunAlreadyExistsError(StorageError):
    """A Run directory with the generated id already exists."""

    code = "RUN_ALREADY_EXISTS"


class ArtifactAlreadyExistsError(StorageError):
    """An immutable artifact was written twice within the same Run."""

    code = "ARTIFACT_ALREADY_EXISTS"


# --- writer stage (Round 2) ----------------------------------------------


class WriterError(PipelineError):
    """Base class for every failure of the Claude Writer stage."""

    code = "WRITER_ERROR"


class WriterConfigurationError(WriterError):
    """The writer is not configured well enough to run.

    Missing credentials, unusable model id, nonsensical timeout. Never carries
    the credential itself - only the name of the setting that is wrong.
    """

    code = "WRITER_CONFIGURATION_ERROR"


class WriterProviderError(WriterError):
    """The provider rejected the request or failed to serve it.

    Message text is deliberately summarised: provider payloads can echo request
    content back, and this message travels into the manifest.
    """

    code = "WRITER_PROVIDER_ERROR"


class WriterTimeoutError(WriterError):
    """The provider did not respond within the configured timeout."""

    code = "WRITER_TIMEOUT"


class WriterResponseError(WriterError):
    """The provider answered, but the answer does not satisfy the contract.

    Malformed JSON, missing fields, an empty article, a mismatched ``run_id``.
    No draft is written when this is raised.
    """

    code = "WRITER_RESPONSE_ERROR"


class WriterArtifactExistsError(WriterError):
    """The Run already has writer artifacts.

    Runs are immutable; a second write is refused rather than silently
    producing a different article under the same run id.
    """

    code = "WRITER_ARTIFACT_EXISTS"


class RunNotReadyError(WriterError):
    """The Run is missing a context, or is not in a state the writer accepts."""

    code = "RUN_NOT_READY"


class ContextIntegrityError(WriterError):
    """``context.json`` does not match the digest recorded in the manifest.

    The Run's inputs are supposed to be immutable. If the file on disk has
    changed, the writer refuses to work from it rather than producing an
    article whose provenance cannot be trusted.
    """

    code = "CONTEXT_INTEGRITY_ERROR"


# --- shared artifact integrity -------------------------------------------


class ArtifactIntegrityError(WriterError):
    """An artifact's bytes no longer match the digest recorded for it.

    Subclasses :class:`WriterError` so existing writer-stage handling keeps
    working; the reviewer stage treats it the same way. Either way the meaning
    is identical: a Run's supposedly immutable files have changed, so nothing
    downstream may be built on them.
    """

    code = "ARTIFACT_INTEGRITY_ERROR"


# --- reviewer stage (Round 3) --------------------------------------------


class ReviewError(PipelineError):
    """Base class for every failure of the ChatGPT Reviewer stage."""

    code = "REVIEW_ERROR"


class ReviewConfigurationError(ReviewError):
    """The reviewer is not configured well enough to run.

    Names the setting that is wrong; never its value.
    """

    code = "REVIEW_CONFIGURATION_ERROR"


class ReviewProviderError(ReviewError):
    """The provider rejected the request or failed to serve it.

    Explicitly *not* a review verdict. A network failure says nothing about the
    article, so it must never be recorded as a REJECT.
    """

    code = "REVIEW_PROVIDER_ERROR"


class ReviewTimeoutError(ReviewError):
    """The provider did not respond within the configured timeout."""

    code = "REVIEW_TIMEOUT"


class ReviewResponseError(ReviewError):
    """The provider answered, but the answer cannot be trusted as a review.

    Covers malformed output, a mismatched ``run_id``, and verdicts that
    contradict their own evidence - a PASS listing a HIGH issue, a
    NEEDS_REVISION with nothing to revise. No review artifact is written.
    """

    code = "REVIEW_RESPONSE_ERROR"


class ReviewArtifactExistsError(ReviewError):
    """The Run already has a review.

    Runs are immutable; a second review is refused rather than silently
    replacing a verdict someone may already have acted on.
    """

    code = "REVIEW_ARTIFACT_EXISTS"


class RunNotReviewableError(ReviewError):
    """The Run is not in a state the reviewer accepts."""

    code = "RUN_NOT_REVIEWABLE"


# --- finalizer stage (Round 4) -------------------------------------------


class FinalizeError(PipelineError):
    """Base class for every failure of the Claude Finalizer stage."""

    code = "FINALIZE_ERROR"


class FinalizationBlockedError(FinalizeError):
    """The review verdict forbids finalization.

    Raised for a ``REJECT``. Deliberately not a provider or response failure:
    nothing went wrong technically, the pipeline is refusing to auto-correct an
    article a reviewer judged unsalvageable. No provider is contacted and no
    artifact is written; the Run waits for a human.
    """

    code = "FINALIZATION_BLOCKED"


class FinalizeConfigurationError(FinalizeError):
    """The finalizer is not configured well enough to run.

    Only ever raised on the revision path - a passthrough finalization needs no
    credentials at all.
    """

    code = "FINALIZE_CONFIGURATION_ERROR"


class FinalizeProviderError(FinalizeError):
    """The provider rejected the request or failed to serve it."""

    code = "FINALIZE_PROVIDER_ERROR"


class FinalizeTimeoutError(FinalizeError):
    """The provider did not respond within the configured timeout."""

    code = "FINALIZE_TIMEOUT"


class FinalizeResponseError(FinalizeError):
    """The provider answered, but the answer cannot be trusted as a revision.

    Covers malformed output, a mismatched ``run_id``, resolutions that do not
    account for every issue, and a HIGH or CRITICAL issue the model tried to
    wave away. No final artifact is written.
    """

    code = "FINALIZE_RESPONSE_ERROR"


class FinalizePostcheckError(FinalizeError):
    """The revised article failed the deterministic checks run after the fact.

    The model returned something well-formed that is nonetheless worse: a fact
    it invented, a flaw it was asked to fix and did not, or a new HIGH finding
    the original did not have. Failing here is the point - a plausible revision
    that reintroduces an error is more dangerous than no revision at all.
    """

    code = "FINALIZE_POSTCHECK_ERROR"


class FinalizeArtifactExistsError(FinalizeError):
    """The Run already has finalizer artifacts."""

    code = "FINALIZE_ARTIFACT_EXISTS"


class RunNotFinalizableError(FinalizeError):
    """The Run is not in a state the finalizer accepts."""

    code = "RUN_NOT_FINALIZABLE"


# --- publish gate (Round 5) ----------------------------------------------


class PublishGateError(PipelineError):
    """Base class for failures of the deterministic publish gate.

    Note what is *not* here: a blocked article. A ``BLOCKED`` decision is the
    gate working, not the gate failing, so it is an artifact rather than an
    exception. These errors are for the cases where no trustworthy decision can
    be reached at all.
    """

    code = "PUBLISH_GATE_ERROR"


class PublishDecisionExistsError(PublishGateError):
    """The Run already has a publish decision.

    Decisions are immutable. Re-evaluating a Run under a newer gate would mean
    an approval could appear where a block used to be, with nothing in the
    artifact chain recording that it changed.
    """

    code = "PUBLISH_DECISION_EXISTS"


class RunNotGateableError(PublishGateError):
    """The Run is not in a state the gate accepts."""

    code = "RUN_NOT_GATEABLE"


class UntrustworthyRunError(PublishGateError):
    """The Run is too damaged to decide about.

    Distinct from an integrity *finding*: a tampered artifact still yields a
    ``BLOCKED`` decision, because the Run is identifiable and the block is
    meaningful. This is for a manifest that cannot be parsed at all - there is
    no trustworthy identity to attach a decision to.
    """

    code = "UNTRUSTWORTHY_RUN"


# --- publisher (Round 6) -------------------------------------------------


class PublisherError(PipelineError):
    """Base class for failures of the Telegram publisher."""

    code = "PUBLISHER_ERROR"


class PublisherConfigurationError(PublisherError):
    """Publishing is not configured. Names the setting, never its value."""

    code = "PUBLISHER_CONFIGURATION_ERROR"


class PublisherNotApprovedError(PublisherError):
    """The Run is not cleared for publication.

    Either it never reached ``READY_TO_PUBLISH`` or its decision is not
    ``APPROVED``. No intent is written and no request is made.
    """

    code = "PUBLISHER_NOT_APPROVED"


class PublisherIntegrityError(PublisherError):
    """An artifact changed after the gate approved it.

    The approval describes exact bytes. If those bytes moved, the approval no
    longer covers what would be sent, so nothing is sent.
    """

    code = "PUBLISHER_INTEGRITY_ERROR"


class PublisherArtifactExistsError(PublisherError):
    """The Run has already been attempted.

    One attempt per Run in V1. A second attempt cannot know what the first one
    delivered, so retrying is how duplicates get published.
    """

    code = "PUBLISHER_ARTIFACT_EXISTS"


class PublisherPreviousAttemptUncertainError(PublisherError):
    """A previous attempt left an intent with no result.

    The process died somewhere around the network call, so Telegram may or may
    not have the message. Resending is exactly the wrong move.
    """

    code = "PUBLISHER_PREVIOUS_ATTEMPT_UNCERTAIN"


class PublisherAuthenticationError(PublisherError):
    """The provider rejected the credentials. Never carries the token."""

    code = "PUBLISHER_AUTHENTICATION_ERROR"


class PublisherPermissionError(PublisherError):
    """The bot may not post to the configured target."""

    code = "PUBLISHER_PERMISSION_ERROR"


class PublisherRejectedError(PublisherError):
    """The provider explicitly refused the request.

    An explicit refusal is *good* news relative to silence: it means the message
    was not delivered, so the outcome is knowable.
    """

    code = "PUBLISHER_REJECTED"


class PublisherRateLimitError(PublisherError):
    """Flood control. The only condition this pipeline retries.

    A 429 is the provider stating plainly that it did not accept the request and
    saying when to try again - the one case where a retry cannot duplicate.
    """

    code = "PUBLISHER_RATE_LIMITED"


class PublisherTransportAmbiguousError(PublisherError):
    """The outcome cannot be determined.

    A timeout, a reset connection, a 5xx, a reply that does not parse. The
    request may have been delivered and the acknowledgement lost. This never
    triggers a retry - it ends the attempt as ``UNCERTAIN`` for a human.
    """

    code = "PUBLISHER_TRANSPORT_AMBIGUOUS"


class PublisherResponseError(PublisherError):
    """The provider answered, but not in a way that confirms delivery."""

    code = "PUBLISHER_RESPONSE_ERROR"


# --- orchestration (Round 7) ---------------------------------------------


class OrchestrationError(PipelineError):
    """Base class for failures of the end-to-end orchestrator.

    Deliberately narrow. The orchestrator owns sequencing and concurrency and
    nothing else, so the only failures it can originate are about *those*:
    a Run it must not resume, and a Run someone else is already driving.
    Everything a stage decides keeps that stage's own error type.
    """

    code = "ORCHESTRATION_ERROR"


class RunLockedError(OrchestrationError):
    """Another process holds this Run's lock.

    Carries whatever the lock file said about its holder - pid, host, when it
    was taken - because deciding whether that process is still alive is a
    judgement for a human. A stale lock is never removed automatically: doing so
    would turn a crashed publisher into a duplicated article.
    """

    code = "RUN_LOCKED"


class RunNotResumableError(OrchestrationError):
    """The Run is in a state the orchestrator must not drive forward.

    Every publish-side state except ``READY_TO_PUBLISH`` lands here. The
    dangerous one is ``PUBLISH_UNCERTAIN``: Telegram may already hold the
    article, so the safe move is to stop and let a human reconcile.
    """

    code = "RUN_NOT_RESUMABLE"


__all__ = [
    "AnalysisTextTooLargeError",
    "ArtifactAlreadyExistsError",
    "ArtifactIntegrityError",
    "ContextIntegrityError",
    "DuplicateTimestampError",
    "EmptyAnalysisTextError",
    "EmptyBarsError",
    "FinalizationBlockedError",
    "FinalizeArtifactExistsError",
    "FinalizeConfigurationError",
    "FinalizeError",
    "FinalizePostcheckError",
    "FinalizeProviderError",
    "FinalizeResponseError",
    "FinalizeTimeoutError",
    "InputValidationError",
    "InvalidBarError",
    "LatestBarMismatchError",
    "NaiveTimestampError",
    "NormalizationError",
    "OrchestrationError",
    "PipelineError",
    "PublisherArtifactExistsError",
    "PublisherAuthenticationError",
    "PublisherConfigurationError",
    "PublisherError",
    "PublisherIntegrityError",
    "PublisherNotApprovedError",
    "PublisherPermissionError",
    "PublisherPreviousAttemptUncertainError",
    "PublisherRateLimitError",
    "PublisherRejectedError",
    "PublisherResponseError",
    "PublisherTransportAmbiguousError",
    "PublishDecisionExistsError",
    "PublishGateError",
    "ReviewArtifactExistsError",
    "ReviewConfigurationError",
    "ReviewError",
    "ReviewProviderError",
    "ReviewResponseError",
    "ReviewTimeoutError",
    "RunAlreadyExistsError",
    "RunLockedError",
    "RunNotFinalizableError",
    "RunNotGateableError",
    "RunNotReadyError",
    "RunNotResumableError",
    "RunNotReviewableError",
    "StorageError",
    "SymbolMismatchError",
    "UnknownTimezoneError",
    "UntrustworthyRunError",
    "WriterArtifactExistsError",
    "WriterConfigurationError",
    "WriterError",
    "WriterProviderError",
    "WriterResponseError",
    "WriterTimeoutError",
]
