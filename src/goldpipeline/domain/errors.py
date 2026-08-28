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
    "PipelineError",
    "ReviewArtifactExistsError",
    "ReviewConfigurationError",
    "ReviewError",
    "ReviewProviderError",
    "ReviewResponseError",
    "ReviewTimeoutError",
    "RunAlreadyExistsError",
    "RunNotFinalizableError",
    "RunNotReadyError",
    "RunNotReviewableError",
    "StorageError",
    "SymbolMismatchError",
    "UnknownTimezoneError",
    "WriterArtifactExistsError",
    "WriterConfigurationError",
    "WriterError",
    "WriterProviderError",
    "WriterResponseError",
    "WriterTimeoutError",
]
