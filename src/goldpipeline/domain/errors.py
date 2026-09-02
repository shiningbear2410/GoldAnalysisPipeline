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


class ReviewSchemaError(ReviewError):
    """A deterministic precheck could not be recorded against its own schema.

    Raised when data pulled from the Run's own artifacts - a claim's resolved
    value, say - does not fit the constraints ``PrecheckFinding`` declares.
    This is local computation, not a provider or the network: it happens
    before ``review.start`` and no request is ever sent. Permanent rather than
    retried, for the same reason as :class:`ArtifactIntegrityError` - the next
    attempt reads the same bytes and fails the same way.

    Details never carry the rejected value itself, only where in the schema it
    was rejected and why.
    """

    code = "REVIEW_SCHEMA_ERROR"


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


# --- live market data (Round 8) ------------------------------------------


class MarketDataError(PipelineError):
    """Base class for failures of a live market data provider."""

    code = "MARKET_DATA_ERROR"


class MarketDataConfigurationError(MarketDataError):
    """Market data is misconfigured. Names the setting, never a credential."""

    code = "MARKET_DATA_CONFIGURATION_ERROR"


class Mt5NotInstalledError(MarketDataError):
    """The MetaTrader5 package is not importable in this interpreter.

    Kept distinct from a connection failure: one is fixed with ``pip``, the
    other by starting a terminal, and telling an operator the wrong one wastes
    an afternoon.
    """

    code = "MT5_NOT_INSTALLED"


class Mt5InitializeError(MarketDataError):
    """The terminal could not be reached. Carries the provider's own code only."""

    code = "MT5_INITIALIZE_FAILED"


class Mt5SymbolNotFoundError(MarketDataError):
    """The configured symbol does not exist on this broker.

    Never resolved by guessing a similar name. ``XAUUSD`` and ``XAUUSD.a`` can
    be different instruments with different spreads, and quietly substituting
    one for the other would publish prices nobody asked about.
    """

    code = "SYMBOL_NOT_FOUND"


class Mt5SymbolNotSelectedError(MarketDataError):
    """The symbol exists but could not be made visible in Market Watch."""

    code = "SYMBOL_NOT_SELECTED"


class InsufficientBarsError(MarketDataError):
    """The provider returned fewer candles than were asked for."""

    code = "INSUFFICIENT_BARS"


class StaleMarketDataError(MarketDataError):
    """The latest closed candle is too old to write about.

    Often means the market is simply closed rather than that anything is
    broken, so the message says so - but it still stops the Run, because an
    article quoting Friday's close on Sunday evening is wrong either way.
    """

    code = "STALE_MARKET_DATA"


class FormingCandleError(MarketDataError):
    """The newest bar has not closed yet.

    The one failure this provider exists to make impossible. A forming candle's
    high, low and close all still move, so quoting one produces an article that
    was false by the time anybody read it.
    """

    code = "FORMING_CANDLE_DETECTED"


class Mt5ProviderError(MarketDataError):
    """The provider failed in a way with no more specific meaning."""

    code = "MT5_PROVIDER_ERROR"


# --- ingestion (Round 8) -------------------------------------------------


class IngestionError(PipelineError):
    """Base class for failures of the analysis inbox."""

    code = "INGESTION_ERROR"


class InboxPayloadError(IngestionError):
    """An inbox payload is missing, malformed, or not a shape we accept."""

    code = "INBOX_PAYLOAD_INVALID"


class EventConflictError(IngestionError):
    """An ``event_id`` was reused for different content.

    Fails closed and never overwrites the original mapping. The alternative -
    accepting the newer payload - would silently detach an audit trail from the
    article that was actually published under it.
    """

    code = "EVENT_ID_CONFLICT"


class EventUnresolvedError(IngestionError):
    """A previous attempt at this event never finished, and left a reservation.

    Whether a Run exists for it is knowable but not from here; ``inbox-reconcile``
    answers it deterministically. Until then the event is not re-ingested.
    """

    code = "EVENT_UNRESOLVED"


class LedgerError(IngestionError):
    """The ingestion ledger could not be read or written."""

    code = "INGESTION_LEDGER_ERROR"


# --- news collection -----------------------------------------------------


class NewsError(PipelineError):
    """Base class for failures of the news collector."""

    code = "NEWS_ERROR"


class NewsFetchError(NewsError):
    """A public preview page could not be retrieved.

    Per source. One channel being unreachable says nothing about the others, so
    this is caught per channel and recorded rather than ending the collection.
    """

    code = "NEWS_FETCH_ERROR"


class NewsParseError(NewsError):
    """A page was retrieved but contains no recognisable message.

    Usually means the markup changed. Reported loudly rather than treated as an
    empty channel: "nothing published" and "we can no longer read this" look
    identical in an item count and are not the same problem.
    """

    code = "NEWS_PARSE_ERROR"


class NewsConfigurationError(NewsError):
    """The collector's settings are unusable - no sources, absurd bounds."""

    code = "NEWS_CONFIGURATION_ERROR"


# --- article types -------------------------------------------------------


class ArticleTypeNotReadyError(PipelineError):
    """A valid article type whose implementation does not exist yet.

    Raised *before* any provider is called, so a Run in an unfinished mode costs
    nothing and produces nothing. Deliberately not a fallback to ``ANALYSIS``:
    quietly writing a different kind of article than the one asked for is worse
    than refusing, because the refusal is visible and the substitution is not.

    Permanent by nature - the mode becomes available when code ships, not when a
    scheduler tries again.
    """

    code = "ARTICLE_TYPE_NOT_READY"


class RemoteArticleTypeNotAllowedError(PipelineError):
    """A remote producer offered an article type only local producers may use.

    The restriction lives at the transport boundary because that is the only
    layer that knows an event arrived over a network rather than from a local
    process. An event cannot be trusted to describe its own origin.
    """

    code = "REMOTE_ARTICLE_TYPE_NOT_ALLOWED"


# --- remote intake (optional upstream source) ----------------------------


class RemoteIntakeError(IngestionError):
    """Base class for failures of the optional remote event source.

    Separate from the inbox errors above because the blast radius is different.
    An unreadable *local* event is evidence about a Run that may already exist;
    an unreachable *remote* producer says nothing about local work at all. These
    are caught at the fetch step and recorded, never allowed to end a tick - a
    source that is switched off by default must not be able to stop the pipeline
    that ran fine without it.
    """

    code = "REMOTE_INTAKE_ERROR"


class RemoteIntakeConfigurationError(RemoteIntakeError):
    """Remote intake is on but its settings or credential are unusable.

    A human must fix the environment, so this is the one remote failure that is
    worth reporting loudly rather than retrying quickly.
    """

    code = "REMOTE_INTAKE_CONFIGURATION_ERROR"


class RemoteIntakeTransportError(RemoteIntakeError):
    """The remote producer could not be reached, or refused to answer.

    Unlike a publish, nothing is ambiguous here: a fetch that fails delivered
    nothing, changed nothing, and can be retried on the next tick at no cost.
    """

    code = "REMOTE_INTAKE_TRANSPORT_ERROR"


class RemoteIntakeResponseError(RemoteIntakeError):
    """The remote producer answered with something this pipeline will not read.

    Covers a body that is not JSON, an envelope of the wrong shape, a response
    larger than the cap, or more events than the cap allows. The payload is
    discarded whole; nothing partial is admitted.
    """

    code = "REMOTE_INTAKE_RESPONSE_ERROR"


# --- automation (Round 9) ------------------------------------------------


class AutomationError(PipelineError):
    """Base class for failures of the automation worker.

    The worker owns scheduling and nothing else: when to look, what to look at
    next, and when to stop. Every judgement about an article still belongs to
    the stage that made it, and reaches the worker as a verdict it must respect.
    """

    code = "AUTOMATION_ERROR"


class AutomationConfigurationError(AutomationError):
    """Automation is misconfigured. Names the setting, never its value."""

    code = "AUTOMATION_CONFIGURATION_ERROR"


class AutoPublishNotAllowedError(AutomationError):
    """Unattended publishing was asked for but is not authorised.

    Raised only where the guard is a hard refusal - an allowlist that is missing
    or does not match the configured destination. Per-Run refusals, such as a
    Run being older than the cutoff, are recorded as outcomes rather than
    raised: they stop one article, not the worker.
    """

    code = "AUTO_PUBLISH_NOT_ALLOWED"


class AutoPublishTargetMismatchError(AutoPublishNotAllowedError):
    """The allowlisted destination is not the configured one.

    Two settings have to name the same channel: enabling unattended publishing
    is one decision, and saying where it may publish is a second. Requiring both
    means a copied environment cannot silently point the pipeline at a channel
    nobody authorised.
    """

    code = "AUTO_PUBLISH_TARGET_MISMATCH"


# --- credentials (Round 9.1) ---------------------------------------------


class CredentialError(PipelineError):
    """Base class for failures of the secure credential store.

    Every message here names a setting, a service, or a backend - never a
    value, and never the underlying exception's text. A credential store's own
    errors can quote what they were handling.
    """

    code = "CREDENTIAL_ERROR"


class CredentialBackendUnavailableError(CredentialError):
    """No credential backend is installed or importable.

    Distinct from an insecure one: this is fixed with ``pip``, that is fixed by
    changing the machine's configuration, and telling an operator the wrong one
    wastes an afternoon.
    """

    code = "CREDENTIAL_BACKEND_UNAVAILABLE"


class InsecureCredentialBackendError(CredentialError):
    """The active backend does not store secrets securely.

    Fails closed rather than falling back. A store that keeps credentials in a
    plaintext file is worse than no store at all, because it looks like one.
    """

    code = "INSECURE_CREDENTIAL_BACKEND"


class CredentialNotFoundError(CredentialError):
    """The store has no entry under that name."""

    code = "CREDENTIAL_NOT_FOUND"


class CredentialReadError(CredentialError):
    """The store refused or failed to answer."""

    code = "CREDENTIAL_READ_FAILED"


class CredentialWriteError(CredentialError):
    """The store refused or failed to save an entry."""

    code = "CREDENTIAL_WRITE_FAILED"


class CredentialDeleteError(CredentialError):
    """The store refused or failed to remove an entry."""

    code = "CREDENTIAL_DELETE_FAILED"


# --- runtime configuration and task activation (Round 9.2) ---------------


class RuntimeConfigError(PipelineError):
    """The persisted non-secret configuration is unusable.

    Covers a corrupt file, an unknown setting name, and a value the pipeline
    would refuse at load time. All three fail closed rather than falling back to
    defaults: a scheduled task silently running on built-in defaults would fetch
    a symbol and target a channel nobody chose.
    """

    code = "RUNTIME_CONFIG_ERROR"


class SecretNotPersistableError(RuntimeConfigError):
    """Someone tried to persist a credential as configuration.

    Refused by name rather than merely absent from the schema. "Unknown setting"
    would be the wrong message: the person needs to be told where an API key
    actually goes, not that it is misspelt.
    """

    code = "SECRET_NOT_PERSISTABLE"


class ProductionConfigError(RuntimeConfigError):
    """The scheduled worker could not obtain a complete production configuration.

    Distinct from :class:`RuntimeConfigError` because the *response* differs. An
    operator at a keyboard may reasonably run on defaults; a scheduled task may
    not. Every subclass here means the same thing to the worker - stop, touch
    nothing, exit non-zero - and differs only in what a human is told to fix.

    The whole family exists because the alternative was observed in production:
    a missing file read as "no settings", filled in from built-in defaults, and
    reported as a healthy ``exit 0`` every minute for hours.
    """

    code = "PRODUCTION_CONFIG_ERROR"


class ConfigPathUnavailableError(ProductionConfigError):
    """The production configuration path could not be resolved at all.

    On Windows that means ``%LOCALAPPDATA%`` was absent from the environment.
    Deliberately not softened into "use ``~/.config`` instead": a second
    production path is a second place for the truth to hide.
    """

    code = "CONFIG_PATH_UNAVAILABLE"


class PersistentConfigNotFoundError(ProductionConfigError):
    """There is no configuration file where production expects one."""

    code = "PERSISTENT_CONFIG_NOT_FOUND"


class PersistentConfigUnreadableError(ProductionConfigError):
    """The configuration file exists but could not be read."""

    code = "PERSISTENT_CONFIG_UNREADABLE"


class PersistentConfigInvalidJsonError(ProductionConfigError):
    """The configuration file is not valid UTF-8 JSON."""

    code = "PERSISTENT_CONFIG_INVALID_JSON"


class PersistentConfigSchemaMismatchError(ProductionConfigError):
    """The configuration file declares a schema this build does not accept."""

    code = "PERSISTENT_CONFIG_SCHEMA_MISMATCH"


class PersistentConfigIncompleteError(ProductionConfigError):
    """A required production setting is absent.

    Absent, not merely falsy. "The operator switched automation off" and "the
    configuration disappeared" are different states of the world, and a
    scheduled worker that cannot tell them apart will report the second as the
    first for as long as nobody looks.
    """

    code = "PERSISTENT_CONFIG_INCOMPLETE"


class PersistentConfigUnknownKeyError(ProductionConfigError):
    """The configuration file holds a setting this build does not know.

    Refused rather than ignored: an unknown key is usually a misspelt known one,
    and silently dropping it means the setting the operator believed they had
    configured is quietly at its default.
    """

    code = "PERSISTENT_CONFIG_UNKNOWN_KEY"


class PersistentConfigSecretKeyError(ProductionConfigError):
    """A credential name appeared in the configuration file."""

    code = "PERSISTENT_CONFIG_SECRET_KEY_FORBIDDEN"


class TaskSchedulerError(PipelineError):
    """A Windows Task Scheduler operation failed."""

    code = "TASK_SCHEDULER_ERROR"


class TaskSchedulerUnavailableError(TaskSchedulerError):
    """The Task Scheduler could not be reached at all."""

    code = "TASK_SCHEDULER_UNAVAILABLE"


class TaskDefinitionMismatchError(TaskSchedulerError):
    """A task with this name exists but is not the one we intend.

    Left untouched. It might be an older definition from a previous round or
    something a person created by hand, and overwriting either on the way past
    is the kind of helpfulness that loses work.
    """

    code = "TASK_DEFINITION_MISMATCH"


__all__ = [
    "AnalysisTextTooLargeError",
    "ArticleTypeNotReadyError",
    "ArtifactAlreadyExistsError",
    "ArtifactIntegrityError",
    "AutoPublishNotAllowedError",
    "AutoPublishTargetMismatchError",
    "AutomationConfigurationError",
    "AutomationError",
    "ConfigPathUnavailableError",
    "ContextIntegrityError",
    "CredentialBackendUnavailableError",
    "CredentialDeleteError",
    "CredentialError",
    "CredentialNotFoundError",
    "CredentialReadError",
    "CredentialWriteError",
    "DuplicateTimestampError",
    "EmptyAnalysisTextError",
    "EmptyBarsError",
    "EventConflictError",
    "EventUnresolvedError",
    "FinalizationBlockedError",
    "FinalizeArtifactExistsError",
    "FinalizeConfigurationError",
    "FinalizeError",
    "FinalizePostcheckError",
    "FinalizeProviderError",
    "FinalizeResponseError",
    "FinalizeTimeoutError",
    "FormingCandleError",
    "InboxPayloadError",
    "IngestionError",
    "InputValidationError",
    "InsecureCredentialBackendError",
    "InsufficientBarsError",
    "InvalidBarError",
    "LatestBarMismatchError",
    "LedgerError",
    "MarketDataConfigurationError",
    "MarketDataError",
    "Mt5InitializeError",
    "Mt5NotInstalledError",
    "Mt5ProviderError",
    "Mt5SymbolNotFoundError",
    "Mt5SymbolNotSelectedError",
    "NaiveTimestampError",
    "NewsConfigurationError",
    "NewsError",
    "NewsFetchError",
    "NewsParseError",
    "NormalizationError",
    "OrchestrationError",
    "PersistentConfigIncompleteError",
    "PersistentConfigInvalidJsonError",
    "PersistentConfigNotFoundError",
    "PersistentConfigSchemaMismatchError",
    "PersistentConfigSecretKeyError",
    "PersistentConfigUnknownKeyError",
    "PersistentConfigUnreadableError",
    "PipelineError",
    "ProductionConfigError",
    "PublishDecisionExistsError",
    "PublishGateError",
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
    "ReviewArtifactExistsError",
    "ReviewConfigurationError",
    "ReviewError",
    "ReviewProviderError",
    "ReviewResponseError",
    "ReviewSchemaError",
    "ReviewTimeoutError",
    "RunAlreadyExistsError",
    "RunLockedError",
    "RunNotFinalizableError",
    "RunNotGateableError",
    "RemoteArticleTypeNotAllowedError",
    "RemoteIntakeConfigurationError",
    "RemoteIntakeError",
    "RemoteIntakeResponseError",
    "RemoteIntakeTransportError",
    "RunNotReadyError",
    "RunNotResumableError",
    "RunNotReviewableError",
    "RuntimeConfigError",
    "SecretNotPersistableError",
    "StaleMarketDataError",
    "StorageError",
    "SymbolMismatchError",
    "TaskDefinitionMismatchError",
    "TaskSchedulerError",
    "TaskSchedulerUnavailableError",
    "UnknownTimezoneError",
    "UntrustworthyRunError",
    "WriterArtifactExistsError",
    "WriterConfigurationError",
    "WriterError",
    "WriterProviderError",
    "WriterResponseError",
    "WriterTimeoutError",
]
