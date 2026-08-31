"""Command-line entry point.

    python -m goldpipeline create-run --telegram fixtures/telegram_sample.json \\
                                      --ohlc fixtures/ohlc_sample.json

argparse from the stdlib is enough here; a CLI framework would be a dependency
without a job.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from goldpipeline import PIPELINE_VERSION
from goldpipeline.adapters.fake_finalizer import FakeFinalizerClient
from goldpipeline.adapters.fake_publisher import FakePublisherClient
from goldpipeline.adapters.fake_reviewer import FakeReviewerClient
from goldpipeline.adapters.fake_writer import FakeWriterClient
from goldpipeline.adapters.file_source import JsonFileAnalysisSource, JsonFileMarketDataSource
from goldpipeline.adapters.finalizer_client import FinalizerClient, LazyFinalizerClient
from goldpipeline.adapters.publisher_client import PublisherClient
from goldpipeline.adapters.reviewer_client import ReviewerClient
from goldpipeline.adapters.writer_client import WriterClient
from goldpipeline.config import (
    FinalizerSettings,
    ReviewerSettings,
    TelegramSettings,
    WriterSettings,
)
from goldpipeline.domain.errors import (
    FinalizationBlockedError,
    FinalizeConfigurationError,
    PipelineError,
    PublisherConfigurationError,
    ReviewConfigurationError,
    RunLockedError,
    WriterConfigurationError,
)
from goldpipeline.logging_setup import configure_logging
from goldpipeline.schemas.orchestration import PipelineMode, PipelineStatus
from goldpipeline.schemas.quality import DataQuality
from goldpipeline.services.finalizer import finalize_run
from goldpipeline.services.orchestrator import (
    PipelineClients,
    PipelineRunResult,
    resume_pipeline,
    run_pipeline,
)
from goldpipeline.services.pipeline import create_run, validate_sources
from goldpipeline.services.publish_gate import gate_publish
from goldpipeline.services.publisher import publish_run
from goldpipeline.services.reviewer import review_draft
from goldpipeline.services.writer import write_draft
from goldpipeline.storage.run_store import RunStore

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_INVALID_DATA = 2
EXIT_BLOCKED = 3
"""A gate declined. Nothing went wrong; retrying will not help.

Used by both `finalize` (the review rejected the article) and `gate-publish`
(the article is not safe to publish). One code for one concept: a caller can
always tell *which* gate spoke from the command it ran, so a second number for
the same meaning would only invite mistakes.
"""

DEFAULT_RUNS_DIR = Path("runs")

DEFAULT_PIPELINE_MODE = "ready-for-publish"
"""What `pipeline-run` does when nobody says otherwise.

Every check runs, including the publish gate, and nothing is sent. Publishing is
a separate decision, and it stays that way even when the pipeline is driven by
one command.
"""


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="goldpipeline",
        description="XAUUSD analysis pipeline - Round 1 (normalize + immutable runs).",
    )
    parser.add_argument("--version", action="version", version=f"goldpipeline {PIPELINE_VERSION}")
    parser.add_argument(
        "--log-level",
        default=None,
        help="Logging level (DEBUG, INFO, WARNING, ERROR). Default: INFO.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser(
        "create-run",
        help="Normalize an analysis + OHLC pair into a new immutable Run.",
    )
    create.add_argument(
        "--telegram",
        required=True,
        type=Path,
        metavar="PATH",
        help="JSON file holding the raw analysis message.",
    )
    create.add_argument(
        "--ohlc",
        required=True,
        type=Path,
        metavar="PATH",
        help="JSON file holding the OHLC payload.",
    )
    create.add_argument(
        "--symbol",
        default=None,
        help="Instrument you expect, e.g. XAUUSD. A mismatch fails the Run.",
    )
    create.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"Root directory for Runs. Default: {DEFAULT_RUNS_DIR}.",
    )
    create.add_argument(
        "--now",
        type=_parse_now,
        default=None,
        metavar="ISO8601",
        help=(
            "Treat this instant as 'now', e.g. 2026-08-28T02:20:12Z. Must carry an "
            "explicit offset. Use it to re-create a Run over historical data without "
            "tripping the recency checks. Defaults to the current UTC time."
        ),
    )
    create.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the inputs and report, without creating a Run on disk.",
    )
    create.add_argument("--json", action="store_true", help="Emit the result as JSON.")

    draft = subparsers.add_parser(
        "write-draft",
        help="Run the Claude Writer over a normalized Run and store the draft.",
    )
    draft.add_argument("--run-id", required=True, help="Run to write for.")
    draft.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"Root directory for Runs. Default: {DEFAULT_RUNS_DIR}.",
    )
    draft.add_argument(
        "--model",
        default=None,
        help="Model id, overriding ANTHROPIC_MODEL. Ignored with --fake-writer.",
    )
    draft.add_argument(
        "--fake-writer",
        action="store_true",
        help="Use the offline deterministic writer. No API call, no cost.",
    )
    draft.add_argument("--json", action="store_true", help="Emit the result as JSON.")

    review = subparsers.add_parser(
        "review-draft",
        help="Audit a drafted Run with the ChatGPT Reviewer and store the verdict.",
    )
    review.add_argument("--run-id", required=True, help="Run to review.")
    review.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"Root directory for Runs. Default: {DEFAULT_RUNS_DIR}.",
    )
    review.add_argument(
        "--model",
        default=None,
        help="Model id, overriding OPENAI_REVIEW_MODEL. Ignored with --fake-reviewer.",
    )
    review.add_argument(
        "--fake-reviewer",
        action="store_true",
        help="Use the offline deterministic reviewer. No API call, no cost.",
    )
    review.add_argument("--json", action="store_true", help="Emit the result as JSON.")

    final = subparsers.add_parser(
        "finalize",
        help="Apply the review to a drafted Run and store the final article.",
    )
    final.add_argument("--run-id", required=True, help="Run to finalize.")
    final.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"Root directory for Runs. Default: {DEFAULT_RUNS_DIR}.",
    )
    final.add_argument(
        "--model",
        default=None,
        help=(
            "Model id, overriding ANTHROPIC_FINALIZER_MODEL. Only used when the "
            "review asked for revisions."
        ),
    )
    final.add_argument(
        "--fake-finalizer",
        action="store_true",
        help="Use the offline deterministic finalizer. No API call, no cost.",
    )
    final.add_argument("--json", action="store_true", help="Emit the result as JSON.")

    gate = subparsers.add_parser(
        "gate-publish",
        help="Decide deterministically whether a finalized Run may be published.",
    )
    gate.add_argument("--run-id", required=True, help="Run to gate.")
    gate.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"Root directory for Runs. Default: {DEFAULT_RUNS_DIR}.",
    )
    gate.add_argument("--json", action="store_true", help="Emit the result as JSON.")

    publish = subparsers.add_parser(
        "publish",
        help="Send an approved Run to Telegram. One attempt per Run.",
    )
    publish.add_argument("--run-id", required=True, help="Run to publish.")
    publish.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"Root directory for Runs. Default: {DEFAULT_RUNS_DIR}.",
    )
    publish.add_argument(
        "--fake-publisher",
        action="store_true",
        help="Use the offline publisher. Nothing is sent anywhere.",
    )
    publish.add_argument("--json", action="store_true", help="Emit the result as JSON.")

    pipeline_run = subparsers.add_parser(
        "pipeline-run",
        help="Create a Run from JSON inputs and drive it through the whole pipeline.",
    )
    pipeline_run.add_argument(
        "--telegram",
        required=True,
        type=Path,
        metavar="PATH",
        help="JSON file holding the raw analysis message.",
    )
    pipeline_run.add_argument(
        "--ohlc",
        required=True,
        type=Path,
        metavar="PATH",
        help="JSON file holding the OHLC payload.",
    )
    pipeline_run.add_argument(
        "--symbol",
        default=None,
        help="Instrument you expect, e.g. XAUUSD. A mismatch fails the Run.",
    )
    pipeline_run.add_argument(
        "--now",
        type=_parse_now,
        default=None,
        metavar="ISO8601",
        help="Treat this instant as 'now' in every stage. Must carry an explicit offset.",
    )
    _add_pipeline_arguments(pipeline_run)

    pipeline_resume = subparsers.add_parser(
        "pipeline-resume",
        help="Continue an existing Run from whichever stage it is due for.",
    )
    pipeline_resume.add_argument("--run-id", required=True, help="Run to continue.")
    pipeline_resume.add_argument(
        "--now",
        type=_parse_now,
        default=None,
        metavar="ISO8601",
        help="Treat this instant as 'now' in every stage. Must carry an explicit offset.",
    )
    _add_pipeline_arguments(pipeline_resume)

    show = subparsers.add_parser("show-run", help="Print a summary of an existing Run.")
    show.add_argument("run_id", help="Run identifier, e.g. 20260828_022701_a83f2c.")
    show.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)

    subparsers.add_parser("list-runs", help="List Run ids under the runs directory.").add_argument(
        "--runs-dir", type=Path, default=DEFAULT_RUNS_DIR
    )

    return parser


def _add_pipeline_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach the flags shared by ``pipeline-run`` and ``pipeline-resume``.

    Note what is *not* here: there is no ``--all``, and no single flag that both
    runs the pipeline and sends the result. Reaching Telegram takes two flags
    that each say so.
    """
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"Root directory for Runs. Default: {DEFAULT_RUNS_DIR}.",
    )
    parser.add_argument(
        "--mode",
        choices=[mode.value.lower().replace("_", "-") for mode in PipelineMode],
        default=DEFAULT_PIPELINE_MODE,
        help=(
            "How far to go: generate-only stops at FINALIZED, ready-for-publish "
            "stops at the gate's verdict, publish continues into the publisher. "
            f"Default: {DEFAULT_PIPELINE_MODE}."
        ),
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Shorthand for --mode publish. Nothing is sent without it.",
    )
    parser.add_argument(
        "--confirm-real-publish",
        action="store_true",
        help=(
            "Required alongside --publish when the transport is real. Two "
            "deliberate flags, because a published article cannot be unpublished."
        ),
    )
    parser.add_argument(
        "--fake-ai",
        action="store_true",
        help="Use the offline writer, reviewer and finalizer. No API calls, no cost.",
    )
    parser.add_argument("--fake-writer", action="store_true", help="Offline writer only.")
    parser.add_argument("--fake-reviewer", action="store_true", help="Offline reviewer only.")
    parser.add_argument("--fake-finalizer", action="store_true", help="Offline finalizer only.")
    parser.add_argument(
        "--fake-publisher",
        action="store_true",
        help="Offline publisher. Nothing is sent anywhere, even in publish mode.",
    )
    parser.add_argument("--writer-model", default=None, help="Overrides ANTHROPIC_MODEL.")
    parser.add_argument("--reviewer-model", default=None, help="Overrides OPENAI_REVIEW_MODEL.")
    parser.add_argument(
        "--finalizer-model", default=None, help="Overrides ANTHROPIC_FINALIZER_MODEL."
    )
    parser.add_argument("--json", action="store_true", help="Emit the execution result as JSON.")


def _parse_now(value: str) -> datetime:
    """Parse an ISO-8601 instant for ``--now``, rejecting naive input."""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a valid ISO-8601 datetime: {value!r}") from exc
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError(f"--now needs an explicit timezone offset, got {value!r}")
    return parsed


def _cmd_create_run(args: argparse.Namespace) -> int:
    analysis_source = JsonFileAnalysisSource(args.telegram)
    market_source = JsonFileMarketDataSource(args.ohlc)

    if args.dry_run:
        try:
            context = validate_sources(
                analysis_source=analysis_source,
                market_source=market_source,
                expected_symbol=args.symbol,
                now=args.now,
            )
        except PipelineError as exc:
            return _report_failure(exc, as_json=args.json)

        quality = context.data_quality
        if args.json:
            print(
                json.dumps(
                    {
                        "dry_run": True,
                        "valid": True,
                        "symbol": context.market.symbol,
                        "bar_count": quality.bar_count,
                        "quality_status": quality.status,
                        "warnings": [w.code for w in quality.warnings],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print("Dry run: inputs are valid, no Run was created.")
            print(f"Symbol: {context.market.symbol} {context.market.timeframe}")
            print(f"Bars: {quality.bar_count}")
            print(f"Data quality: {quality.status}")
            _print_warnings(quality)
        return EXIT_OK

    result = create_run(
        analysis_source=analysis_source,
        market_source=market_source,
        store=RunStore(args.runs_dir),
        expected_symbol=args.symbol,
        now=args.now,
    )

    if not result.succeeded:
        assert result.error is not None
        if args.json:
            print(
                json.dumps(
                    {
                        "run_id": result.run_id,
                        "status": result.status,
                        "run_dir": str(result.run_dir),
                        "error": result.error.to_dict(),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(f"Run failed: {result.run_id}", file=sys.stderr)
            print(f"Status: {result.status}", file=sys.stderr)
            print(f"Directory: {result.run_dir}", file=sys.stderr)
            print(f"Error: {result.error}", file=sys.stderr)
            if result.error.details:
                print(
                    f"Details: {json.dumps(result.error.details, ensure_ascii=False)}",
                    file=sys.stderr,
                )
        return EXIT_INVALID_DATA

    assert result.context is not None
    quality = result.context.data_quality
    if args.json:
        print(
            json.dumps(
                {
                    "run_id": result.run_id,
                    "status": result.status,
                    "run_dir": str(result.run_dir),
                    "context": str(result.context_path),
                    "bar_count": quality.bar_count,
                    "quality_status": quality.status,
                    "warnings": [w.code for w in quality.warnings],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(f"Run created: {result.run_id}")
        print(f"Status: {result.status}")
        print(f"Context: {result.context_path}")
        print(f"Data quality: {quality.status} ({quality.bar_count} bars)")
        _print_warnings(quality)
    return EXIT_OK


def _print_warnings(quality: DataQuality) -> None:
    for warning in quality.warnings:
        print(f"  ! {warning.code}: {warning.message}")


class _UsageError(Exception):
    """A flag combination the parser cannot express on its own."""


def _classify_failure(exc: PipelineError) -> int:
    """Exit code for a failure: configuration problems are not data problems.

    A missing API key and a duplicated candle are fixed in different places -
    one in the environment, one in the input - so anything scripting these
    commands needs to tell them apart.
    """
    configuration = isinstance(
        exc,
        WriterConfigurationError
        | ReviewConfigurationError
        | FinalizeConfigurationError
        | PublisherConfigurationError,
    )
    return EXIT_ERROR if configuration else EXIT_INVALID_DATA


def _report_failure(exc: PipelineError, *, as_json: bool) -> int:
    """Print a failure and choose an exit code that says what kind it was."""
    code = _classify_failure(exc)
    configuration = code == EXIT_ERROR

    if as_json:
        print(json.dumps({"valid": False, "error": exc.to_dict()}, ensure_ascii=False, indent=2))
    else:
        label = "Configuration error" if configuration else "Validation failed"
        print(f"{label}: {exc}", file=sys.stderr)
        if exc.details:
            print(f"Details: {json.dumps(exc.details, ensure_ascii=False)}", file=sys.stderr)
    return code


def _writer_client(*, fake: bool, model: str | None) -> WriterClient:
    """Pick the writer client for this invocation.

    ``fake`` short-circuits before any credential is read, so a smoke run cannot
    accidentally reach the provider or need a key present.
    """
    if fake:
        return FakeWriterClient()
    settings = WriterSettings.from_env(model_override=model)
    from goldpipeline.adapters.anthropic_writer import AnthropicWriterClient

    return AnthropicWriterClient(settings)


def _cmd_write_draft(args: argparse.Namespace) -> int:
    client = _writer_client(fake=args.fake_writer, model=args.model)
    result = write_draft(
        run_id=args.run_id,
        store=RunStore(args.runs_dir),
        client=client,
    )

    if not result.succeeded:
        assert result.error is not None
        if args.json:
            print(
                json.dumps(
                    {
                        "run_id": result.run_id,
                        "status": result.status,
                        "writer_status": "FAILED",
                        "error": result.error.to_dict(),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(f"Run: {result.run_id}", file=sys.stderr)
            print("Writer: FAILED", file=sys.stderr)
            print(f"Error: {result.error}", file=sys.stderr)
            if result.error.details:
                print(
                    f"Details: {json.dumps(result.error.details, ensure_ascii=False)}",
                    file=sys.stderr,
                )
        return EXIT_INVALID_DATA

    written = result.result
    assert written is not None
    if args.json:
        print(
            json.dumps(
                {
                    "run_id": result.run_id,
                    "status": result.status,
                    "writer_status": written.status,
                    "provider": written.provider,
                    "model": written.model,
                    "prompt_version": written.prompt_version,
                    "draft": str(result.draft_path),
                    "metadata": str(result.metadata_path),
                    "article_chars": written.article_chars,
                    "source_claims": len(written.source_claims),
                    "warnings": [w.code for w in written.warnings],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(f"Run: {result.run_id}")
        print(f"Writer: {written.provider} ({written.model})")
        print(f"Status: {written.status}")
        print(f"Draft: {result.draft_path}")
        print(f"Metadata: {result.metadata_path}")
        print(f"Article: {written.article_chars} chars, {len(written.source_claims)} claims")
        for warning in written.warnings:
            print(f"  ! {warning.code}: {warning.message}")
    return EXIT_OK


def _reviewer_client(*, fake: bool, model: str | None) -> ReviewerClient:
    """Pick the reviewer client for this invocation.

    ``fake`` short-circuits before any credential is read, so a smoke run cannot
    accidentally reach the provider or need a key present.
    """
    if fake:
        return FakeReviewerClient()
    settings = ReviewerSettings.from_env(model_override=model)
    from goldpipeline.adapters.openai_reviewer import OpenAIReviewerClient

    return OpenAIReviewerClient(settings)


def _cmd_review_draft(args: argparse.Namespace) -> int:
    client = _reviewer_client(fake=args.fake_reviewer, model=args.model)
    result = review_draft(run_id=args.run_id, store=RunStore(args.runs_dir), client=client)

    if not result.succeeded:
        assert result.error is not None
        if args.json:
            print(
                json.dumps(
                    {
                        "run_id": result.run_id,
                        "status": result.status,
                        "review_status": "FAILED",
                        "error": result.error.to_dict(),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(f"Run: {result.run_id}", file=sys.stderr)
            print("Reviewer: FAILED", file=sys.stderr)
            print(f"Error: {result.error}", file=sys.stderr)
            if result.error.details:
                print(
                    f"Details: {json.dumps(result.error.details, ensure_ascii=False)}",
                    file=sys.stderr,
                )
        return EXIT_INVALID_DATA

    review = result.result
    assert review is not None
    if args.json:
        print(
            json.dumps(
                {
                    "run_id": result.run_id,
                    "status": result.status,
                    "verdict": review.status,
                    "model_verdict": review.model_status,
                    "verdict_source": review.verdict_source,
                    "score": review.score,
                    "issues": len(review.issues),
                    "blocking_issues": len(review.blocking_issues),
                    "provider": review.provider,
                    "model": review.model,
                    "prompt_version": review.prompt_version,
                    "review": str(result.review_path),
                    "deterministic_findings": [
                        str(finding.code) for finding in review.deterministic_findings
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(f"Run: {result.run_id}")
        print(f"Reviewer: {review.provider} ({review.model})")
        print(f"Verdict: {review.status}")
        print(f"Score: {review.score}")
        print(f"Issues: {len(review.issues)}")
        print(f"Review: {result.review_path}")
        for note in review.policy_notes:
            print(f"  * {note}")
        for issue in review.issues:
            print(f"  ! [{issue.severity}] {issue.category}: {issue.message}")
    return EXIT_OK


def _finalizer_client(*, fake: bool, model: str | None) -> FinalizerClient:
    """Build the finalizer client.

    Only ever called through :class:`LazyFinalizerClient`, so credentials are
    read lazily and only for the real revision path: a PASS is a byte copy and a
    REJECT is a refusal, and neither should demand an API key from an operator
    who is only trying to finish a Run.
    """
    if fake:
        return FakeFinalizerClient()

    from goldpipeline.adapters.anthropic_finalizer import AnthropicFinalizerClient

    return AnthropicFinalizerClient(FinalizerSettings.from_env(model_override=model))


def _cmd_finalize(args: argparse.Namespace) -> int:
    result = finalize_run(
        run_id=args.run_id,
        store=RunStore(args.runs_dir),
        client=LazyFinalizerClient(
            lambda: _finalizer_client(fake=args.fake_finalizer, model=args.model)
        ),
    )

    if not result.succeeded:
        assert result.error is not None
        blocked = result.blocked

        # A missing key surfaces from inside the stage, because credentials are
        # read lazily and only on the revision path. It is still a configuration
        # problem, so it must exit like one rather than like bad data.
        if isinstance(result.error, FinalizeConfigurationError):
            return _report_failure(result.error, as_json=args.json)

        if args.json:
            print(
                json.dumps(
                    {
                        "run_id": result.run_id,
                        "status": result.status,
                        "finalization": "BLOCKED" if blocked else "FAILED",
                        "error": result.error.to_dict(),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        elif blocked:
            print(f"Run: {result.run_id}", file=sys.stderr)
            print("Review: REJECT", file=sys.stderr)
            print("Finalization blocked.", file=sys.stderr)
            print(f"Reason: {result.error.message}", file=sys.stderr)
        else:
            print(f"Run: {result.run_id}", file=sys.stderr)
            print("Finalization: FAILED", file=sys.stderr)
            print(f"Error: {result.error}", file=sys.stderr)
            if result.error.details:
                print(
                    f"Details: {json.dumps(result.error.details, ensure_ascii=False)}",
                    file=sys.stderr,
                )
        return EXIT_BLOCKED if blocked else EXIT_INVALID_DATA

    final = result.result
    assert final is not None
    if args.json:
        print(
            json.dumps(
                {
                    "run_id": result.run_id,
                    "status": result.status,
                    "review": final.review_status,
                    "finalization": final.finalization_mode,
                    "provider_called": final.provider_called,
                    "issues_applied": final.applied_count,
                    "issues_total": len(final.issue_resolutions),
                    "article_chars": final.article_chars,
                    "model": final.model,
                    "prompt_version": final.prompt_version,
                    "final": str(result.final_path),
                    "metadata": str(result.metadata_path),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(f"Run: {result.run_id}")
        print(f"Review: {final.review_status}")
        print(f"Finalization: {final.finalization_mode}")
        print(f"Provider called: {'Yes' if final.provider_called else 'No'}")
        if final.issue_resolutions:
            print(f"Issues resolved: {final.applied_count}/{len(final.issue_resolutions)}")
        if final.model:
            print(f"Finalizer: {final.provider} ({final.model})")
        print(f"Final: {result.final_path}")
        for resolution in final.issue_resolutions:
            print(f"  - [{resolution.resolution}] {resolution.issue_id}: {resolution.description}")
    return EXIT_OK


def _cmd_gate_publish(args: argparse.Namespace) -> int:
    """Run the final publish gate. No provider, no credentials, no network."""
    result = gate_publish(run_id=args.run_id, store=RunStore(args.runs_dir))
    decision = result.decision
    passed, warned, failed = decision.counts

    if args.json:
        print(
            json.dumps(
                {
                    "run_id": result.run_id,
                    "status": result.status,
                    "gate_version": decision.gate_version,
                    "decision": decision.decision,
                    "checks": {"passed": passed, "warnings": warned, "failed": failed},
                    "blockers": [
                        {"code": b.code, "severity": b.severity, "message": b.message}
                        for b in decision.blockers
                    ],
                    "warnings_detail": [
                        {"code": w.code, "severity": w.severity, "message": w.message}
                        for w in decision.warnings
                    ],
                    "artifact": str(result.decision_path),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return EXIT_OK if result.approved else EXIT_BLOCKED

    stream = sys.stdout if result.approved else sys.stderr
    print(f"Run: {result.run_id}", file=stream)
    print(f"Gate: {decision.gate_version}", file=stream)
    print(f"Decision: {decision.decision}", file=stream)
    print(f"Checks: {passed} passed, {warned} warnings, {failed} failed", file=stream)

    if decision.blockers:
        print(f"Blockers: {len(decision.blockers)}", file=stream)
        for blocker in decision.blockers:
            print(f"  - [{blocker.severity}] {blocker.code}: {blocker.message}", file=stream)
    for warning in decision.warnings:
        print(f"  ! {warning.code}: {warning.message}", file=stream)

    print(f"Artifact: {result.decision_path}", file=stream)
    return EXIT_OK if result.approved else EXIT_BLOCKED


FAKE_TARGET_CHAT = "@fake_offline_channel"
"""Destination recorded by `--fake-publisher`.

Obviously not a real channel, so a fake attempt can never be mistaken for one
that reached Telegram - and the offline path needs no credentials at all.
"""


def _publisher_client(*, fake: bool) -> tuple[PublisherClient, str]:
    """Pick the transport and the destination.

    The destination comes from configuration only. There is deliberately no
    `--chat-id`: a flag that redirects where an approved article is posted is
    one typo away from publishing to the wrong channel, and it would also give
    anything that can influence a command line control over the destination.
    """
    if fake:
        return FakePublisherClient(), FAKE_TARGET_CHAT

    settings = TelegramSettings.from_env()
    from goldpipeline.adapters.telegram_publisher import TelegramPublisherClient

    return TelegramPublisherClient(settings), settings.target_chat


def _cmd_publish(args: argparse.Namespace) -> int:
    client, target = _publisher_client(fake=args.fake_publisher)
    outcome = publish_run(
        run_id=args.run_id,
        store=RunStore(args.runs_dir),
        client=client,
        target_chat=target,
    )
    result = outcome.result
    ok = outcome.published

    if args.json:
        print(
            json.dumps(
                {
                    "run_id": outcome.run_id,
                    "status": outcome.status,
                    "publish_status": result.status,
                    "provider": result.provider,
                    "target_chat": result.target_chat,
                    "attempt_id": result.attempt_id,
                    "chunk_count": result.chunk_count,
                    "confirmed_count": result.confirmed_count,
                    "message_ids": [m.message_id for m in result.messages],
                    "failure": (
                        {
                            "category": result.failure.category,
                            "code": result.failure.safe_code,
                            "message": result.failure.safe_message,
                        }
                        if result.failure
                        else None
                    ),
                    "artifact": str(outcome.result_path),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return EXIT_OK if ok else EXIT_BLOCKED

    stream = sys.stdout if ok else sys.stderr
    print(f"Run: {outcome.run_id}", file=stream)
    print(f"Publisher: {result.provider} -> {result.target_chat}", file=stream)
    print(f"Status: {result.status}", file=stream)
    print(f"Delivered: {result.confirmed_count}/{result.chunk_count} message(s)", file=stream)
    for message in result.messages:
        print(f"  chunk {message.chunk_index}: message_id={message.message_id}", file=stream)
    if result.failure is not None:
        print(f"Failure: [{result.failure.safe_code}] {result.failure.safe_message}", file=stream)
    for warning in result.warnings:
        print(f"  ! {warning}", file=stream)
    print(f"Artifact: {outcome.result_path}", file=stream)

    return EXIT_OK if ok else EXIT_BLOCKED


def _resolve_mode(args: argparse.Namespace) -> PipelineMode:
    """Turn the flags into a mode, refusing an unguarded real publish.

    ``--publish`` alone is enough for the offline transport, because nothing
    leaves the machine. For the real one it is not: ``--confirm-real-publish``
    has to be there too. Both flags name what they do, so no combination of them
    can post an article by accident.

    Passing ``--fake-publisher`` and ``--confirm-real-publish`` together is
    allowed and the fake wins. The alternative - rejecting the pair - would push
    an operator towards dropping ``--fake-publisher``, which is exactly the
    wrong flag to drop.
    """
    mode = (
        PipelineMode.PUBLISH if args.publish else PipelineMode(args.mode.upper().replace("-", "_"))
    )

    if mode is PipelineMode.PUBLISH and not args.fake_publisher and not args.confirm_real_publish:
        raise _UsageError(
            "publishing for real needs --confirm-real-publish alongside --publish. "
            "Use --fake-publisher to rehearse the whole pipeline offline instead."
        )
    return mode


def _pipeline_clients(args: argparse.Namespace) -> PipelineClients:
    """Wire up client factories. Nothing is built here.

    Each lambda runs at most once, and only if its stage is actually reached, so
    a Run that stops at the reviewer never needs a finalizer key and a Run
    resumed at the gate needs no keys at all.
    """
    fake_ai = args.fake_ai
    return PipelineClients(
        writer=lambda: _writer_client(fake=fake_ai or args.fake_writer, model=args.writer_model),
        reviewer=lambda: _reviewer_client(
            fake=fake_ai or args.fake_reviewer, model=args.reviewer_model
        ),
        finalizer=lambda: _finalizer_client(
            fake=fake_ai or args.fake_finalizer, model=args.finalizer_model
        ),
        publisher=lambda: _publisher_client(fake=args.fake_publisher),
    )


def _cmd_pipeline_run(args: argparse.Namespace) -> int:
    mode = _resolve_mode(args)
    outcome = run_pipeline(
        analysis_source=JsonFileAnalysisSource(args.telegram),
        market_source=JsonFileMarketDataSource(args.ohlc),
        store=RunStore(args.runs_dir),
        clients=_pipeline_clients(args),
        mode=mode,
        expected_symbol=args.symbol,
        now=args.now,
    )
    return _report_execution(outcome, as_json=args.json)


def _cmd_pipeline_resume(args: argparse.Namespace) -> int:
    mode = _resolve_mode(args)
    try:
        outcome = resume_pipeline(
            run_id=args.run_id,
            store=RunStore(args.runs_dir),
            clients=_pipeline_clients(args),
            mode=mode,
            now=args.now,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"{exc}", file=sys.stderr)
        return EXIT_ERROR
    return _report_execution(outcome, as_json=args.json)


_STAGE_LABELS = {
    "NORMALIZE": "NORMALIZE",
    "WRITE": "WRITER",
    "REVIEW": "REVIEW",
    "FINALIZE": "FINALIZE",
    "GATE": "GATE",
    "PUBLISH": "PUBLISH",
}


def _report_execution(outcome: PipelineRunResult, *, as_json: bool) -> int:
    """Print the execution and choose an exit code.

    The article itself is never printed. A pipeline summary is something an
    operator reads dozens of times a week; dumping the full text into it would
    make the one line that matters - the final status - the hardest to find.
    """
    result = outcome.result

    if as_json:
        print(result.model_dump_json(indent=2))
        return _pipeline_exit_code(outcome)

    stream = sys.stdout if outcome.succeeded else sys.stderr
    print(f"Run: {result.run_id}", file=stream)
    print(f"Mode: {result.mode}", file=stream)

    if result.stages:
        print(file=stream)
        for stage in result.stages:
            label = _STAGE_LABELS.get(str(stage.stage), str(stage.stage))
            print(f"{label:<10} {stage.detail or stage.outcome}", file=stream)

    print(file=stream)
    print(f"Final status: {result.run_status}", file=stream)
    print(f"Execution: {result.status}", file=stream)
    print(f"Run directory: {outcome.run_dir}", file=stream)

    if outcome.error is not None:
        print(f"Error: {outcome.error}", file=stream)
        if outcome.error.details:
            print(
                f"Details: {json.dumps(outcome.error.details, ensure_ascii=False)}",
                file=stream,
            )
    return _pipeline_exit_code(outcome)


def _pipeline_exit_code(outcome: PipelineRunResult) -> int:
    """Map an execution onto an exit code.

    A gate declining and a stage breaking are different events for whatever is
    scripting this, so they get different numbers - and a missing API key keeps
    exiting 1 the way it does from every other command here.
    """
    if outcome.succeeded:
        return EXIT_OK
    if outcome.result.status is PipelineStatus.FAILED and outcome.error is not None:
        return _classify_failure(outcome.error)
    return EXIT_BLOCKED


def _cmd_show_run(args: argparse.Namespace) -> int:
    store = RunStore(args.runs_dir)
    try:
        run = store.open(args.run_id)
    except (FileNotFoundError, ValueError) as exc:
        print(f"{exc}", file=sys.stderr)
        return EXIT_ERROR

    manifest = run.load_manifest()
    print(f"Run: {manifest.run_id}")
    print(f"Status: {manifest.status}")
    print(f"Created: {manifest.created_at.isoformat()}")
    print(f"Pipeline: {manifest.pipeline_version} / schema {manifest.schema_version}")
    print("Files:")
    for ref in [*manifest.source_files, *manifest.artifact_files]:
        print(f"  {ref.name:<22} {ref.size_bytes:>8} bytes  sha256={ref.sha256[:12]}...")
    if manifest.error:
        print(f"Error: [{manifest.error.code}] {manifest.error.message}")
    return EXIT_OK


def _cmd_list_runs(args: argparse.Namespace) -> int:
    run_ids = RunStore(args.runs_dir).list_run_ids()
    if not run_ids:
        print(f"No runs under {args.runs_dir}")
        return EXIT_OK
    for run_id in run_ids:
        print(run_id)
    return EXIT_OK


def _force_utf8_output() -> None:
    """Make stdout/stderr UTF-8 regardless of the console code page.

    A Windows console defaults to a legacy code page (cp1252 here), which cannot
    encode Vietnamese. Any path, warning or error message containing Vietnamese
    would then crash the CLI with a UnicodeEncodeError instead of printing -
    and this pipeline exists to handle Vietnamese text.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        # a detached or test-captured stream may refuse reconfiguration
        with contextlib.suppress(ValueError, OSError):
            reconfigure(encoding="utf-8", errors="replace")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    _force_utf8_output()
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)

    handlers = {
        "create-run": _cmd_create_run,
        "write-draft": _cmd_write_draft,
        "review-draft": _cmd_review_draft,
        "finalize": _cmd_finalize,
        "gate-publish": _cmd_gate_publish,
        "publish": _cmd_publish,
        "pipeline-run": _cmd_pipeline_run,
        "pipeline-resume": _cmd_pipeline_resume,
        "show-run": _cmd_show_run,
        "list-runs": _cmd_list_runs,
    }
    try:
        return handlers[args.command](args)
    except _UsageError as exc:
        print(f"{exc}", file=sys.stderr)
        return EXIT_ERROR
    except RunLockedError as exc:
        # Not a data problem and not a gate declining: someone else is driving
        # this Run right now, or a crash left a lock behind for a human.
        print(f"{exc}", file=sys.stderr)
        print(f"Details: {json.dumps(exc.details, ensure_ascii=False)}", file=sys.stderr)
        return EXIT_ERROR
    except FinalizationBlockedError as exc:
        print(f"Finalization blocked: {exc}", file=sys.stderr)
        return EXIT_BLOCKED
    except PipelineError as exc:
        return _report_failure(exc, as_json=getattr(args, "json", False))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
