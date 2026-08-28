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
from typing import Any

from goldpipeline import PIPELINE_VERSION
from goldpipeline.adapters.fake_finalizer import FakeFinalizerClient
from goldpipeline.adapters.fake_reviewer import FakeReviewerClient
from goldpipeline.adapters.fake_writer import FakeWriterClient
from goldpipeline.adapters.file_source import JsonFileAnalysisSource, JsonFileMarketDataSource
from goldpipeline.adapters.finalizer_client import FinalizerClient
from goldpipeline.adapters.reviewer_client import ReviewerClient
from goldpipeline.adapters.writer_client import WriterClient
from goldpipeline.config import FinalizerSettings, ReviewerSettings, WriterSettings
from goldpipeline.domain.errors import (
    FinalizationBlockedError,
    FinalizeConfigurationError,
    PipelineError,
    ReviewConfigurationError,
    WriterConfigurationError,
)
from goldpipeline.logging_setup import configure_logging
from goldpipeline.schemas.quality import DataQuality
from goldpipeline.services.finalizer import finalize_run
from goldpipeline.services.pipeline import create_run, validate_sources
from goldpipeline.services.reviewer import review_draft
from goldpipeline.services.writer import write_draft
from goldpipeline.storage.run_store import RunStore

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_INVALID_DATA = 2
EXIT_BLOCKED = 3
"""A stage declined to run because an earlier verdict forbade it.

Distinct from a failure: nothing went wrong, and retrying will not help.
A caller automating the pipeline needs to tell the two apart.
"""

DEFAULT_RUNS_DIR = Path("runs")


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

    show = subparsers.add_parser("show-run", help="Print a summary of an existing Run.")
    show.add_argument("run_id", help="Run identifier, e.g. 20260828_022701_a83f2c.")
    show.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)

    subparsers.add_parser("list-runs", help="List Run ids under the runs directory.").add_argument(
        "--runs-dir", type=Path, default=DEFAULT_RUNS_DIR
    )

    return parser


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


def _report_failure(exc: PipelineError, *, as_json: bool) -> int:
    """Print a failure and choose an exit code that says what kind it was.

    A missing API key and a duplicated candle are different problems: one is
    fixed in the environment, the other in the data. Anything scripting this
    command needs to tell them apart, so configuration failures exit 1 and data
    failures exit 2.
    """
    configuration = isinstance(
        exc,
        WriterConfigurationError | ReviewConfigurationError | FinalizeConfigurationError,
    )
    code = EXIT_ERROR if configuration else EXIT_INVALID_DATA

    if as_json:
        print(json.dumps({"valid": False, "error": exc.to_dict()}, ensure_ascii=False, indent=2))
    else:
        label = "Configuration error" if configuration else "Validation failed"
        print(f"{label}: {exc}", file=sys.stderr)
        if exc.details:
            print(f"Details: {json.dumps(exc.details, ensure_ascii=False)}", file=sys.stderr)
    return code


def _build_writer_client(args: argparse.Namespace) -> WriterClient:
    """Pick the writer client for this invocation.

    ``--fake-writer`` short-circuits before any credential is read, so a smoke
    run cannot accidentally reach the provider or need a key present.
    """
    if args.fake_writer:
        return FakeWriterClient()
    settings = WriterSettings.from_env(model_override=args.model)
    from goldpipeline.adapters.anthropic_writer import AnthropicWriterClient

    return AnthropicWriterClient(settings)


def _cmd_write_draft(args: argparse.Namespace) -> int:
    client = _build_writer_client(args)
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


def _build_reviewer_client(args: argparse.Namespace) -> ReviewerClient:
    """Pick the reviewer client for this invocation.

    ``--fake-reviewer`` short-circuits before any credential is read, so a smoke
    run cannot accidentally reach the provider or need a key present.
    """
    if args.fake_reviewer:
        return FakeReviewerClient()
    settings = ReviewerSettings.from_env(model_override=args.model)
    from goldpipeline.adapters.openai_reviewer import OpenAIReviewerClient

    return OpenAIReviewerClient(settings)


def _cmd_review_draft(args: argparse.Namespace) -> int:
    client = _build_reviewer_client(args)
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


def _build_finalizer_client(args: argparse.Namespace) -> FinalizerClient | None:
    """Pick the finalizer client, or none when the verdict will not need one.

    Credentials are read lazily and only for the real revision path: a PASS is a
    byte copy and a REJECT is a refusal, and neither should demand an API key
    from an operator who is only trying to finish a Run.
    """
    if args.fake_finalizer:
        return FakeFinalizerClient()

    from goldpipeline.adapters.anthropic_finalizer import AnthropicFinalizerClient

    return AnthropicFinalizerClient(FinalizerSettings.from_env(model_override=args.model))


def _cmd_finalize(args: argparse.Namespace) -> int:
    result = finalize_run(
        run_id=args.run_id,
        store=RunStore(args.runs_dir),
        client=_LazyFinalizerClient(args),
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


class _LazyFinalizerClient:
    """Defers building the real client until a revision actually needs one.

    The finalizer only asks for a client on the ``NEEDS_REVISION`` path, so
    wrapping it this way keeps a passthrough and a block free of any credential
    requirement while still failing clearly when a revision has no key.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self._inner: FinalizerClient | None = None

    def _client(self) -> FinalizerClient:
        if self._inner is None:
            built = _build_finalizer_client(self._args)
            assert built is not None
            self._inner = built
        return self._inner

    @property
    def provider(self) -> str:
        return self._client().provider

    @property
    def model(self) -> str:
        return self._client().model

    def finalize(self, request: Any) -> Any:
        return self._client().finalize(request)


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
        "show-run": _cmd_show_run,
        "list-runs": _cmd_list_runs,
    }
    try:
        return handlers[args.command](args)
    except FinalizationBlockedError as exc:
        print(f"Finalization blocked: {exc}", file=sys.stderr)
        return EXIT_BLOCKED
    except PipelineError as exc:
        return _report_failure(exc, as_json=getattr(args, "json", False))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
