"""Pipeline stages: normalization, context building, orchestration."""

from goldpipeline.services.chunking import plan_chunks, utf16_length
from goldpipeline.services.context_builder import build_context
from goldpipeline.services.finalizer import FinalizeRunResult, finalize_run
from goldpipeline.services.finalizer_prompt import build_finalizer_prompt
from goldpipeline.services.market_facts import build_market_facts, format_price
from goldpipeline.services.normalizer import (
    NormalizedAnalysis,
    NormalizedMarketData,
    normalize_analysis,
    normalize_market_data,
    sanitize_analysis_text,
)
from goldpipeline.services.pipeline import RunResult, create_run, validate_sources
from goldpipeline.services.precheck import PrecheckReport, run_prechecks
from goldpipeline.services.publish_gate import GateResult, gate_publish
from goldpipeline.services.publisher import PublishRunResult, publish_run
from goldpipeline.services.reviewer import ReviewRunResult, review_draft
from goldpipeline.services.reviewer_prompt import build_reviewer_prompt
from goldpipeline.services.source_guard import screen_source_prices
from goldpipeline.services.writer import WriterRunResult, write_draft
from goldpipeline.services.writer_prompt import WriterPrompt, build_writer_prompt

__all__ = [
    "NormalizedAnalysis",
    "NormalizedMarketData",
    "FinalizeRunResult",
    "GateResult",
    "PublishRunResult",
    "PrecheckReport",
    "ReviewRunResult",
    "RunResult",
    "WriterPrompt",
    "WriterRunResult",
    "build_context",
    "build_finalizer_prompt",
    "build_market_facts",
    "build_reviewer_prompt",
    "build_writer_prompt",
    "create_run",
    "finalize_run",
    "gate_publish",
    "plan_chunks",
    "publish_run",
    "utf16_length",
    "normalize_analysis",
    "normalize_market_data",
    "format_price",
    "sanitize_analysis_text",
    "review_draft",
    "run_prechecks",
    "screen_source_prices",
    "validate_sources",
    "write_draft",
]
