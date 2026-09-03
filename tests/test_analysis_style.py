"""The ANALYSIS shape, and the days it has to survive.

A corpus rather than a golden article. Human writing varies, so the tests pin
*properties* - the piece has a verdict, it did not invent a driver to balance
the page, it said price had not confirmed rather than that news caused a move -
and pin exact strings only where the contract fixes them: the headings, the
disclaimer, and the placeholder that stands in for a side with nothing on it.

The samples are written the way the prompt asks for, so they double as a record
of what the round was aiming at. Anything a real model produces will differ in
wording; if it differs in *property*, one of these tests fails.

Offline throughout: no provider, no network, no clock dependency.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from goldpipeline.prompts import (
    DEFAULT_WRITER_PROMPT,
    GOLD_HUMAN_STYLE_V1,
    GOLD_WRITER_V3,
    GOLD_WRITER_V4,
    PROMPTS_DIR,
    load_prompt,
)
from goldpipeline.schemas.article import ArticleType
from goldpipeline.schemas.article_contract import contract_for
from goldpipeline.schemas.output_findings import OutputFindingCode
from goldpipeline.services.analysis_contract import (
    BLOCKING_CODES,
    inspect_article,
    is_enforced,
    missing_article_date,
)

DATE = "04.09.2026"
DISCLAIMER = "🔴 Nhận định cá nhân, không phải lời khuyên đầu tư."
NO_DRIVER = "Chưa thấy gì đáng kể."


def article(
    *,
    verdict: str,
    up: str,
    down: str,
    price: str,
    watching: str,
    date: str = DATE,
    disclaimer: str = DISCLAIMER,
) -> str:
    return "\n".join(
        [
            f"🕯 PHÂN TÍCH VÀNG — {date}",
            "",
            f"⚡ Chốt: {verdict}",
            "",
            "🟢 Đẩy lên:",
            up,
            "",
            "🔴 Kéo xuống:",
            down,
            "",
            "📈 Giá đang nói gì?",
            price,
            "",
            "🧭 Mình đang chờ:",
            watching,
            "",
            disclaimer,
        ]
    )


def blocking(text: str) -> set[OutputFindingCode]:
    return {f.code for f in inspect_article(text, ArticleType.ANALYSIS).blocking}


def observed(text: str) -> set[OutputFindingCode]:
    return {f.code for f in inspect_article(text, ArticleType.ANALYSIS).observed}


# --------------------------------------------------------------------------
# 1-7: the days
# --------------------------------------------------------------------------

BULLISH_HEAVY = article(
    verdict="dòng tin nghiêng hẳn về phía vàng, nhưng CPI vẫn treo đó.",
    up="ETF mua ròng gần 10 tấn - tiền thật, không phải kỳ vọng. USD mất giá, và "
    "Fed chưa nói gì cứng hơn.",
    down="Chỉ còn CPI. Nhưng một mình CPI đủ để lật.",
    price="Giá đi cùng hướng với câu chuyện trên. Nhịp mạnh nhất rơi vào sáng nay.",
    watching="Dữ liệu Mỹ mềm tiếp thì vàng còn được đỡ. CPI nóng thì áp lực đến nhanh.",
)

ONE_SIDED = article(
    verdict="mọi thứ đang chỉ về một phía, và mình không đi tìm phía còn lại.",
    up="USD yếu cả phiên, ETF mua ròng, lợi suất hạ. Ba tin nhỏ cùng chiều.",
    down=NO_DRIVER,
    price="Giá bật lên và giữ được. Không có gì mâu thuẫn.",
    watching="Nếu USD hồi lại thì câu chuyện này mỏng đi rất nhanh.",
)

MIXED = article(
    verdict="hai bên đang cân, nhưng mình vẫn nghiêng nhẹ về phía mua.",
    up="ETF mua ròng, và USD chưa lấy lại được vùng cũ.",
    down="Lợi suất trái phiếu nhích lên. Đây là thứ duy nhất mình thấy đáng ngại.",
    price="Giá nghiêng lên nhẹ. Chưa đủ để gọi là xác nhận.",
    watching="Lợi suất tiếp tục tăng thì cán cân đổi phía.",
)

WEAK_EVIDENCE = article(
    verdict="tin thì có, nhưng mỏng. Đoạn này mình chưa tin lắm.",
    up="Một bản tin về dòng vốn ETF. Chỉ một, và số không lớn.",
    down=NO_DRIVER,
    price="Giá gần như đứng yên. Trùng thời điểm thôi, chưa đủ để nói là nguyên nhân.",
    watching="Chờ thêm một phiên nữa rồi tính.",
)

PRICE_CONFIRMS = article(
    verdict="tin nghiêng tích cực và giá đang đi cùng hướng.",
    up="USD yếu, ETF mua vào.",
    down=NO_DRIVER,
    price="Giá tăng sau thời điểm tin ra và giữ được vùng đó tới cuối phiên.",
    watching="Giữ được vùng này thì câu chuyện còn nguyên.",
)

PRICE_CONTRADICTS = article(
    verdict="tin thì tốt, giá thì không nghe.",
    up="USD yếu và ETF mua ròng - trên giấy là hỗ trợ.",
    down=NO_DRIVER,
    price="Giá đang đi ngược câu chuyện. Tin nghiêng tích cực nhưng giá chưa xác nhận.",
    watching="Khi giá không nghe tin, mình chờ chứ không đoán.",
)

QUIET_DAY = article(
    verdict="phiên trống, không có gì để nói.",
    up=NO_DRIVER,
    down=NO_DRIVER,
    price="Giá đi ngang cả phiên.",
    watching="Chờ tin.",
)


class TestTheDays:
    def test_a_bullish_heavy_day_states_a_leaning_and_keeps_the_risk(self) -> None:
        assert blocking(BULLISH_HEAVY) == set()
        assert "⚡ Chốt:" in BULLISH_HEAVY
        # Asymmetric by construction: the bullish side is longer because the
        # day was, not because the template has two headings.
        up = BULLISH_HEAVY.split("🟢 Đẩy lên:")[1].split("🔴 Kéo xuống:")[0]
        down = BULLISH_HEAVY.split("🔴 Kéo xuống:")[1].split("📈")[0]
        assert len(up) > len(down)
        assert NO_DRIVER not in BULLISH_HEAVY

    def test_a_one_sided_day_says_so_instead_of_inventing_a_bear_case(self) -> None:
        assert blocking(ONE_SIDED) == set()
        down = ONE_SIDED.split("🔴 Kéo xuống:")[1].split("📈")[0].strip()
        assert down == NO_DRIVER

    def test_a_mixed_day_still_commits_to_a_leaning(self) -> None:
        assert blocking(MIXED) == set()
        assert NO_DRIVER not in MIXED
        verdict = MIXED.split("⚡ Chốt:")[1].split("\n")[0]
        assert "nghiêng" in verdict

    def test_a_weak_evidence_day_says_the_evidence_is_weak(self) -> None:
        assert blocking(WEAK_EVIDENCE) == set()
        assert "chưa tin lắm" in WEAK_EVIDENCE or "chưa đủ" in WEAK_EVIDENCE

    def test_price_confirming_uses_temporal_wording(self) -> None:
        assert blocking(PRICE_CONFIRMS) == set()
        assert "sau thời điểm" in PRICE_CONFIRMS
        assert OutputFindingCode.UNATTRIBUTED_CAUSAL_CLAIM not in blocking(PRICE_CONFIRMS)

    def test_price_contradicting_is_said_plainly(self) -> None:
        assert blocking(PRICE_CONTRADICTS) == set()
        assert "đi ngược" in PRICE_CONTRADICTS or "chưa xác nhận" in PRICE_CONTRADICTS

    def test_a_quiet_day_is_short_and_is_not_padded(self) -> None:
        report = inspect_article(QUIET_DAY, ArticleType.ANALYSIS)
        assert report.blocking == ()
        assert report.chars < contract_for(ArticleType.ANALYSIS).target_min_chars
        # Below the guidance target and perfectly acceptable: only the ceiling
        # is enforced, so a thin day is never a reason to write filler.
        assert not report.over_hard_cap
        assert not report.within_target

    @pytest.mark.parametrize(
        "sample",
        [BULLISH_HEAVY, ONE_SIDED, MIXED, WEAK_EVIDENCE, PRICE_CONFIRMS, PRICE_CONTRADICTS],
        ids=["bullish", "one_sided", "mixed", "weak", "confirms", "contradicts"],
    )
    def test_every_realistic_day_fits_the_ceiling(self, sample: str) -> None:
        report = inspect_article(sample, ArticleType.ANALYSIS)
        assert not report.over_hard_cap
        assert report.chars <= 1300


# --------------------------------------------------------------------------
# 8-9: style symptoms, observable but not a verdict
# --------------------------------------------------------------------------

AI_ISH = article(
    verdict="thị trường vàng đang trong giai đoạn tích luỹ.",
    up="Trong bối cảnh USD suy yếu, dòng vốn ETF quay lại thị trường vàng. "
    "Bên cạnh đó, lợi suất trái phiếu hạ nhiệt. Hơn nữa, Fed chưa cứng rắn.",
    down="Tuy nhiên, CPI vẫn là rủi ro. Do đó, cần thận trọng. "
    "Ngoài ra, thanh khoản đang mỏng dần.",
    price="Có thể thấy rằng giá đang phản ánh kỳ vọng. Nhìn chung xu hướng vẫn tích cực.",
    watching="Tóm lại, cần theo dõi thêm dữ liệu để có đánh giá chính xác hơn.",
)


class TestStyleSymptoms:
    def test_an_ai_ish_sample_produces_symptoms(self) -> None:
        found = observed(AI_ISH)
        assert OutputFindingCode.THROAT_CLEARING in found
        assert OutputFindingCode.CONNECTIVE_DENSITY in found

    def test_style_symptoms_do_not_block_in_this_round(self) -> None:
        """Writer-first: the reviewer gets a style verdict next round, not now."""
        assert blocking(AI_ISH) == set()
        assert not (
            {
                OutputFindingCode.THROAT_CLEARING,
                OutputFindingCode.CONNECTIVE_DENSITY,
                OutputFindingCode.SENTENCE_LENGTH_UNIFORM,
                OutputFindingCode.PARAGRAPH_LENGTH_UNIFORM,
                OutputFindingCode.REPEATED_OPENER,
            }
            & BLOCKING_CODES
        )

    @pytest.mark.parametrize(
        "sample",
        [BULLISH_HEAVY, ONE_SIDED, MIXED, WEAK_EVIDENCE, PRICE_CONFIRMS, QUIET_DAY],
        ids=["bullish", "one_sided", "mixed", "weak", "confirms", "quiet"],
    )
    def test_natural_samples_do_not_over_trigger(self, sample: str) -> None:
        """The expensive failure: a check that fires on good writing."""
        cheap = {
            OutputFindingCode.THROAT_CLEARING,
            OutputFindingCode.CONNECTIVE_DENSITY,
            OutputFindingCode.SENTENCE_LENGTH_UNIFORM,
            OutputFindingCode.BULLET_SHAPE_UNIFORM,
            OutputFindingCode.DUPLICATE_STATEMENT,
        }
        assert not (observed(sample) & cheap), sample[:60]


# --------------------------------------------------------------------------
# 10-11: it must not become another product
# --------------------------------------------------------------------------


class TestNotAnotherProduct:
    def test_a_news_digest_shaped_article_is_refused(self) -> None:
        digest_like = article(
            verdict="tin hôm nay.",
            up="🔹 08:30 — SPDR mua ròng 9.98 tấn\n🔹 10:00 — USD suy yếu\n"
            "🔹 14:15 — Fed giữ nguyên lãi suất",
            down=NO_DRIVER,
            price="Giá đi lên.",
            watching="Chờ thêm.",
        )
        assert OutputFindingCode.FORBIDDEN_STRUCTURE_PRESENT in blocking(digest_like)

    def test_a_trade_plan_shaped_article_is_refused(self) -> None:
        plan_like = article(
            verdict="canh vùng.",
            up="Mua quanh vùng dưới.",
            down=NO_DRIVER,
            price="▸ 4415 – 4417 · scalp\n▸ 4435 – 4437 · vùng chính",
            watching="Chờ giá về vùng.",
        )
        assert OutputFindingCode.FORBIDDEN_STRUCTURE_PRESENT in blocking(plan_like)


# --------------------------------------------------------------------------
# 12-13: causality
# --------------------------------------------------------------------------


class TestCausality:
    def test_an_unattributed_causal_claim_blocks(self) -> None:
        bad = article(
            verdict="vàng giảm hôm nay.",
            up=NO_DRIVER,
            down="Tin CPI khiến vàng giảm.",
            price="Giá xuống cả phiên.",
            watching="Chờ phiên sau.",
        )
        assert OutputFindingCode.UNATTRIBUTED_CAUSAL_CLAIM in blocking(bad)

    def test_the_temporal_phrasing_of_the_same_fact_is_allowed(self) -> None:
        good = article(
            verdict="vàng giảm hôm nay.",
            up=NO_DRIVER,
            down="Vàng giảm sau thời điểm CPI được công bố.",
            price="Giá xuống cả phiên.",
            watching="Chờ phiên sau.",
        )
        assert blocking(good) == set()

    def test_causality_is_a_blocking_finding_in_this_round(self) -> None:
        assert OutputFindingCode.UNATTRIBUTED_CAUSAL_CLAIM in BLOCKING_CODES


# --------------------------------------------------------------------------
# 14: numeric vocabulary
# --------------------------------------------------------------------------


class TestNumericSemantics:
    def test_net_change_and_range_stay_distinct(self) -> None:
        from goldpipeline.services.numeric_semantics import SemanticType

        assert len({SemanticType.PRICE_NET_CHANGE, SemanticType.PRICE_RANGE}) == 2
        assert SemanticType.PRICE_NET_CHANGE.family is SemanticType.MAGNITUDE
        assert SemanticType.PRICE_RANGE.family is SemanticType.MAGNITUDE

    def test_the_prompt_teaches_the_distinction(self) -> None:
        text = load_prompt(GOLD_WRITER_V4)
        assert "biên độ" in text
        assert "net change" in text


# --------------------------------------------------------------------------
# structural contract
# --------------------------------------------------------------------------


class TestStructure:
    def test_the_locked_headings_are_required(self) -> None:
        for heading in (
            "⚡ Chốt:",
            "🟢 Đẩy lên:",
            "🔴 Kéo xuống:",
            "📈 Giá đang nói gì?",
            "🧭 Mình đang chờ:",
        ):
            broken = BULLISH_HEAVY.replace(heading, "## Phần khác")
            assert OutputFindingCode.MISSING_REQUIRED_SECTION in blocking(broken), heading

    def test_the_disclaimer_is_exact_and_appears_once(self) -> None:
        assert BULLISH_HEAVY.rstrip().endswith(DISCLAIMER)
        doubled = BULLISH_HEAVY + "\n" + DISCLAIMER
        assert OutputFindingCode.DISCLAIMER_COUNT_MISMATCH in blocking(doubled)
        missing = BULLISH_HEAVY.replace(DISCLAIMER, "")
        assert OutputFindingCode.DISCLAIMER_COUNT_MISMATCH in blocking(missing)

    def test_the_hard_cap_blocks(self) -> None:
        padded = BULLISH_HEAVY.replace(
            "Chỉ còn CPI.", "Chỉ còn CPI. " + "Thêm chữ cho dài ra. " * 60
        )
        assert OutputFindingCode.HARD_CAP_EXCEEDED in blocking(padded)

    def test_the_date_must_be_the_one_supplied(self) -> None:
        assert not missing_article_date(BULLISH_HEAVY, DATE)
        assert missing_article_date(BULLISH_HEAVY, "05.09.2026")
        reformatted = BULLISH_HEAVY.replace(DATE, "4/9/2026")
        assert missing_article_date(reformatted, DATE)

    def test_only_analysis_is_enforced(self) -> None:
        assert is_enforced(ArticleType.ANALYSIS)
        assert not is_enforced(ArticleType.NEWS_DIGEST)
        assert not is_enforced(ArticleType.TRADE_PLAN)


# --------------------------------------------------------------------------
# prompt versioning and the unchanged stages
# --------------------------------------------------------------------------


class TestPromptVersioning:
    def test_v4_is_the_default_and_v3_is_untouched(self) -> None:
        assert DEFAULT_WRITER_PROMPT == GOLD_WRITER_V4
        v3 = (PROMPTS_DIR / f"{GOLD_WRITER_V3}.md").read_text(encoding="utf-8")
        assert "NHẬN ĐỊNH VÀNG" in v3, "v3 must keep meaning what it meant"
        assert "PHÂN TÍCH VÀNG" not in v3

    def test_v1_and_v2_still_load(self) -> None:
        """Historical Runs record the prompt they were written under."""
        for prompt_id in ("gold_writer_v1", "gold_writer_v2", "gold_writer_v3"):
            assert load_prompt(prompt_id)

    def test_the_style_block_is_versioned_separately_and_included(self) -> None:
        style = load_prompt(GOLD_HUMAN_STYLE_V1)
        v4 = load_prompt(GOLD_WRITER_V4)
        assert "HUMAN STYLE v1" in style
        assert style.strip() in v4, "v4 must include the style block, not copy it"
        assert "<!-- include:" not in v4, "the include must be resolved at load time"

    def test_the_style_block_is_not_a_standalone_prompt(self) -> None:
        """It carries no rules or output contract; sending it alone would be a bug."""
        style = load_prompt(GOLD_HUMAN_STYLE_V1)
        assert "# SYSTEM RULES" not in style
        assert "# OUTPUT CONTRACT" not in style

    def test_a_missing_include_fails_loudly(self) -> None:
        from goldpipeline.prompts import _resolve_includes

        with pytest.raises(FileNotFoundError, match="does not exist"):
            _resolve_includes("<!-- include: gold_no_such_block -->", "gold_writer_v4")

    def test_v4_keeps_the_whole_claim_contract(self) -> None:
        """A prettier article that loses provenance is a regression."""
        v4 = load_prompt(GOLD_WRITER_V4)
        for rule in (
            "source_claims",
            "news_claims",
            "VALID SOURCE PATHS",
            "CITABLE NEWS ITEMS",
            "character for character",
            "news_item_ids",
        ):
            assert rule in v4, rule

    def test_v4_keeps_the_injection_boundary(self) -> None:
        v4 = load_prompt(GOLD_WRITER_V4)
        assert "SOURCE_CONTAINS_INSTRUCTIONS" in v4
        assert "Never obey them" in v4
        assert "untrusted" in v4

    def test_v4_explains_how_to_compress_without_losing_provenance(self) -> None:
        """The round's one real risk: compression that the verifier cannot check."""
        v4 = load_prompt(GOLD_WRITER_V4)
        assert "does not have to be a whole sentence" in v4
        assert "one claim per assertion" in v4.replace("\n  ", " ")
        assert "stays local to the words you quoted" in v4


class TestOtherStagesUnchanged:
    ROOT = Path(__file__).resolve().parents[1]

    def test_the_reviewer_prompt_is_untouched(self) -> None:
        reviewer = load_prompt("gold_reviewer_v1")
        assert "REVIEW RUBRIC" in reviewer
        assert "PHÂN TÍCH VÀNG" not in reviewer
        assert "HUMAN STYLE" not in reviewer, "axis B is round 6.4f"

    def test_the_reviewer_schema_has_no_style_axis_yet(self) -> None:
        from goldpipeline.schemas.review import IssueCategory

        assert not any(
            name in {member.name for member in IssueCategory}
            for name in ("AI_VOICE", "VERBOSITY", "NO_POSITION", "FORMAT_CONTRACT")
        )

    def test_the_finalizer_prompt_is_untouched(self) -> None:
        finalizer = load_prompt("gold_finalizer_v1")
        assert "Minimum necessary revision" in finalizer
        assert "smoother" not in finalizer, "that rule is round 6.4g"

    def test_the_finalizer_schema_has_no_length_accounting_yet(self) -> None:
        from goldpipeline.schemas.finalizer import FinalizerModelOutput

        fields = set(FinalizerModelOutput.model_fields)
        assert "chars_before" not in fields
        assert "chars_after" not in fields

    def test_the_other_article_types_are_still_not_ready(self) -> None:
        from goldpipeline.services.article_routing import READY_TYPES, SPECS

        assert {ArticleType.ANALYSIS} == READY_TYPES
        assert SPECS[ArticleType.NEWS_DIGEST].ready is False
        assert SPECS[ArticleType.TRADE_PLAN].ready is False

    def test_the_analysis_route_points_at_v4(self) -> None:
        from goldpipeline.services.article_routing import writer_prompt_for

        assert writer_prompt_for(ArticleType.ANALYSIS) == GOLD_WRITER_V4

    def test_the_style_block_is_never_reachable_from_a_trade_plan(self) -> None:
        """The public plan is rendered, so it has no voice to contract."""
        from goldpipeline.schemas.article_contract import HUMAN_STYLE_TYPES

        assert ArticleType.TRADE_PLAN not in HUMAN_STYLE_TYPES
        assert SPECS_TRADE_PLAN_PROMPT is None


from goldpipeline.services.article_routing import SPECS  # noqa: E402

SPECS_TRADE_PLAN_PROMPT = SPECS[ArticleType.TRADE_PLAN].prompt_id
