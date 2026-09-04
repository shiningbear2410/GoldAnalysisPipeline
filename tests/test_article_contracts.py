"""The output-contract foundation: contracts, checks, vocabulary. Nothing wired.

Three things are pinned here that later rounds must not loosen:

* **one registry**, complete over the enum, with ``TRADE_PLAN`` contracted as
  a deterministic document that no writer prompt may be assigned to;
* **symptom checks that stay conservative** - every rule has a "natural text
  does not trigger" case beside its "this triggers" case;
* **semantic meaning is not provenance** - the same value, the same type,
  whichever provider produced it.

And one thing pinned in the negative: after this round, production has not
changed. The routing table, readiness and the prompts are asserted untouched.

Offline throughout.
"""

from __future__ import annotations

import ast
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from goldpipeline.schemas.article import ArticleType
from goldpipeline.schemas.article_contract import (
    CONTRACTS,
    DISCLAIMER_TEXT,
    HUMAN_STYLE_TYPES,
    TRADE_PLAN_SIDE_LABELS,
    ArticleContract,
    DisclaimerPolicy,
    GenerationMode,
    SectionKey,
    StructureKind,
    contract_for,
)
from goldpipeline.schemas.output_findings import OutputFinding, OutputFindingCode
from goldpipeline.schemas.review import Severity
from goldpipeline.services.article_contract_checks import (
    check_contract,
    check_disclaimer,
    check_length,
    check_terminology,
    detect_sections,
    detect_structures,
    length_report,
)
from goldpipeline.services.article_routing import READY_TYPES, SPECS
from goldpipeline.services.causality_language import find_causal_claims
from goldpipeline.services.numeric_mentions import (
    AuthorisedFact,
    FactProvenance,
    ResolutionStatus,
    compatible,
    extract_numeric_mentions,
    resolve_mention,
)
from goldpipeline.services.numeric_semantics import (
    REFINED_DERIVED_KIND,
    DerivedFactKind,
    KnownNumber,
    SemanticType,
)
from goldpipeline.services.style_symptoms import (
    CONNECTIVE_THRESHOLD,
    find_style_symptoms,
)

SRC = Path(__file__).resolve().parents[1] / "src" / "goldpipeline"

ANALYSIS = contract_for(ArticleType.ANALYSIS)
DIGEST = contract_for(ArticleType.NEWS_DIGEST)
PLAN = contract_for(ArticleType.TRADE_PLAN)

ANALYSIS_OK = """🕯 PHÂN TÍCH VÀNG — 03.09.2026

⚡ Chốt: tin 24h nghiêng về phía vàng, nhưng CPI vẫn treo ở đó.

🟢 Đẩy lên:
ETF mua thật, USD mất giá, và Fed chưa nói gì cứng hơn. Dòng tiền là phần đáng tin nhất.

🔴 Kéo xuống:
Chưa thấy gì đáng kể.

📈 Giá đang nói gì?
Giá bật mạnh nhất sáng nay, đúng cụm tin USD yếu. Trùng giờ thôi, chưa phải nhân quả.

🧭 Mình đang chờ:
Dữ liệu Mỹ mềm tiếp thì vàng còn đỡ. CPI nóng thì áp lực đến nhanh hơn nhiều người nghĩ.

🔴 Nhận định cá nhân, không phải lời khuyên đầu tư.
"""

DIGEST_OK = """📰 TIN VÀNG 02.09 → 03.09.2026
🕐 02/09 14:07 → 03/09 14:07 (giờ VN)

📌 Tin đáng chú ý

🔹 19:19 — Fed Williams: lợi suất tăng không phải do lạm phát
→ 🟢 Hỗ trợ vàng

🔹 07:00 — SPDR mua ròng 9.98 tấn
Dòng tiền ETF quay lại.
→ 🟢 Tích cực

🔹 07:20 — USD suy yếu 0.21%
→ 🟢 Hỗ trợ trực tiếp

📈 Giá phản ứng
Vàng đi từ 4323 lên 4428 USD, biên độ 139 USD. Nhịp mạnh nhất sáng 03/09.

🧭 Cán cân
Đang nghiêng tích cực: USD yếu, ETF mua, Fed chưa cứng rắn.
Rủi ro phía trước: NFP và CPI.

🔴 Nhận định cá nhân, không phải lời khuyên đầu tư.
"""

PLAN_OK = """🎯 CHIẾN LƯỢC VÀNG

🔴 SEO
▸ 4415 – 4417 · scalp
▸ 4435 – 4437 · vùng chính
▸ 4460 – 4462

🟢 BAI
▸ 4397 – 4400 · scalp
▸ 4368 – 4370
▸ 4323 – 4325
"""


def codes(findings: list[OutputFinding]) -> set[OutputFindingCode]:
    return {f.code for f in findings}


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------


class TestRegistry:
    def test_every_type_registered_exactly_once(self) -> None:
        assert set(CONTRACTS) == set(ArticleType)
        for kind, contract in CONTRACTS.items():
            assert contract.article_type is kind

    @pytest.mark.parametrize(
        ("kind", "hard_cap"),
        [
            (ArticleType.ANALYSIS, 1300),
            (ArticleType.NEWS_DIGEST, 1900),
            (ArticleType.TRADE_PLAN, 650),
        ],
    )
    def test_locked_hard_caps(self, kind: ArticleType, hard_cap: int) -> None:
        assert contract_for(kind).hard_max_chars == hard_cap

    def test_locked_targets(self) -> None:
        assert (ANALYSIS.target_min_chars, ANALYSIS.target_max_chars) == (600, 1000)
        assert (DIGEST.target_min_chars, DIGEST.target_max_chars) == (900, 1500)
        assert (PLAN.target_min_chars, PLAN.target_max_chars) == (200, 450)

    def test_generation_modes(self) -> None:
        assert ANALYSIS.generation_mode is GenerationMode.LLM
        assert DIGEST.generation_mode is GenerationMode.LLM
        assert PLAN.generation_mode is GenerationMode.DETERMINISTIC

    def test_disclaimer_policy_is_per_type(self) -> None:
        assert ANALYSIS.disclaimer.expected_count == 1
        assert DIGEST.disclaimer.expected_count == 1
        assert PLAN.disclaimer.expected_count == 0
        assert ANALYSIS.disclaimer.text == DISCLAIMER_TEXT == DIGEST.disclaimer.text
        assert DISCLAIMER_TEXT == "🔴 Nhận định cá nhân, không phải lời khuyên đầu tư."

    def test_human_style_applies_to_prose_types_only(self) -> None:
        assert {ArticleType.ANALYSIS, ArticleType.NEWS_DIGEST} == HUMAN_STYLE_TYPES
        assert PLAN.human_style is False

    def test_trade_plan_shape(self) -> None:
        assert PLAN.required_sections == (SectionKey.PLAN_TITLE, SectionKey.SEO, SectionKey.BAI)
        assert SectionKey.DISCLAIMER in PLAN.forbidden_sections
        assert {
            StructureKind.EXPLANATORY_PROSE,
            StructureKind.DATE_LINE,
            StructureKind.RISK_PARAMETERS,
        } <= PLAN.forbidden_structures
        assert TRADE_PLAN_SIDE_LABELS == ("SEO", "BAI")

    def test_the_three_products_answer_different_questions(self) -> None:
        questions = {c.question for c in CONTRACTS.values()}
        assert len(questions) == 3

    def test_prose_types_exclude_each_others_bodies(self) -> None:
        assert StructureKind.NEWS_ITEM_LIST in ANALYSIS.forbidden_structures
        assert StructureKind.SCENARIO_ESSAY in DIGEST.forbidden_structures
        assert StructureKind.TRADE_ZONE_LIST in ANALYSIS.forbidden_structures
        assert StructureKind.TRADE_ZONE_LIST in DIGEST.forbidden_structures

    def test_contracts_are_frozen(self) -> None:
        with pytest.raises(ValidationError):
            PLAN.hard_max_chars = 9999  # type: ignore[misc]


class TestContractValidation:
    def base(self, **overrides: object) -> dict[str, object]:
        fields: dict[str, object] = {
            "article_type": ArticleType.ANALYSIS,
            "question": "q",
            "generation_mode": GenerationMode.LLM,
            "human_style": True,
            "target_min_chars": 10,
            "target_max_chars": 20,
            "hard_max_chars": 30,
            "required_sections": (SectionKey.VERDICT, SectionKey.DISCLAIMER),
            "disclaimer": DisclaimerPolicy(expected_count=1),
        }
        fields.update(overrides)
        return fields

    def test_deterministic_documents_have_no_voice_to_judge(self) -> None:
        with pytest.raises(ValidationError, match="human-style"):
            ArticleContract(
                **self.base(generation_mode=GenerationMode.DETERMINISTIC, human_style=True)
            )

    def test_length_order(self) -> None:
        with pytest.raises(ValidationError, match="target_min"):
            ArticleContract(**self.base(hard_max_chars=15))

    def test_disclaimer_policy_and_sections_must_agree(self) -> None:
        with pytest.raises(ValidationError, match="disclaimer"):
            ArticleContract(**self.base(disclaimer=DisclaimerPolicy(expected_count=0)))

    def test_required_and_forbidden_disjoint(self) -> None:
        with pytest.raises(ValidationError, match="both required and forbidden"):
            ArticleContract(**self.base(forbidden_sections=frozenset({SectionKey.VERDICT})))

    def test_at_most_one_disclaimer_is_expressible(self) -> None:
        with pytest.raises(ValidationError):
            DisclaimerPolicy(expected_count=2)


# --------------------------------------------------------------------------
# routing consistency and no production change
# --------------------------------------------------------------------------


class TestRoutingConsistency:
    def test_a_writer_prompt_implies_an_llm_contract(self) -> None:
        for kind, spec in SPECS.items():
            if spec.prompt_id is not None:
                assert contract_for(kind).generation_mode is GenerationMode.LLM, kind

    def test_trade_plan_can_never_have_a_writer_prompt(self) -> None:
        assert PLAN.generation_mode is GenerationMode.DETERMINISTIC
        assert SPECS[ArticleType.TRADE_PLAN].prompt_id is None

    def test_readiness_matches_the_prompts_that_exist(self) -> None:
        """A type is ready exactly when something can write it.

        Round 6.5b gave NEWS_DIGEST its own prompt and turned it on. TRADE_PLAN
        still has neither, and a readiness flag that ran ahead of a prompt is
        the failure this assertion exists to catch.
        """
        assert {ArticleType.ANALYSIS, ArticleType.NEWS_DIGEST} == READY_TYPES
        assert SPECS[ArticleType.NEWS_DIGEST].prompt_id is not None
        assert SPECS[ArticleType.TRADE_PLAN].ready is False
        assert SPECS[ArticleType.TRADE_PLAN].prompt_id is None

    def test_only_the_analysis_contract_wires_the_checks_into_production(self) -> None:
        """Round 6.4e connected these, and only through one seam.

        They were unused when this file was written; the ANALYSIS writer now
        enforces its own contract. What the test still pins is the *shape* of
        that dependency: exactly one service composes the checks, and the
        writer reaches them only through it. A second call site appearing
        elsewhere is how a narrow rule quietly becomes a global gate.

        Round 6.4f split the original set in two rather than adding modules to
        an allowlist. The distinction is what the guard was always protecting:
        a module that *decides whether an article is acceptable* must be
        reachable from one place, while a vocabulary of section names or a
        counter of observable symptoms decides nothing and may be read by
        anyone who needs to name a thing. The reviewer reads the second group -
        it labels where a style finding sits, and it receives symptoms as
        hints - and must never touch the first, because a second enforcer is
        exactly the failure this test exists to catch.
        """
        enforcing = {
            "goldpipeline.services.article_contract_checks",
            "goldpipeline.services.causality_language",
            "goldpipeline.services.digest_provenance",
            "goldpipeline.services.numeric_mentions",
            "goldpipeline.services.prose",
        }
        vocabulary = {
            "goldpipeline.schemas.article_contract",
            "goldpipeline.schemas.output_findings",
            "goldpipeline.services.style_symptoms",
        }
        own = {name.rsplit(".", 1)[1] for name in enforcing | vocabulary}

        # Round 6.4g added a second enforcement point, and says so here rather
        # than smuggling it in. `final_postcheck` judges the article the
        # *finalizer* produced, which the writer's composer never sees - a
        # different stage, a different article, a different question. What the
        # guard still forbids is a *third* one appearing without a decision:
        # two named enforcers are an architecture, and an open list is not.
        # Round 6.5c.1 added the third, and it is named here for the same
        # reason. `digest_provenance` decides whether a 🧭 Cán cân paragraph
        # outran its evidence, and `digest_writer` is the only module allowed
        # to ask it - the digest's `analysis_contract`. The digest reviewer
        # deliberately does not appear in this set: it is *handed* a
        # `DigestPrecheckReport`, exactly as `build_reviewer_prompt` is handed
        # a `PrecheckReport`, so that the answer shown to a model and the
        # answer that rejects a response are the same answer.
        enforcers = {"analysis_contract", "digest_writer", "final_postcheck"}

        # What is forbidden is *running a check*, not touching the module that
        # holds one. `detect_sections` and `detect_structures` are parsers: they
        # answer "where is the price section", decide nothing, and both the
        # postcheck and the offline finalizer need one to tell a rewritten
        # section from an untouched one. Anything named `check_*` is the part
        # that judges, and that stays behind a named enforcer.
        parsers = {"detect_sections", "detect_structures", "length_report", "count_disclaimers"}

        for path in SRC.rglob("*.py"):
            if path.stem in own or path.stem in enforcers:
                continue
            imported = _imports(path)
            judging = {
                name
                for name in imported
                if name.rsplit(".", 1)[0] in enforcing
                and name.rsplit(".", 1)[1] not in parsers
                and name not in enforcing
            }
            assert not judging, f"{path.name} runs a check directly: {judging}"
            modules = imported & enforcing
            for module in modules:
                used = {
                    name.rsplit(".", 1)[1] for name in imported if name.startswith(f"{module}.")
                }
                assert used and used <= parsers, (
                    f"{path.name} imports {module} for more than parsing: {used - parsers}"
                )

        writer = _imports(SRC / "services" / "writer.py")
        assert "goldpipeline.services.analysis_contract" in writer
        assert not (writer & enforcing), "the writer must go through analysis_contract"

        # The finalizer's postcheck goes through the same composer for the
        # contract itself, and reaches past it for exactly two things: the
        # section parser, which tells a rewritten section from an untouched one,
        # and the numeric resolution seam, which is the one vocabulary for
        # "does this number mean what the article says it means". Pinned as an
        # exact set so a third dependency is a decision rather than a drift.
        postcheck = _imports(SRC / "services" / "final_postcheck.py")
        assert "goldpipeline.services.analysis_contract" in postcheck
        assert postcheck & enforcing == {
            "goldpipeline.services.article_contract_checks",
            "goldpipeline.services.numeric_mentions",
        }

        # The reviewer reads vocabulary and hints, and enforces nothing.
        for module in ("reviewer.py", "reviewer_prompt.py", "style_review.py", "review_action.py"):
            imports = _imports(SRC / "services" / module)
            assert not (imports & enforcing), f"{module} must not import an enforcing check"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


# --------------------------------------------------------------------------
# market-data agnosticism
# --------------------------------------------------------------------------


class TestMarketDataAgnosticism:
    NEW_MODULES = (
        "schemas/article_contract.py",
        "schemas/output_findings.py",
        "services/article_contract_checks.py",
        "services/style_symptoms.py",
        "services/causality_language.py",
        "services/numeric_mentions.py",
        "services/numeric_semantics.py",
        "services/prose.py",
    )

    @pytest.mark.parametrize("relative", NEW_MODULES)
    def test_no_adapter_or_provider_imports(self, relative: str) -> None:
        imported = _imports(SRC / relative)
        offenders = {
            name
            for name in imported
            if name.startswith("goldpipeline.adapters")
            or "mt5" in name.lower()
            or "tradingview" in name.lower()
            or "MetaTrader" in name
        }
        assert not offenders, offenders

    def test_contract_fields_are_plain(self) -> None:
        for name, field in ArticleContract.model_fields.items():
            text = str(field.annotation)
            assert "market" not in text.lower(), name
            assert "context" not in text.lower(), name

    def test_semantic_types_carry_no_provider(self) -> None:
        for member in SemanticType:
            assert not any(word in member.name for word in ("MT5", "TRADINGVIEW", "OANDA"))

    @pytest.mark.parametrize(
        "semantic",
        [SemanticType.ABSOLUTE_PRICE, SemanticType.PRICE_NET_CHANGE, SemanticType.PRICE_RANGE],
    )
    def test_meaning_does_not_change_with_provenance(self, semantic: SemanticType) -> None:
        value = Decimal("4323.50")
        mt5 = AuthorisedFact(value, semantic, "x", FactProvenance("mt5", "XAUUSD", "M15"))
        tv = AuthorisedFact(value, semantic, "x", FactProvenance("tradingview", "OANDA:XAUUSD"))
        assert mt5.semantic is tv.semantic is semantic
        assert mt5.as_known() == tv.as_known()
        mention = extract_numeric_mentions("giá 4323.50")[0]
        assert resolve_mention(mention, [mt5]).status is ResolutionStatus.RESOLVED
        assert resolve_mention(mention, [tv]).status is ResolutionStatus.RESOLVED


# --------------------------------------------------------------------------
# contract checks
# --------------------------------------------------------------------------


class TestContractChecks:
    def test_the_locked_shapes_pass_their_own_contracts(self) -> None:
        assert check_contract(ANALYSIS_OK, ANALYSIS) == []
        assert check_contract(DIGEST_OK, DIGEST) == []
        assert check_contract(PLAN_OK, PLAN) == []

    def test_each_shape_fails_the_other_contracts(self) -> None:
        assert codes(check_contract(DIGEST_OK, ANALYSIS)) >= {
            OutputFindingCode.MISSING_REQUIRED_SECTION,
            OutputFindingCode.FORBIDDEN_SECTION_PRESENT,
            OutputFindingCode.FORBIDDEN_STRUCTURE_PRESENT,
        }
        assert OutputFindingCode.FORBIDDEN_STRUCTURE_PRESENT in codes(
            check_contract(PLAN_OK, DIGEST)
        )
        plan_findings = check_contract(ANALYSIS_OK, PLAN)
        assert OutputFindingCode.DISCLAIMER_COUNT_MISMATCH in codes(plan_findings)
        assert OutputFindingCode.FORBIDDEN_STRUCTURE_PRESENT in codes(plan_findings)

    def test_hard_cap_is_the_only_length_finding(self) -> None:
        assert check_length("x" * PLAN.hard_max_chars, PLAN) == []
        over = check_length("x" * (PLAN.hard_max_chars + 1), PLAN)
        assert codes(over) == {OutputFindingCode.HARD_CAP_EXCEEDED}
        assert over[0].severity is Severity.HIGH

    def test_a_short_article_is_not_a_finding(self) -> None:
        short = "⚡ Chốt: chưa có gì mới."
        assert check_length(short, ANALYSIS) == []
        report = length_report(short, ANALYSIS)
        assert not report.within_target and not report.over_hard_cap

    def test_analysis_rejects_a_news_item_list(self) -> None:
        polluted = ANALYSIS_OK.replace(
            "Chưa thấy gì đáng kể.",
            "🔹 07:00 — SPDR mua ròng\n🔹 07:20 — USD suy yếu",
        )
        found = check_contract(polluted, ANALYSIS)
        assert OutputFindingCode.FORBIDDEN_STRUCTURE_PRESENT in codes(found)
        assert any("NEWS_ITEM_LIST" in f.message for f in found)

    def test_analysis_rejects_trade_zones(self) -> None:
        polluted = ANALYSIS_OK + "\n▸ 4415 – 4417\n▸ 4435 – 4437\n"
        found = check_contract(polluted, ANALYSIS)
        assert any("TRADE_ZONE_LIST" in f.message for f in found)

    def test_digest_rejects_a_scenario_essay(self) -> None:
        polluted = DIGEST_OK.replace(
            "Rủi ro phía trước: NFP và CPI.",
            "Nếu CPI nóng thì vàng chịu áp lực. Nếu dữ liệu mềm thì vàng còn đỡ.",
        )
        found = check_contract(polluted, DIGEST)
        assert any("SCENARIO_ESSAY" in f.message for f in found)

    def test_digest_window_is_recognised_by_shape(self) -> None:
        assert SectionKey.DIGEST_WINDOW in detect_sections(DIGEST_OK)
        assert SectionKey.DIGEST_WINDOW not in detect_sections(ANALYSIS_OK)

    def test_missing_required_section_is_named(self) -> None:
        without = ANALYSIS_OK.replace("📈 Giá đang nói gì?\n", "")
        found = check_contract(without, ANALYSIS)
        assert any(
            f.code is OutputFindingCode.MISSING_REQUIRED_SECTION and "PRICE_READ" in f.message
            for f in found
        )


class TestTradePlanTerminology:
    def test_seo_bai_are_valid(self) -> None:
        assert check_terminology(PLAN_OK, PLAN) == []
        sections = detect_sections(PLAN_OK)
        assert SectionKey.SEO in sections and SectionKey.BAI in sections

    def test_sell_buy_substitution_is_a_finding(self) -> None:
        normalised = PLAN_OK.replace("SEO", "SELL").replace("BAI", "BUY")
        found = check_contract(normalised, PLAN)
        terminology = [f for f in found if f.code is OutputFindingCode.FORBIDDEN_TERMINOLOGY]
        assert len(terminology) == 2
        assert all(f.severity is Severity.HIGH for f in terminology)
        assert OutputFindingCode.MISSING_REQUIRED_SECTION in codes(found)

    def test_terminology_rule_is_scoped_to_the_plan(self) -> None:
        assert check_terminology("Bên mua (buy side) đang thắng.", ANALYSIS) == []

    def test_plan_rejects_prose_dates_and_risk_parameters(self) -> None:
        polluted = PLAN_OK.replace(
            "▸ 4460 – 4462",
            "▸ 4460 – 4462 · SL 4470 TP 4420\n"
            "Vùng này quan trọng vì giá đã phản ứng nhiều lần trong tuần qua và khối lượng tăng.\n"
            "Ngày 03/09/2026",
        )
        found = detect_structures(polluted)
        assert {
            StructureKind.RISK_PARAMETERS,
            StructureKind.EXPLANATORY_PROSE,
            StructureKind.DATE_LINE,
        } <= set(found)
        assert len(check_contract(polluted, PLAN)) >= 3


class TestDisclaimers:
    def test_analysis_expects_exactly_one(self) -> None:
        assert check_disclaimer(ANALYSIS_OK, ANALYSIS) == []
        doubled = ANALYSIS_OK + DISCLAIMER_TEXT + "\n"
        found = check_disclaimer(doubled, ANALYSIS)
        assert codes(found) == {OutputFindingCode.DISCLAIMER_COUNT_MISMATCH}
        assert found[0].count == 2 and found[0].threshold == 1
        missing = ANALYSIS_OK.replace(DISCLAIMER_TEXT, "")
        assert codes(check_disclaimer(missing, ANALYSIS)) == {
            OutputFindingCode.DISCLAIMER_COUNT_MISMATCH
        }

    def test_digest_expects_exactly_one(self) -> None:
        assert check_disclaimer(DIGEST_OK, DIGEST) == []
        missing = DIGEST_OK.replace(DISCLAIMER_TEXT, "")
        assert codes(check_disclaimer(missing, DIGEST)) == {
            OutputFindingCode.DISCLAIMER_COUNT_MISMATCH
        }

    def test_trade_plan_expects_zero(self) -> None:
        assert check_disclaimer(PLAN_OK, PLAN) == []
        with_one = PLAN_OK + "\n" + DISCLAIMER_TEXT + "\n"
        found = check_disclaimer(with_one, PLAN)
        assert codes(found) == {OutputFindingCode.DISCLAIMER_COUNT_MISMATCH}
        assert found[0].count == 1 and found[0].threshold == 0

    def test_a_paraphrased_disclaimer_still_counts(self) -> None:
        paraphrased = ANALYSIS_OK.replace(
            DISCLAIMER_TEXT, "Đây là nhận định cá nhân, không phải lời khuyên đầu tư nhé."
        )
        assert codes(check_disclaimer(paraphrased, ANALYSIS)) == {
            OutputFindingCode.DISCLAIMER_TEXT_MISMATCH
        }

    def test_there_is_no_global_disclaimer_invariant(self) -> None:
        assert {c.disclaimer.expected_count for c in CONTRACTS.values()} == {0, 1}


# --------------------------------------------------------------------------
# style symptoms
# --------------------------------------------------------------------------


class TestStyleSymptoms:
    def test_natural_varied_text_does_not_trigger(self) -> None:
        assert find_style_symptoms(ANALYSIS_OK, contract=ANALYSIS) == []
        assert find_style_symptoms(DIGEST_OK, contract=DIGEST) == []

    def test_one_connective_is_vietnamese(self) -> None:
        text = "Vàng tăng nhẹ. Tuy nhiên, USD vẫn mạnh. Đáng chú ý là ETF đã quay lại mua."
        assert OutputFindingCode.CONNECTIVE_DENSITY not in codes(
            find_style_symptoms(text, contract=ANALYSIS)
        )

    def test_connective_density_triggers(self) -> None:
        text = (
            "Vàng tăng nhẹ. Tuy nhiên, USD vẫn mạnh. Bên cạnh đó, ETF mua vào. "
            "Do đó, xu hướng ngắn hạn nghiêng lên. Ngoài ra, CPI sắp ra."
        )
        found = [
            f
            for f in find_style_symptoms(text, contract=ANALYSIS)
            if f.code is OutputFindingCode.CONNECTIVE_DENSITY
        ]
        assert len(found) == 1
        assert found[0].count == 4 and found[0].threshold == CONNECTIVE_THRESHOLD

    def test_throat_clearing(self) -> None:
        text = "Trong bài viết này, mình sẽ nhìn lại phiên hôm qua. Vàng tăng."
        found = find_style_symptoms(text, contract=ANALYSIS)
        assert OutputFindingCode.THROAT_CLEARING in codes(found)

    def test_repeated_construction(self) -> None:
        text = (
            "Vàng không chỉ tăng mà còn giữ được đà. "
            "USD không chỉ yếu mà còn yếu thêm. "
            "Đà này chưa dừng."
        )
        assert OutputFindingCode.REPEATED_CONSTRUCTION in codes(
            find_style_symptoms(text, contract=ANALYSIS)
        )
        once = "Vàng không chỉ tăng mà còn giữ được đà. USD yếu thêm."
        assert OutputFindingCode.REPEATED_CONSTRUCTION not in codes(
            find_style_symptoms(once, contract=ANALYSIS)
        )

    def test_uniform_bullets(self) -> None:
        text = "\n".join(
            [
                "🟢 USD suy yếu rõ rệt — hỗ trợ vàng trong ngắn hạn này",
                "🟢 ETF mua ròng mạnh tay — dòng tiền thật đã quay trở lại",
                "🟢 Fed chưa cứng rắn thêm — lợi suất chưa gây áp lực gì",
                "🟢 CPI vẫn đang treo đó — rủi ro chính của tuần này rồi",
            ]
        )
        found = find_style_symptoms(text, contract=ANALYSIS)
        assert OutputFindingCode.BULLET_SHAPE_UNIFORM in codes(found)

    def test_uniform_bullets_are_correct_in_a_rendered_plan(self) -> None:
        """The plan has no author; its rows are supposed to be identical."""
        assert find_style_symptoms(PLAN_OK, contract=PLAN) == []
        uniform = "\n".join(
            [f"▸ {4400 + i * 20} – {4402 + i * 20} · vùng canh scalp trong phiên" for i in range(6)]
        )
        assert find_style_symptoms(uniform, contract=PLAN) == []

    def test_digest_item_lines_are_not_prose_bullets(self) -> None:
        """Time-stamped item lines are uniform by contract, not by habit."""
        items = "\n".join(
            [f"🔹 0{i}:00 — Tin thứ {i} về vàng hôm nay ra lúc sáng" for i in range(1, 6)]
        )
        assert OutputFindingCode.BULLET_SHAPE_UNIFORM not in codes(
            find_style_symptoms(items, contract=DIGEST)
        )

    def test_sentence_rhythm(self) -> None:
        uniform = " ".join(["Vàng tăng nhẹ trong phiên sáng nay."] * 6)
        assert OutputFindingCode.SENTENCE_LENGTH_UNIFORM in codes(
            find_style_symptoms(uniform, contract=ANALYSIS)
        )

    def test_duplicate_statement(self) -> None:
        text = (
            "Bên mua đang chiếm ưu thế nhưng câu chuyện chưa một chiều.\n"
            "USD yếu.\n"
            "Bên mua đang chiếm ưu thế, câu chuyện vẫn chưa một chiều."
        )
        assert OutputFindingCode.DUPLICATE_STATEMENT in codes(
            find_style_symptoms(text, contract=ANALYSIS)
        )

    def test_number_restated(self) -> None:
        text = "Vàng chạm 4323. Mốc 4323 giữ được. Nếu mất 4323 thì tính lại."
        found = [
            f
            for f in find_style_symptoms(text, contract=ANALYSIS)
            if f.code is OutputFindingCode.NUMBER_RESTATED
        ]
        assert len(found) == 1 and found[0].count == 3

    def test_times_do_not_count_as_restated_numbers(self) -> None:
        text = "Lúc 14:07 giá bật. Lúc 14:07 USD rơi. Đến 14:07 mọi thứ đã rõ."
        assert OutputFindingCode.NUMBER_RESTATED not in codes(
            find_style_symptoms(text, contract=DIGEST)
        )

    def test_repeated_opener_against_prior_articles(self) -> None:
        today = "🕯 PHÂN TÍCH VÀNG\n\nChốt lại trong một câu: vàng vẫn được đỡ hôm nay."
        yesterday = "🕯 PHÂN TÍCH VÀNG\n\nChốt lại trong một câu: vàng đang yếu đi rõ."
        found = find_style_symptoms(today, contract=ANALYSIS, prior_openings=[yesterday])
        assert OutputFindingCode.REPEATED_OPENER in codes(found)
        different = "🕯 PHÂN TÍCH VÀNG\n\nTin 24h qua nghiêng về phía vàng, nhưng CPI vẫn treo."
        assert find_style_symptoms(today, contract=ANALYSIS, prior_openings=[different]) == []

    def test_severity_is_data_only(self) -> None:
        text = "Trong bài viết này, mình sẽ nhìn lại phiên hôm qua."
        found = find_style_symptoms(text, contract=ANALYSIS)
        assert all(isinstance(f.severity, Severity) for f in found)
        assert all(f.threshold is not None for f in found)


# --------------------------------------------------------------------------
# causality
# --------------------------------------------------------------------------


class TestCausalityLanguage:
    @pytest.mark.parametrize(
        "text",
        [
            "Giá tăng sau thời điểm tin Fed xuất hiện.",
            "Nhịp tăng của vàng trùng thời điểm ETF mua vào.",
            "Giá không đi ngược câu chuyện USD yếu.",
            "Tin gì đang kéo vàng lên?",
            "Lãi suất tăng khiến chi phí vay tăng.",
        ],
    )
    def test_allowed_wording(self, text: str) -> None:
        assert find_causal_claims(text) == []

    @pytest.mark.parametrize(
        "text",
        [
            "Tin Fed khiến vàng tăng mạnh.",
            "Do USD yếu nên vàng bật lên.",
            "Dữ liệu CPI nóng làm giá vàng giảm sâu.",
            "Nguyên nhân vàng giảm là lợi suất tăng.",
            "USD yếu đẩy vàng lên đỉnh tuần.",
        ],
    )
    def test_unattributed_causal_claims(self, text: str) -> None:
        found = find_causal_claims(text)
        assert codes(found) == {OutputFindingCode.UNATTRIBUTED_CAUSAL_CLAIM}
        assert found[0].severity is Severity.HIGH
        assert found[0].excerpt == text

    @pytest.mark.parametrize(
        "text",
        [
            "Theo Reuters, vàng tăng do USD yếu.",
            "Kitco cho rằng tin CPI khiến vàng giảm.",
            "Chiến lược gia của UBS nhận định rằng ETF mua ròng đẩy vàng lên.",
        ],
    )
    def test_attributed_claims_are_reported_speech(self, text: str) -> None:
        assert find_causal_claims(text) == []

    def test_theo_doi_is_not_attribution(self) -> None:
        text = "Theo dõi kỹ: tin Fed khiến vàng tăng."
        assert len(find_causal_claims(text)) == 1

    def test_no_co_location_no_finding(self) -> None:
        text = "Fed giữ nguyên lãi suất, nguyên nhân là lạm phát.\nVàng tăng mạnh trong phiên."
        assert find_causal_claims(text) == []

    def test_the_locked_shapes_carry_no_causal_claims(self) -> None:
        assert find_causal_claims(ANALYSIS_OK) == []
        assert find_causal_claims(DIGEST_OK) == []

    def test_findings_are_per_sentence(self) -> None:
        text = "Tin Fed khiến vàng tăng. Sau đó USD yếu làm vàng tăng thêm."
        assert len(find_causal_claims(text)) == 2


# --------------------------------------------------------------------------
# numeric semantics
# --------------------------------------------------------------------------


class TestNumericVocabulary:
    def test_price_net_change_and_range_are_distinct(self) -> None:
        distinct = {
            SemanticType.ABSOLUTE_PRICE,
            SemanticType.PRICE_NET_CHANGE,
            SemanticType.PRICE_RANGE,
        }
        assert len(distinct) == 3
        assert SemanticType.PRICE_NET_CHANGE.family is SemanticType.MAGNITUDE
        assert SemanticType.PRICE_RANGE.family is SemanticType.MAGNITUDE
        assert SemanticType.ABSOLUTE_PRICE.family is SemanticType.ABSOLUTE_PRICE
        assert not compatible(SemanticType.PRICE_NET_CHANGE, SemanticType.PRICE_RANGE)
        assert not compatible(SemanticType.PRICE_RANGE, SemanticType.PRICE_NET_CHANGE)
        assert compatible(SemanticType.MAGNITUDE, SemanticType.PRICE_RANGE)

    def test_coarse_members_keep_identity_semantics(self) -> None:
        """The production scanner's behaviour is unchanged by the refinements."""
        coarse = [m for m in SemanticType if m.family is m]
        assert {m.name for m in coarse} >= {
            "ABSOLUTE_PRICE",
            "MAGNITUDE",
            "PERCENTAGE",
            "NON_MARKET_NUMBER",
            "UNKNOWN_PRICE_LIKE",
        }
        for a in coarse:
            for b in coarse:
                known = KnownNumber(Decimal(1), a, "x")
                assert known.satisfies(b) is (a is b)

    def test_refined_satisfies_coarse_but_not_sibling(self) -> None:
        known = KnownNumber(Decimal("139"), SemanticType.PRICE_RANGE, "derived:WINDOW_RANGE")
        assert known.satisfies(SemanticType.MAGNITUDE)
        assert not known.satisfies(SemanticType.PRICE_NET_CHANGE)
        assert not known.satisfies(SemanticType.ABSOLUTE_PRICE)

    def test_every_derived_kind_has_a_refined_meaning(self) -> None:
        assert set(REFINED_DERIVED_KIND) == set(DerivedFactKind)
        assert REFINED_DERIVED_KIND[DerivedFactKind.NET_CHANGE] is SemanticType.PRICE_NET_CHANGE
        assert REFINED_DERIVED_KIND[DerivedFactKind.WINDOW_RANGE] is SemanticType.PRICE_RANGE
        assert (
            REFINED_DERIVED_KIND[DerivedFactKind.NET_CHANGE_PERCENT] is SemanticType.PERCENT_CHANGE
        )

    def test_locked_vocabulary_present(self) -> None:
        names = {m.name for m in SemanticType}
        assert names >= {
            "ABSOLUTE_PRICE",
            "PRICE_NET_CHANGE",
            "PRICE_RANGE",
            "PERCENT_CHANGE",
            "QUANTITY",
            "MASS_TONNES",
            "COUNT",
            "MONETARY_NON_PRICE",
            "DATE_TIME",
            "UNKNOWN",
        }


class TestNumericMentions:
    def by_literal(self, text: str) -> dict[str, SemanticType]:
        return {m.literal: m.semantic for m in extract_numeric_mentions(text)}

    def test_tonnes_are_not_a_price(self) -> None:
        assert self.by_literal("SPDR mua ròng 9.98 tấn") == {"9.98": SemanticType.MASS_TONNES}

    def test_percent(self) -> None:
        assert self.by_literal("USD suy yếu 0.21%") == {"0.21": SemanticType.PERCENT_CHANGE}

    def test_millions_are_not_automatically_a_quote(self) -> None:
        assert self.by_literal("dòng vốn 141 triệu USD") == {"141": SemanticType.MONETARY_NON_PRICE}
        assert self.by_literal("khoảng 141 triệu") == {"141": SemanticType.QUANTITY}

    def test_a_bare_quote_needs_vouching(self) -> None:
        assert self.by_literal("vàng đang ở 4323") == {"4323": SemanticType.UNKNOWN_PRICE_LIKE}
        assert self.by_literal("vàng đang ở 4323 USD") == {"4323": SemanticType.UNKNOWN_PRICE_LIKE}
        assert self.by_literal("một con số 9.98 lơ lửng") == {"9.98": SemanticType.UNKNOWN}

    def test_timestamps_are_dates_not_prices(self) -> None:
        found = extract_numeric_mentions("lúc 14:07 ngày 03/09/2026, năm 2026")
        assert {m.semantic for m in found} == {SemanticType.DATE_TIME}
        assert [m.literal for m in found] == ["14:07", "03/09/2026", "2026"]

    def test_distances_and_counts(self) -> None:
        assert self.by_literal("tăng 105 điểm sau 3 phiên") == {
            "105": SemanticType.MAGNITUDE,
            "3": SemanticType.COUNT,
        }

    def test_vietnamese_thousand_separator(self) -> None:
        assert extract_numeric_mentions("giá 4.323,5")[0].value == Decimal("4323.5")

    def test_digest_numbers_classify_as_written(self) -> None:
        found = self.by_literal(DIGEST_OK)
        assert found["9.98"] is SemanticType.MASS_TONNES
        assert found["0.21"] is SemanticType.PERCENT_CHANGE
        assert found["4323"] is SemanticType.UNKNOWN_PRICE_LIKE
        assert found["139"] is SemanticType.UNKNOWN_PRICE_LIKE
        assert found["19:19"] is SemanticType.DATE_TIME


class TestResolutionSeam:
    def test_tonnes_are_not_resolved_by_a_price(self) -> None:
        mention = extract_numeric_mentions("SPDR mua 9.98 tấn")[0]
        price = AuthorisedFact(Decimal("9.98"), SemanticType.ABSOLUTE_PRICE, "bar")
        assert resolve_mention(mention, [price]).status is ResolutionStatus.TYPE_MISMATCH
        tonnes = AuthorisedFact(
            Decimal("9.98"), SemanticType.MASS_TONNES, "news:spdr", FactProvenance("news")
        )
        resolved = resolve_mention(mention, [price, tonnes])
        assert resolved.status is ResolutionStatus.RESOLVED and resolved.fact is tonnes

    def test_a_bare_quote_takes_the_facts_meaning(self) -> None:
        mention = extract_numeric_mentions("vàng ở 4323")[0]
        assert mention.semantic is SemanticType.UNKNOWN_PRICE_LIKE
        fact = AuthorisedFact(Decimal("4323.00"), SemanticType.ABSOLUTE_PRICE, "bars[3].close")
        resolved = resolve_mention(mention, [fact])
        assert resolved.status is ResolutionStatus.RESOLVED
        assert resolved.fact is not None and resolved.fact.semantic is SemanticType.ABSOLUTE_PRICE
        # Integer literals get no rounding latitude - the scanner's rule, reused.
        near = AuthorisedFact(Decimal("4323.40"), SemanticType.ABSOLUTE_PRICE, "bars[3].close")
        assert resolve_mention(mention, [near]).status is ResolutionStatus.UNRESOLVED

    def test_net_change_is_not_resolved_by_range(self) -> None:
        mention = extract_numeric_mentions("tăng 105 điểm")[0]
        assert mention.semantic is SemanticType.MAGNITUDE
        rng = AuthorisedFact(Decimal("105"), SemanticType.PRICE_RANGE, "derived:WINDOW_RANGE")
        assert resolve_mention(mention, [rng]).status is ResolutionStatus.RESOLVED
        # A future check that reads "tăng" as a net change must not accept a range.
        assert not compatible(SemanticType.PRICE_NET_CHANGE, rng.semantic)

    def test_an_unexplained_number_is_unresolved_not_preserved(self) -> None:
        """The future rule: a wrong draft number is removable, never sacred."""
        mention = extract_numeric_mentions("vàng ở 4999")[0]
        assert resolve_mention(mention, []).status is ResolutionStatus.UNRESOLVED

    def test_dates_are_not_fact_claims(self) -> None:
        mention = extract_numeric_mentions("lúc 14:07")[0]
        assert resolve_mention(mention, []).status is ResolutionStatus.NOT_A_FACT_CLAIM
