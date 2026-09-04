"""The 🧭 Cán cân provenance rule, case by case.

Round 6.5c.1. The defect this covers is not hypothetical: the single live digest
Round 6.5b produced summarised an item's "mua ròng 9.98 tấn" as "mua ròng gần 10
tấn", and every check in the pipeline passed it. The evidence was in the item,
the statement was in the article, and nothing compared the two - because
`news_provenance.verify` deliberately checks each against its own source and
never asks whether one entails the other.

The matrix below is arranged around the distinction that matters: a quantity the
digest already holds may be restated, and a quantity nothing holds may not -
however close it is to one that does. Nearness is exactly the judgement this
layer must not make, so 10 against 9.98 fails for the same reason 141 does.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from goldpipeline.domain.errors import WriterResponseError
from goldpipeline.schemas.common import Timeframe
from goldpipeline.schemas.digest import (
    DigestWindow,
    MarketActivity,
    PriceReaction,
    PriceReference,
)
from goldpipeline.schemas.news_digest import (
    DigestEditorial,
    DigestItem,
    DigestSourceItem,
    ImpactMarker,
)
from goldpipeline.schemas.provenance import ClaimVerdict
from goldpipeline.schemas.writer import NewsClaim, WriterStatus
from goldpipeline.services.digest_context import build_digest_facts
from goldpipeline.services.digest_provenance import (
    EXEMPT_SEMANTICS,
    authorised_quantities,
    unsupported_balance_numbers,
)
from goldpipeline.services.digest_writer import digest_precheck, validate_editorial
from goldpipeline.services.numeric_semantics import SemanticType

RUN_ID = "20260904_060000_abcdef"
SYMBOL = "XAUUSD"
END = datetime(2026, 9, 4, 6, 0, tzinfo=UTC)
WINDOW = DigestWindow.ending_at(END, timedelta(hours=6))


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------


def source(index: int, text: str) -> DigestSourceItem:
    return DigestSourceItem(
        item_id=f"goldnewsvn:{900 + index}",
        published_at=END - timedelta(minutes=30),
        text=text,
    )


SOURCES = (
    source(1, "Fed Williams: lợi suất tăng không phản ánh kỳ vọng lạm phát cao hơn."),
    source(2, "Chỉ số USD giảm 0.21% trong phiên."),
    source(3, "SPDR Gold Trust mua ròng 9.98 tấn."),
    source(4, "Tổng nắm giữ của quỹ đạt 1096 tấn, cao nhất kể từ 2010."),
)


def reaction(*, net: str = "-50.24", span: str = "61.30") -> PriceReaction:
    """A measured window. Negative by default: the live one fell."""
    start = Decimal("4000")
    end_close = start + Decimal(net)
    low = min(start, end_close) - Decimal("1")
    return PriceReaction(
        window=WINDOW,
        symbol=SYMBOL,
        timeframe=Timeframe.M5,
        market_activity=MarketActivity.NORMAL,
        start_reference=PriceReference(
            candle_open_at=WINDOW.start - timedelta(minutes=5),
            candle_close_at=WINDOW.start,
            close=start,
        ),
        end_reference=PriceReference(
            candle_open_at=WINDOW.end - timedelta(minutes=5),
            candle_close_at=WINDOW.end,
            close=end_close,
        ),
        window_high=low + Decimal(span),
        window_low=low,
        net_change=Decimal(net),
        price_range=Decimal(span),
        percent_change=(Decimal(net) / start) * Decimal(100),
        closed_bars_in_window=72,
        overlapping_bars=72,
    )


def facts(*, sources: tuple[DigestSourceItem, ...] = SOURCES, net: str = "-50.24"):  # type: ignore[no-untyped-def]
    return build_digest_facts(
        window=WINDOW,
        price_reaction=reaction(net=net),
        symbol=SYMBOL,
        timeframe=Timeframe.M5,
        news_items=sources,
    )


def check(balance: str, *, sources: tuple[DigestSourceItem, ...] = SOURCES) -> list[str]:
    """Run the rule the way production runs it, shell included."""
    prepared = facts(sources=sources)
    return unsupported_balance_numbers(balance, sources, WINDOW, prepared.deterministic_lines)


# --------------------------------------------------------------------------
# §21 A-O: what the balance may and may not quantify
# --------------------------------------------------------------------------


def test_a_a_figure_written_exactly_as_the_item_wrote_it_is_allowed() -> None:
    assert check("SPDR mua ròng 9.98 tấn, đủ để giữ tin nghiêng tích cực.") == []


def test_b_the_live_defect_rounding_9_98_up_to_10_is_refused() -> None:
    """The Round 6.5b balance, verbatim. It must not survive this rule."""
    assert check("SPDR mua ròng gần 10 tấn, USD yếu thêm 0.21%, trong 6 giờ qua.") == ["10"]


def test_c_rounding_down_is_refused_for_the_same_reason() -> None:
    assert check("Quỹ mua vào khoảng 9 tấn trong phiên.") == ["9"]


def test_d_a_percentage_carried_exactly_is_allowed() -> None:
    assert check("USD yếu thêm 0.21%, hỗ trợ vàng.") == []


def test_e_a_percentage_rounded_to_a_friendlier_number_is_refused() -> None:
    assert check("USD yếu thêm 0.2%, hỗ trợ vàng.") == ["0.2"]


def test_f_the_synthesis_this_section_exists_for_needs_no_numbers() -> None:
    """The passing case is the one the prompt actually asks for."""
    assert check("Tin nghiêng tích cực nhờ USD yếu và dòng tiền ETF, nhưng giá vẫn đi xuống.") == []


def test_g_the_window_s_own_length_is_a_fact_the_run_owns() -> None:
    assert check("Trong 6 giờ qua dòng tiền ETF là điểm sáng duy nhất.") == []


def test_h_a_duration_that_is_not_the_window_is_refused() -> None:
    """ "trong 24 giờ qua" on a six-hour digest is a claim about a day nobody read."""
    assert check("Trong 24 giờ qua dòng tiền ETF là điểm sáng duy nhất.") == ["24"]


def test_i_a_clock_time_asserts_no_magnitude() -> None:
    assert check("Phiên Mỹ mở lúc 20:30 sẽ là phép thử tiếp theo.") == []


def test_j_a_date_asserts_no_magnitude_either() -> None:
    assert check("Số liệu CPI ngày 06/09 là phép thử tiếp theo.") == []


def test_k_a_number_inside_a_longer_number_does_not_vouch_for_it() -> None:
    """`109` is a substring of the item's `1096`, and that is not a source.

    A check written with `in` against the item text would pass this, which is
    how a rule like this quietly stops being a rule.
    """
    assert check("Nắm giữ của quỹ quanh 109 tấn.") == ["109"]
    assert check("Nắm giữ của quỹ đạt 1096 tấn.") == []


def test_l_a_computed_figure_restated_exactly_is_allowed() -> None:
    """The market section already printed it; repeating it invents nothing."""
    assert check("Giá vẫn giảm 50.24 USD dù tin hỗ trợ.") == []


def test_m_a_computed_figure_restated_approximately_is_refused() -> None:
    assert check("Giá vẫn giảm khoảng 50 USD dù tin hỗ trợ.") == ["50"]


def test_n_an_item_the_model_did_not_headline_still_vouches_for_its_figures() -> None:
    """Selection is editorial; provenance is about what was collected.

    A balance may rest on an item that did not earn a bullet - that is what
    synthesis means - so the authorised set is every offered item, not the
    chosen ones.
    """
    editorial = DigestEditorial(
        run_id=RUN_ID,
        status=WriterStatus.COMPLETED,
        items=(
            DigestItem(
                news_item_id="goldnewsvn:901",
                headline="Fed Williams hạ giọng về lợi suất.",
                impact=ImpactMarker.SUPPORTS_GOLD,
            ),
        ),
        balance="Dòng tiền ETF 9.98 tấn củng cố hướng tích cực.",
    )
    validate_editorial(editorial, facts(), run_id=RUN_ID)


def test_o_direction_is_carried_by_words_so_sign_is_not_compared() -> None:
    """The computed change is -50.24 and the prose says "giảm 50.24".

    Asking the two to match character for character would refuse the correct
    sentence. Whether the direction is right is the price block's business, and
    that block is copied rather than written.
    """
    assert reaction().net_change == Decimal("-50.24")
    assert check("Giá giảm 50.24 USD trong phiên.") == []


# --------------------------------------------------------------------------
# the rule's own edges
# --------------------------------------------------------------------------


def test_an_empty_balance_has_nothing_to_check() -> None:
    assert unsupported_balance_numbers("   ", SOURCES, WINDOW) == []


def test_every_offending_literal_is_reported_not_only_the_first() -> None:
    """A writer that rounded twice should learn both, in one rejection."""
    assert check("Quỹ mua gần 10 tấn và USD yếu 0.2%.") == ["10", "0.2"]


def test_the_literals_are_reported_as_the_model_wrote_them() -> None:
    """`9,98` is reported as `9,98` - a message quoting a form the writer never
    used sends it looking for text that is not in its own output."""
    assert check("Quỹ mua ròng 12,5 tấn.") == ["12,5"]


def test_a_date_in_an_item_authorises_nothing() -> None:
    """The fourth item ends "kể từ 2010", and that must not license a bare 2010
    used as a quantity somewhere else."""
    assert SemanticType.DATE_TIME in EXEMPT_SEMANTICS
    assert Decimal(2010) not in authorised_quantities(SOURCES, WINDOW)


def test_the_authorised_set_is_the_union_of_all_three_holders() -> None:
    prepared = facts()
    authorised = authorised_quantities(SOURCES, WINDOW, prepared.deterministic_lines)

    assert Decimal("9.98") in authorised, "stated by a collected item"
    assert Decimal("50.24") in authorised, "computed and printed by the shell"
    assert Decimal(6) in authorised, "the window's own length"
    assert Decimal(10) not in authorised, "nothing holds it"


# --------------------------------------------------------------------------
# enforcement: the rule reaches production through validate_editorial
# --------------------------------------------------------------------------


def editorial(balance: str) -> DigestEditorial:
    return DigestEditorial(
        run_id=RUN_ID,
        status=WriterStatus.COMPLETED,
        items=(
            DigestItem(
                news_item_id="goldnewsvn:903",
                headline="SPDR Gold Trust mua ròng 9.98 tấn.",
                impact=ImpactMarker.SUPPORTS_GOLD,
            ),
        ),
        balance=balance,
    )


def test_a_rounded_balance_is_refused_by_the_writer_validator() -> None:
    with pytest.raises(WriterResponseError) as excinfo:
        validate_editorial(editorial("SPDR mua ròng gần 10 tấn."), facts(), run_id=RUN_ID)

    assert "no collected item or computed figure supports" in str(excinfo.value)


def test_the_rejection_names_the_offending_numbers() -> None:
    """A writer told only "provenance failed" has nothing to correct."""
    with pytest.raises(WriterResponseError) as excinfo:
        validate_editorial(editorial("Quỹ mua gần 10 tấn, USD yếu 0.2%."), facts(), run_id=RUN_ID)

    assert excinfo.value.details["unsupported_numbers"] == ["10", "0.2"]


def test_an_exactly_quoted_balance_passes_the_writer_validator() -> None:
    validate_editorial(editorial("SPDR mua ròng 9.98 tấn, USD yếu 0.21%."), facts(), run_id=RUN_ID)


def test_the_wrong_run_is_still_refused_before_anything_is_measured() -> None:
    """Ordering: a response about another Run is not worth analysing."""
    stray = DigestEditorial(
        run_id="20260904_120000_other",
        status=WriterStatus.COMPLETED,
        items=editorial("x").items,
        balance="Quỹ mua gần 10 tấn.",
    )

    with pytest.raises(WriterResponseError) as excinfo:
        validate_editorial(stray, facts(), run_id=RUN_ID)

    assert "run_id does not match" in str(excinfo.value)


def test_an_uncollected_item_is_still_refused_before_the_balance_is_read() -> None:
    stray = DigestEditorial(
        run_id=RUN_ID,
        status=WriterStatus.COMPLETED,
        items=(
            DigestItem(
                news_item_id="goldnewsvn:999",
                headline="Một tin không có trong danh sách.",
                impact=ImpactMarker.MIXED_OR_UNCLEAR,
            ),
        ),
        balance="Quỹ mua gần 10 tấn.",
    )

    with pytest.raises(WriterResponseError) as excinfo:
        validate_editorial(stray, facts(), run_id=RUN_ID)

    assert "never collected" in str(excinfo.value)


def test_the_check_does_not_read_the_clock_or_the_network() -> None:
    """It compares two texts and a window. Nothing else is available to it."""
    import inspect

    from goldpipeline.services import digest_provenance

    body = inspect.getsource(digest_provenance)
    for forbidden in ("datetime.now", "utcnow", "requests", "httpx", "time.time"):
        assert forbidden not in body, forbidden


# --------------------------------------------------------------------------
# Round 6.5c.1a: the 🧭 Cán cân contract, clause by clause
#
# Round 6.5c.1 answered only the numeric half. These cases fix the whole
# contract in place, including the parts deterministic code deliberately does
# not own - because an unowned case that nobody wrote down is the one that
# quietly becomes owned by accident.
# --------------------------------------------------------------------------


def claimed(
    balance: str,
    *,
    statement: str,
    evidence: str,
    item_ids: tuple[str, ...] = ("goldnewsvn:903",),
    sources: tuple[DigestSourceItem, ...] = SOURCES,
) -> DigestEditorial:
    """An editorial whose balance carries one declared claim."""
    return DigestEditorial(
        run_id=RUN_ID,
        status=WriterStatus.COMPLETED,
        items=(
            DigestItem(
                news_item_id="goldnewsvn:903",
                headline="SPDR Gold Trust mua ròng 9.98 tấn.",
                impact=ImpactMarker.SUPPORTS_GOLD,
            ),
        ),
        balance=balance,
        news_claims=(
            NewsClaim(statement=statement, evidence=evidence, news_item_ids=list(item_ids)),
        ),
    )


def test_contract_a_a_pure_editorial_view_needs_no_source() -> None:
    """ "Tin trong cửa sổ đang nghiêng tích cực" is a judgement, not a fact.

    It appears verbatim in no item and must not have to. Requiring a citation
    for the one sentence the section exists to produce would leave the balance
    able to say nothing at all.
    """
    view = editorial("Tin trong cửa sổ đang nghiêng tích cực.")

    validate_editorial(view, facts(), run_id=RUN_ID)
    assert digest_precheck(view, facts()).ok


def test_contract_b_a_declared_factual_clause_is_checked_against_its_item() -> None:
    answer = claimed(
        "USD giảm 0.21%, đủ để giữ hướng tích cực.",
        statement="USD giảm 0.21%",
        evidence="Chỉ số USD giảm 0.21% trong phiên.",
        item_ids=("goldnewsvn:902",),
    )

    report = digest_precheck(answer, facts())

    assert [c.verdict for c in report.claims] == [ClaimVerdict.SUPPORTED]
    assert report.claims[0].supporting_item_id == "goldnewsvn:902"


def test_contract_b_a_claim_citing_an_item_that_does_not_say_it_is_refused() -> None:
    """The locality half of the contract, and the layer that owns it."""
    answer = claimed(
        "USD giảm 0.21%.",
        statement="USD giảm 0.21%",
        # Real quote, wrong item: this text is in 902, not in 903.
        evidence="Chỉ số USD giảm 0.21% trong phiên.",
        item_ids=("goldnewsvn:903",),
    )

    with pytest.raises(WriterResponseError) as excinfo:
        validate_editorial(answer, facts(), run_id=RUN_ID)

    assert "cited items do not support" in str(excinfo.value)
    assert excinfo.value.details["unsupported_claims"][0]["verdict"] == str(
        ClaimVerdict.EVIDENCE_NOT_IN_ITEM
    )


def test_contract_c_a_numeric_clause_needs_both_halves() -> None:
    """Locality and typed equivalence are independent, and both must hold.

    A claim can quote its item perfectly and still put a number in the balance
    that the item never stated - which is precisely how the Round 6.5b defect
    passed. Neither check subsumes the other.
    """
    rounded = claimed(
        "SPDR mua ròng gần 10 tấn.",
        statement="SPDR mua ròng gần 10 tấn",
        evidence="SPDR Gold Trust mua ròng 9.98 tấn.",
        item_ids=("goldnewsvn:903",),
    )

    report = digest_precheck(rounded, facts())

    assert report.claims[0].supported, "locality is satisfied - the evidence is really there"
    assert report.unsupported_numbers == ("10",), "and the number is still refused"
    assert not report.ok


def test_contract_d_re_quantification_has_no_tolerance() -> None:
    """9.98 -> 10 is refused, and so is 9.98 -> 9.9. There is no near-enough."""
    for approximation in ("gần 10 tấn", "khoảng 9.9 tấn", "xấp xỉ 10.0 tấn"):
        assert check(f"SPDR mua ròng {approximation}."), approximation


def test_contract_e_a_fact_may_be_carried_without_repeating_its_number() -> None:
    """ "SPDR tiếp tục mua ròng" drops the figure rather than approximating it.

    This is the escape the prompt actually wants a writer to take, so it must
    be open: allowed by the numeric guard because it quantifies nothing, and
    supported by locality because the claim quotes its item.
    """
    answer = claimed(
        "SPDR tiếp tục mua ròng, USD yếu đi.",
        statement="SPDR tiếp tục mua ròng",
        evidence="SPDR Gold Trust mua ròng 9.98 tấn.",
        item_ids=("goldnewsvn:903",),
    )

    validate_editorial(answer, facts(), run_id=RUN_ID)
    assert digest_precheck(answer, facts()).ok


def test_contract_f_an_undeclared_causal_bridge_is_not_caught_here() -> None:
    """The boundary, asserted rather than hoped about.

    "ETF mua vì lo lạm phát tăng" adds a motive the item does not report. It
    contains no unsourced quantity and declares no claim, so every
    deterministic check passes it - and that is the honest answer, not a bug.
    Motive is entailment, and this layer judges no meaning.

    What must be true is that the failure is *routed*: the reviewer is handed
    the item and the sentence, and told this gap is its own. The test that the
    prompt says so lives in test_digest_review.
    """
    bridge = editorial("ETF mua vì lo lạm phát tăng.")

    validate_editorial(bridge, facts(), run_id=RUN_ID)
    assert digest_precheck(bridge, facts()).ok, "deterministic code cannot see a motive"


def test_the_price_reaction_is_not_a_news_source() -> None:
    """A claim may not cite the market section as evidence for a news fact.

    The two are different kinds of thing: the price block is arithmetic over
    candles, and a claim is about what somebody published. Quoting one as the
    other would let a computed number vouch for an event.
    """
    prepared = facts()
    answer = claimed(
        "Giá giảm trong cửa sổ.",
        statement="Giá giảm trong cửa sổ",
        evidence=prepared.price_reaction_block.splitlines()[-1],
        item_ids=("goldnewsvn:903",),
    )

    report = digest_precheck(answer, prepared)

    assert report.claims[0].verdict is ClaimVerdict.EVIDENCE_NOT_IN_ITEM


def test_supportive_news_against_a_falling_price_is_not_a_contradiction() -> None:
    """The window this pipeline was built for, and it must pass cleanly.

    News leaning supportive while price fell is two true observations. A check
    that treated the disagreement as an error would force the writer to
    suppress one of them, which is the failure the causally-neutral price
    section exists to prevent.
    """
    prepared = facts(net="-50.24")
    assert prepared.price_reaction.net_change < 0

    honest = editorial("Tin nghiêng tích cực, nhưng giá vẫn đi xuống trong cửa sổ.")

    validate_editorial(honest, prepared, run_id=RUN_ID)
    assert digest_precheck(honest, prepared).ok
