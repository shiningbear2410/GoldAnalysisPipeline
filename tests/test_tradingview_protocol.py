"""The TradingView wire codec, exercised as pure functions.

No socket, no library, no network. The awkward cases are the point: several
frames in one packet, a packet that stops mid-frame, a length header that lies,
and a payload that is framed correctly but is not JSON. Each of those is a way
a feed can hand us something that *looks* like candle history, so each has a
test that pins what happens.

The dividing line the tests enforce: this codec is lenient about **arrival**
(an incomplete tail is buffered, not rejected) and strict about **content** (a
complete frame with an unusable header raises).
"""

from __future__ import annotations

import json

import pytest

from goldpipeline.adapters.fake_tradingview import (
    FEED_INTERNALS_SENTINEL,
    critical_error,
    heartbeat,
    make_series,
    protocol_error,
    series_completed,
    session_chatter,
    timescale_update,
    unparseable,
)
from goldpipeline.adapters.tradingview_protocol import (
    MAX_FRAME_LENGTH,
    MessageKind,
    classify,
    decode_frames,
    encode_frame,
    encode_message,
    error_summary,
    extract_raw_bars,
    is_heartbeat,
)
from goldpipeline.domain.errors import TradingViewFramingError

# --- encoding -------------------------------------------------------------


class TestEncoding:
    def test_frame_declares_the_payload_length(self) -> None:
        assert encode_frame("hello") == "~m~5~m~hello"

    def test_length_counts_characters_and_round_trips(self) -> None:
        payload = "giá vàng ~ 4323"
        frames, remainder = decode_frames(encode_frame(payload))
        assert frames == [payload] and remainder == ""

    def test_message_is_compact_and_stable(self) -> None:
        frame = encode_message("create_series", ["cs_1", "sds_1", "s1", "sym_1", "240", 20, ""])
        payload = frame.split("~m~", 2)[2]
        assert " " not in payload
        assert json.loads(payload) == {
            "m": "create_series",
            "p": ["cs_1", "sds_1", "s1", "sym_1", "240", 20, ""],
        }

    def test_empty_payload_is_not_encodable_as_a_readable_frame(self) -> None:
        """A zero-length frame is rejected on the way back in, so nothing emits one."""
        with pytest.raises(TradingViewFramingError):
            decode_frames(encode_frame(""))


# --- decoding -------------------------------------------------------------


class TestDecoding:
    def test_one_message(self) -> None:
        frames, remainder = decode_frames(series_completed())
        assert len(frames) == 1 and remainder == ""
        assert classify(frames[0]).kind is MessageKind.SERIES_COMPLETED

    def test_multiple_messages_in_one_packet(self) -> None:
        packet = session_chatter() + heartbeat() + series_completed()
        frames, remainder = decode_frames(packet)
        assert remainder == ""
        assert [classify(f).kind for f in frames] == [
            MessageKind.OTHER,
            MessageKind.HEARTBEAT,
            MessageKind.SERIES_COMPLETED,
        ]

    def test_truncated_body_is_buffered_not_rejected(self) -> None:
        whole = series_completed()
        frames, remainder = decode_frames(whole[:-6])
        assert frames == [] and remainder == whole[:-6]
        # Re-reading with the rest appended yields the frame intact.
        frames, remainder = decode_frames(remainder + whole[-6:])
        assert len(frames) == 1 and remainder == ""

    @pytest.mark.parametrize("cut", [1, 2, 3, 4, 5, 6, 7, 8])
    def test_a_split_anywhere_in_the_header_reassembles(self, cut: int) -> None:
        whole = heartbeat() + series_completed()
        head, tail = whole[:cut], whole[cut:]
        first, remainder = decode_frames(head)
        second, tail_remainder = decode_frames(remainder + tail)
        assert tail_remainder == ""
        assert len(first) + len(second) == 2

    def test_complete_frames_are_returned_before_an_incomplete_tail(self) -> None:
        packet = heartbeat() + series_completed()[:5]
        frames, remainder = decode_frames(packet)
        assert len(frames) == 1 and is_heartbeat(frames[0])
        assert remainder.startswith("~m~")

    def test_no_header_where_one_must_be(self) -> None:
        with pytest.raises(TradingViewFramingError, match="frame header"):
            decode_frames("garbage without any framing")

    def test_trailing_garbage_after_a_valid_frame(self) -> None:
        with pytest.raises(TradingViewFramingError, match="frame header"):
            decode_frames(series_completed() + "then nonsense")

    def test_malformed_length_is_rejected(self) -> None:
        with pytest.raises(TradingViewFramingError, match="frame header"):
            decode_frames("~m~abc~m~payload")

    def test_zero_length_is_rejected(self) -> None:
        with pytest.raises(TradingViewFramingError, match="length of zero"):
            decode_frames("~m~0~m~")

    def test_length_past_the_ceiling_is_rejected_without_reading_a_body(self) -> None:
        with pytest.raises(TradingViewFramingError, match="ceiling"):
            decode_frames(f"~m~{MAX_FRAME_LENGTH + 1}~m~x")

    def test_absurd_length_digits_are_rejected_as_a_bad_header(self) -> None:
        with pytest.raises(TradingViewFramingError):
            decode_frames("~m~9999999999999~m~x")

    def test_empty_input(self) -> None:
        assert decode_frames("") == ([], "")

    def test_a_lone_delimiter_is_an_incomplete_header(self) -> None:
        for partial in ("~", "~m", "~m~", "~m~1", "~m~12~", "~m~12~m"):
            frames, remainder = decode_frames(partial)
            assert frames == [] and remainder == partial, partial


# --- classification -------------------------------------------------------


class TestClassification:
    @pytest.mark.parametrize(
        ("frame", "kind"),
        [
            (heartbeat(), MessageKind.HEARTBEAT),
            (timescale_update(make_series(count=3)), MessageKind.TIMESCALE_UPDATE),
            (series_completed(), MessageKind.SERIES_COMPLETED),
            (protocol_error(), MessageKind.PROTOCOL_ERROR),
            (critical_error(), MessageKind.CRITICAL_ERROR),
            (session_chatter(), MessageKind.OTHER),
            (unparseable(), MessageKind.UNPARSEABLE),
        ],
    )
    def test_kinds(self, frame: str, kind: MessageKind) -> None:
        payload = frame.split("~m~", 2)[2]
        assert classify(payload).kind is kind

    def test_data_update_is_recognised(self) -> None:
        from goldpipeline.adapters.fake_tradingview import data_update

        payload = data_update(make_series(count=2)).split("~m~", 2)[2]
        assert classify(payload).kind is MessageKind.DATA_UPDATE

    def test_classify_never_raises(self) -> None:
        for payload in ("", "[]", "null", '{"p":[1]}', '{"m":5}', "{", '"text"', "\x00"):
            assert classify(payload).kind is MessageKind.UNPARSEABLE

    def test_heartbeat_detection(self) -> None:
        assert is_heartbeat("~h~7")
        assert not is_heartbeat('{"m":"du","p":[]}')

    def test_error_summary_never_echoes_the_payload(self) -> None:
        payload = protocol_error().split("~m~", 2)[2]
        summary = error_summary(classify(payload))
        assert "protocol_error" in summary
        assert FEED_INTERNALS_SENTINEL not in summary


# --- series extraction ----------------------------------------------------


class TestSeriesExtraction:
    def test_bars_come_out_in_wire_order(self) -> None:
        rows = make_series(count=4)
        payload = timescale_update(rows).split("~m~", 2)[2]
        bars = extract_raw_bars(classify(payload), "sds_test")
        assert [list(bar.values) for bar in bars] == rows

    def test_a_different_series_id_yields_nothing(self) -> None:
        payload = timescale_update(make_series(count=3)).split("~m~", 2)[2]
        assert extract_raw_bars(classify(payload), "sds_other") == []

    @pytest.mark.parametrize(
        "params",
        [
            [],
            ["cs_test"],
            ["cs_test", "not a dict"],
            ["cs_test", {"sds_test": "not a dict"}],
            ["cs_test", {"sds_test": {"s": "not a list"}}],
            ["cs_test", {"sds_test": {}}],
        ],
    )
    def test_shapes_that_carry_no_bars(self, params: list[object]) -> None:
        payload = encode_message("timescale_update", params).split("~m~", 2)[2]
        assert extract_raw_bars(classify(payload), "sds_test") == []

    def test_rows_without_a_value_list_are_skipped(self) -> None:
        params = [
            "cs_test",
            {"sds_test": {"s": [{"i": 0}, "junk", {"i": 1, "v": [1, 2]}]}},
        ]
        payload = encode_message("timescale_update", params).split("~m~", 2)[2]
        bars = extract_raw_bars(classify(payload), "sds_test")
        assert [list(bar.values) for bar in bars] == [[1, 2]]

    def test_extraction_does_not_judge_values(self) -> None:
        """Deciding what a usable price is belongs to the market source."""
        params = ["cs_test", {"sds_test": {"s": [{"i": 0, "v": [1, None, "x", -5, 0]}]}}]
        payload = encode_message("timescale_update", params).split("~m~", 2)[2]
        bars = extract_raw_bars(classify(payload), "sds_test")
        assert list(bars[0].values) == [1, None, "x", -5, 0]
