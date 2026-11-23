"""Ported from orderflow-terminal (MIT). Deterministic offline tests. Fixtures are synthetic and only ever fed to
MarketState directly; the app has no code path that renders them as LIVE."""
import json
from decimal import Decimal as D

import pytest

from volscape.live.feed import backoff_delay, stream_url
from volscape.live.state import (BookSnapshot, MarketState, ParseError, RollingSum, Trade,
                             parse_message)


def trade_msg(sym="BTCUSDT", tid=1, p="100", q="1", m=False, T=1_700_000_000_000):
    return json.dumps({"stream": f"{sym.lower()}@trade", "data": {
        "e": "trade", "E": T, "s": sym, "t": tid, "p": p, "q": q, "T": T, "m": m, "M": True}})


def depth_msg(sym="BTCUSDT", uid=1, bids=(("100", "30"),), asks=(("102", "10"),)):
    return json.dumps({"stream": f"{sym.lower()}@depth20@100ms", "data": {
        "lastUpdateId": uid, "bids": [list(b) for b in bids], "asks": [list(a) for a in asks]}})


def mk_trade(tid, p="100", q="1", m=False, sym="BTCUSDT", T=1_700_000_000_000):
    return parse_message(trade_msg(sym, tid, p, q, m, T))


# -- aggressor classification ------------------------------------------------
def test_buyer_is_maker_means_seller_initiated():
    t = mk_trade(1, m=True)
    assert t.side == "SELL" and not t.is_buy


def test_buyer_is_taker_means_buyer_initiated():
    t = mk_trade(1, m=False)
    assert t.side == "BUY" and t.is_buy
    assert t.notional == D("100")


# -- parsing ---------------------------------------------------------------
def test_parse_malformed_raises_parse_error():
    with pytest.raises(ParseError):
        parse_message("{not json")
    with pytest.raises(ParseError):
        parse_message(json.dumps({"stream": "btcusdt@trade", "data": {"e": "trade", "s": "BTCUSDT"}}))
    with pytest.raises(ParseError):
        parse_message(json.dumps({"stream": "btcusdt@depth20@100ms", "data": {"lastUpdateId": 1, "bids": [["x", "1"]], "asks": []}}))


def test_parse_ignores_acks():
    assert parse_message(json.dumps({"result": None, "id": 1})) is None


def test_stream_url():
    assert stream_url("wss://x", "BTCUSDT") == "wss://x/stream?streams=btcusdt@trade/btcusdt@depth20@100ms"


# -- book metrics ------------------------------------------------------------
def test_spread_mid_imbalance_example():
    st = MarketState("BTCUSDT")
    st.apply_depth(parse_message(depth_msg()), 0.0)
    assert st.midprice() == D(101)
    assert st.spread() == D(2)
    assert st.spread_bps() == D(2) / D(101) * 10_000
    assert st.imbalance() == D("0.5")


def test_missing_and_zero_data():
    st = MarketState("BTCUSDT")
    assert st.midprice() is None and st.spread() is None and st.spread_bps() is None
    assert st.vwap() is None and st.imbalance() is None
    assert st.trade_rate(10.0) is None
    st.apply_depth(BookSnapshot("BTCUSDT", 1, ((D(100), D(0)),), ((D(102), D(0)),)), 0.0)
    assert st.bids == [] and st.imbalance() is None  # zero-qty levels dropped


def test_snapshot_replacement_and_sorting():
    st = MarketState("BTCUSDT")
    st.apply_depth(parse_message(depth_msg(uid=1, bids=(("99", "1"), ("100", "2")), asks=(("103", "1"), ("101", "4")))), 0.0)
    assert [p for p, _ in st.bids] == [D(100), D(99)]
    assert [p for p, _ in st.asks] == [D(101), D(103)]
    st.apply_depth(parse_message(depth_msg(uid=2, bids=(("98", "5"),), asks=(("104", "5"),))), 0.0)
    assert st.bids == [(D(98), D(5))] and st.asks == [(D(104), D(5))]  # replaced, not merged
    assert not st.apply_depth(parse_message(depth_msg(uid=1)), 0.0)    # out-of-order ignored
    cb, ca = st.cumulative_depth()
    assert cb == [D(5)] and ca == [D(5)]


# -- trades / VWAP -----------------------------------------------------------
def test_vwap():
    st = MarketState("BTCUSDT")
    st.apply_trade(mk_trade(1, "100", "1"), 0.0)
    st.apply_trade(mk_trade(2, "110", "3"), 0.0)
    assert st.vwap() == D(430) / D(4)


def test_duplicate_trade_ids_do_not_inflate():
    st = MarketState("BTCUSDT")
    assert st.apply_trade(mk_trade(5, "100", "1"), 0.0)
    assert not st.apply_trade(mk_trade(5, "100", "1"), 0.1)
    assert not st.apply_trade(mk_trade(4, "100", "1"), 0.2)
    assert st.duplicates == 2 and len(st.trades) == 1
    assert st.vwap_q == D(1) and st.signed_flow(0.3) == D(100)
    st.apply_trade(mk_trade(9), 0.4)
    assert st.missed_trades == 3


# -- rolling windows -----------------------------------------------------------
def test_rolling_sum_expires_without_new_events():
    w = RollingSum(60)
    w.add(0.0, D(10))
    w.add(30.0, D(-4))
    assert w.total(59.0) == D(6)
    assert w.total(61.0) == D(-4)        # first entry expired in a quiet period
    assert w.total(200.0) == D(0) and w.count(200.0) == 0


def test_signed_flow_and_rate_window():
    st = MarketState("BTCUSDT")
    st.apply_trade(mk_trade(1, "100", "2", m=False), 0.0)   # +200
    st.apply_trade(mk_trade(2, "100", "1", m=True), 1.0)    # -100
    assert st.signed_flow(2.0) == D(100)
    assert st.window_coverage(2.0) == 2.0
    assert st.signed_flow(61.5) == D(0)
    assert st.trade_rate(61.5) == 0.0


def test_large_trade_alert():
    st = MarketState("BTCUSDT", D(1000))
    st.apply_trade(mk_trade(1, "100", "5"), 0.0)
    assert not any(a.kind == "LARGE" for a in st.alerts)
    st.apply_trade(mk_trade(2, "100", "10", m=True), 0.0)
    a = st.alerts[0]
    assert a.kind == "LARGE" and a.side == "SELL"


# -- isolation -----------------------------------------------------------------
def test_symbol_change_clears_state_and_rejects_old_symbol():
    st = MarketState("BTCUSDT")
    st.apply_depth(parse_message(depth_msg()), 0.0)
    st.apply_trade(mk_trade(100), 0.0)
    st.reset("ETHUSDT")
    assert not st.bids and not st.trades and st.vwap() is None and st.last_trade_id is None
    assert not st.apply_trade(mk_trade(101, sym="BTCUSDT"), 1.0)
    assert not st.apply_depth(parse_message(depth_msg(sym="BTCUSDT")), 1.0)
    assert st.apply_trade(mk_trade(7, sym="ETHUSDT"), 1.0)  # lower id valid for a new symbol
    assert st.missed_trades == 0


def test_reconnect_clears_book_and_windows_but_keeps_dedupe():
    st = MarketState("BTCUSDT")
    st.apply_depth(parse_message(depth_msg()), 0.0)
    st.apply_trade(mk_trade(10, "100", "1"), 0.0)
    st.on_reconnect(5.0)
    assert not st.bids and st.imbalance() is None
    assert st.signed_flow(5.0) == D(0) and st.window_start is None
    assert st.gaps == 1
    assert not st.apply_trade(mk_trade(10), 6.0)       # replayed id after reconnect
    assert st.vwap() == D(100) and st.duplicates == 1


def test_price_series_forward_fills_quiet_seconds():
    st = MarketState("BTCUSDT")
    st.apply_trade(mk_trade(1, "100", T=10_000), 0)
    st.apply_trade(mk_trade(2, "105", T=13_500), 0)
    assert st.price_series(5, now_wall_s=14) == [D(100), D(100), D(100), D(105), D(105)]


def test_backoff_bounded():
    for attempt in range(1, 20):
        d = backoff_delay(attempt)
        assert 0.5 <= d <= 30.0
