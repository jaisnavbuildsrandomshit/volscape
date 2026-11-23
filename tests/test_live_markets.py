"""Ported from orderflow-terminal (MIT): offline parser tests with recorded payloads."""
import asyncio
import json
from decimal import Decimal as D

from volscape.live import markets as M
from volscape.live.state import MarketState
from tests.test_live_state import mk_trade

CHART = {"chart": {"result": [{
    "meta": {"symbol": "^NSEI", "currency": "INR", "exchangeName": "NSI", "regularMarketPrice": 25387.4,
             "chartPreviousClose": 25144.3, "regularMarketDayHigh": 25501.35, "regularMarketDayLow": 25267.3,
             "regularMarketVolume": 0, "regularMarketTime": 1794909667, "exchangeTimezoneName": "Asia/Kolkata",
             "gmtoffset": 19800,
             "currentTradingPeriod": {"regular": {"start": 1794887100, "end": 1794909600}}},
    "timestamp": [1794887100, 1794887160, 1794887220],
    "indicators": {"quote": [{"open": [25200, None, 25300], "high": [25250, None, 25350],
                              "low": [25190, None, 25280], "close": [25240, None, 25330],
                              "volume": [0, None, 0]}]}}], "error": None}}

SPARK = {"spark": {"result": [{"symbol": "AAPL", "response": [{
    "meta": {"symbol": "AAPL", "currency": "USD", "regularMarketPrice": 200.5, "chartPreviousClose": 198.0,
             "regularMarketDayHigh": 201, "regularMarketDayLow": 197.5, "regularMarketTime": 1794909667},
    "indicators": {"quote": [{"close": [198.2, None, 199.0, 200.5]}]}}]}]}}


def test_parse_chart_skips_null_bars_and_reads_session():
    ch = M.parse_chart(CHART)
    assert ch.name == "NIFTY 50" and ch.currency == "INR"
    assert len(ch.bars) == 2 and ch.bars[-1].c == 25330
    assert ch.market_open(1794887200) and not ch.market_open(1794909700)


def test_parse_chart_error():
    try:
        M.parse_chart({"chart": {"result": None, "error": {"description": "No data found"}}})
    except ValueError as e:
        assert "No data" in str(e)
    else:
        raise AssertionError


def test_parse_spark_and_quote_change():
    d = M.parse_spark(SPARK)["AAPL"]
    assert d["closes"] == [198.2, 199.0, 200.5]
    q = M.Quote("AAPL", "Apple", "US", "YAHOO", price=d["price"], ref=d["prev"])
    assert round(q.change, 2) == 2.5 and round(q.change_pct, 4) == round(2.5 / 198 * 100, 4)
    assert M.Quote("X", "X", "US", "YAHOO", price=1.0, ref=0.0).change_pct is None


def test_parse_mini_tickers_skips_bad_rows():
    raw = json.dumps({"stream": "!miniTicker@arr", "data": [
        {"e": "24hrMiniTicker", "E": 1700000000000, "s": "BTCUSDT", "c": "100", "o": "90",
         "h": "101", "l": "89", "v": "1", "q": "1000"},
        {"s": "ETHUSDT"}]})
    rows = M.parse_mini_tickers(raw)
    assert len(rows) == 1 and rows[0]["symbol"] == "BTCUSDT" and rows[0]["open"] == 90.0


def test_flow_series_buckets_per_second():
    st = MarketState("BTCUSDT")
    st.apply_trade(mk_trade(1, "100", "1", m=False, T=10_000), 0)
    st.apply_trade(mk_trade(2, "100", "3", m=True, T=10_500), 0)
    st.apply_trade(mk_trade(3, "100", "1", m=False, T=12_000), 0)
    assert st.flow_series(4, 12) == [D(0), D(-200), D(0), D(100)]

