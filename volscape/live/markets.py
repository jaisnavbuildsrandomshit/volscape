"""Market overview data: Binance 24h mini-tickers (real time) for crypto and
Yahoo Finance chart/spark endpoints (unofficial, delayed, polled) for stocks
and indices. Watchlist values are display-only floats; the order-flow terminal
keeps exact Decimals.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

from websockets.asyncio.client import connect

from .feed import ENDPOINTS, backoff_delay, describe_error
from .state import SYMBOLS

log = logging.getLogger("orderflow.markets")

YAHOO_HOSTS = ("https://query1.finance.yahoo.com", "https://query2.finance.yahoo.com")
BINANCE_REST = ("https://data-api.binance.vision", "https://api.binance.com")
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept": "application/json"}
SPARK_POLL_S = 30.0
CHART_POLL_S = 15.0
KLINE_POLL_S = 300.0


@dataclass(frozen=True)
class Equity:
    symbol: str   # Yahoo symbol
    name: str
    group: str


EQUITIES: tuple[Equity, ...] = (
    Equity("^NSEI", "NIFTY 50", "INDIA"), Equity("^NSEBANK", "NIFTY BANK", "INDIA"),
    Equity("^BSESN", "SENSEX", "INDIA"), Equity("^CNXIT", "NIFTY IT", "INDIA"),
    Equity("RELIANCE.NS", "Reliance Industries", "INDIA"), Equity("TCS.NS", "Tata Consultancy", "INDIA"),
    Equity("HDFCBANK.NS", "HDFC Bank", "INDIA"), Equity("ICICIBANK.NS", "ICICI Bank", "INDIA"),
    Equity("INFY.NS", "Infosys", "INDIA"), Equity("SBIN.NS", "State Bank of India", "INDIA"),
    Equity("BHARTIARTL.NS", "Bharti Airtel", "INDIA"), Equity("ITC.NS", "ITC", "INDIA"),
    Equity("LT.NS", "Larsen & Toubro", "INDIA"), Equity("BAJFINANCE.NS", "Bajaj Finance", "INDIA"),
    Equity("^GSPC", "S&P 500", "US"), Equity("^IXIC", "NASDAQ Composite", "US"),
    Equity("^DJI", "Dow Jones", "US"), Equity("AAPL", "Apple", "US"), Equity("MSFT", "Microsoft", "US"),
    Equity("NVDA", "NVIDIA", "US"), Equity("AMZN", "Amazon", "US"), Equity("GOOGL", "Alphabet", "US"),
    Equity("META", "Meta Platforms", "US"), Equity("TSLA", "Tesla", "US"),
    Equity("^N225", "Nikkei 225", "GLOBAL"), Equity("^FTSE", "FTSE 100", "GLOBAL"),
    Equity("^GDAXI", "DAX", "GLOBAL"), Equity("^HSI", "Hang Seng", "GLOBAL"),
)
EQUITY_BY_SYMBOL = {e.symbol: e for e in EQUITIES}


@dataclass
class Quote:
    symbol: str
    name: str
    group: str            # CRYPTO / INDIA / US / GLOBAL
    source: str           # "BINANCE" (real time) or "YAHOO" (delayed)
    price: float | None = None
    ref: float | None = None          # 24h open (crypto) or previous close (equity)
    high: float | None = None
    low: float | None = None
    volume: float | None = None       # quote volume (crypto) / shares (equity)
    currency: str = ""
    market_time: int | None = None    # epoch seconds of the last price at the source
    received: float | None = None     # local monotonic receive time
    series: list[float] = field(default_factory=list)

    @property
    def change(self) -> float | None:
        return None if self.price is None or not self.ref else self.price - self.ref

    @property
    def change_pct(self) -> float | None:
        c = self.change
        return None if c is None or not self.ref else c / self.ref * 100


@dataclass
class Bar:
    t: int
    o: float
    h: float
    l: float
    c: float
    v: float


@dataclass
class EquityChart:
    symbol: str
    name: str
    currency: str
    exchange: str
    price: float | None
    prev_close: float | None
    day_high: float | None
    day_low: float | None
    volume: float | None
    market_time: int | None
    tz_name: str
    gmtoffset: int
    regular_start: int | None
    regular_end: int | None
    bars: list[Bar]

    def market_open(self, now_s: float) -> bool:
        return bool(self.regular_start and self.regular_end and self.regular_start <= now_s < self.regular_end)


def _num(x) -> float | None:
    return float(x) if isinstance(x, (int, float)) and x == x else None


def http_json(url: str, timeout: float = 10.0):
    req = urllib.request.Request(url, headers=HTTP_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def parse_chart(data: dict) -> EquityChart:
    res = (data.get("chart") or {}).get("result") or []
    if not res:
        err = (data.get("chart") or {}).get("error") or {}
        raise ValueError(err.get("description") or "empty chart result")
    r = res[0]
    m = r.get("meta") or {}
    ts = r.get("timestamp") or []
    q = ((r.get("indicators") or {}).get("quote") or [{}])[0]
    bars = []
    for i, t in enumerate(ts):
        try:
            o, h, l, c = (_num(q[k][i]) for k in ("open", "high", "low", "close"))
        except (KeyError, IndexError, TypeError):
            continue
        if None in (o, h, l, c):
            continue  # Yahoo emits null bars for gaps
        bars.append(Bar(int(t), o, h, l, c, _num((q.get("volume") or [None] * (i + 1))[i]) or 0.0))
    reg = (m.get("currentTradingPeriod") or {}).get("regular") or {}
    sym = m.get("symbol", "")
    return EquityChart(
        sym, EQUITY_BY_SYMBOL.get(sym, Equity(sym, m.get("shortName") or sym, "")).name,
        m.get("currency") or "", m.get("fullExchangeName") or m.get("exchangeName") or "",
        _num(m.get("regularMarketPrice")), _num(m.get("chartPreviousClose") or m.get("previousClose")),
        _num(m.get("regularMarketDayHigh")), _num(m.get("regularMarketDayLow")),
        _num(m.get("regularMarketVolume")), m.get("regularMarketTime"),
        m.get("exchangeTimezoneName") or "", int(m.get("gmtoffset") or 0),
        reg.get("start"), reg.get("end"), bars)


def parse_spark(data: dict) -> dict[str, dict]:
    """-> {symbol: {price, prev, high, low, time, currency, closes}}"""
    out = {}
    for item in ((data.get("spark") or {}).get("result") or []):
        resp = (item.get("response") or [{}])[0]
        m = resp.get("meta") or {}
        closes = ((resp.get("indicators") or {}).get("quote") or [{}])[0].get("close") or []
        out[item.get("symbol", m.get("symbol", ""))] = {
            "price": _num(m.get("regularMarketPrice")),
            "prev": _num(m.get("chartPreviousClose") or m.get("previousClose")),
            "high": _num(m.get("regularMarketDayHigh")), "low": _num(m.get("regularMarketDayLow")),
            "volume": _num(m.get("regularMarketVolume")),
            "time": m.get("regularMarketTime"), "currency": m.get("currency") or "",
            "closes": [c for c in (_num(x) for x in closes) if c is not None],
        }
    return out


def parse_mini_tickers(raw: str | bytes) -> list[dict]:
    msg = json.loads(raw)
    data = msg.get("data") if isinstance(msg, dict) else None
    if not isinstance(data, list):
        return []
    out = []
    for d in data:
        try:
            out.append({"symbol": d["s"], "price": float(d["c"]), "open": float(d["o"]),
                        "high": float(d["h"]), "low": float(d["l"]), "qvol": float(d["q"]),
                        "time": int(d["E"]) // 1000})
        except (KeyError, TypeError, ValueError):
            continue
    return out


class MarketBoard:
    """Owns the watchlist quotes and the background tasks that refresh them."""

    def __init__(self, on_error: Callable[[str], None] | None = None) -> None:
        self.quotes: dict[str, Quote] = {}
        for s in SYMBOLS.values():
            self.quotes[s.symbol] = Quote(s.symbol, s.base, "CRYPTO", "BINANCE", currency=s.quote)
        for e in EQUITIES:
            self.quotes[e.symbol] = Quote(e.symbol, e.name, e.group, "YAHOO")
        self.on_error = on_error or (lambda m: None)
        self.crypto_status = "CONNECTING"
        self.yahoo_status = "CONNECTING"
        self.yahoo_error = ""
        self.yahoo_last_ok: float | None = None
        self._tasks: list[asyncio.Task] = []

    def start(self) -> None:
        self._tasks = [asyncio.create_task(c) for c in
                       (self._tickers(), self._spark_loop(), self._kline_loop())]

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await asyncio.wait_for(t, 3)
            except BaseException:  # noqa: BLE001 - cancellation/teardown
                pass
        self._tasks = []

    # -- crypto: one real-time all-market mini-ticker stream ---------------
    async def _tickers(self) -> None:
        failures = 0
        while True:
            base = ENDPOINTS[(failures // 2) % len(ENDPOINTS)]
            try:
                async with connect(f"{base}/stream?streams=!miniTicker@arr", open_timeout=10,
                                   ping_interval=20, ping_timeout=20, close_timeout=2,
                                   max_size=2**22) as ws:
                    while True:
                        raw = await asyncio.wait_for(ws.recv(), timeout=30)
                        now = time.monotonic()
                        for d in parse_mini_tickers(raw):
                            q = self.quotes.get(d["symbol"])
                            if q is None:
                                continue
                            q.price, q.ref, q.high, q.low = d["price"], d["open"], d["high"], d["low"]
                            q.volume, q.market_time, q.received = d["qvol"], d["time"], now
                        self.crypto_status = "LIVE"
                        failures = 0
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                failures += 1
                self.crypto_status = "RECONNECTING" if failures < 3 else "OFFLINE"
                self.on_error(f"tickers: {describe_error(e)}")
                await asyncio.sleep(backoff_delay(failures))

    async def _kline_loop(self) -> None:
        """24 x 1h closes per crypto pair for watchlist sparklines (official REST)."""
        while True:
            for sym in list(SYMBOLS):
                for host in BINANCE_REST:
                    try:
                        rows = await asyncio.to_thread(
                            http_json, f"{host}/api/v3/klines?symbol={sym}&interval=1h&limit=24")
                        self.quotes[sym].series = [float(r[4]) for r in rows]
                        break
                    except Exception as e:  # noqa: BLE001
                        log.info("klines %s via %s: %s", sym, host, e)
                await asyncio.sleep(0.25)
            await asyncio.sleep(KLINE_POLL_S)

    # -- equities: Yahoo spark batches -------------------------------------
    async def _spark_loop(self) -> None:
        failures = 0
        syms = [e.symbol for e in EQUITIES]
        while True:
            try:
                for i in range(0, len(syms), 20):
                    batch = ",".join(syms[i:i + 20])
                    data = await asyncio.to_thread(http_json, f"{YAHOO_HOSTS[failures % 2]}/v7/finance/spark"
                                                   f"?symbols={urllib.parse.quote(batch)}&range=1d&interval=5m")
                    now = time.monotonic()
                    for sym, d in parse_spark(data).items():
                        q = self.quotes.get(sym)
                        if q is None or d["price"] is None:
                            continue
                        q.price, q.ref, q.high, q.low = d["price"], d["prev"], d["high"], d["low"]
                        q.volume, q.market_time, q.currency = d["volume"], d["time"], d["currency"]
                        q.series, q.received = d["closes"], now
                self.yahoo_status, self.yahoo_error = "DELAYED", ""
                self.yahoo_last_ok = time.monotonic()
                failures = 0
                await asyncio.sleep(SPARK_POLL_S + random.uniform(0, 3))
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                failures += 1
                self.yahoo_status = "OFFLINE" if failures >= 3 else "RETRYING"
                self.yahoo_error = describe_http(e)
                self.on_error(f"yahoo: {self.yahoo_error}")
                await asyncio.sleep(backoff_delay(failures + 2))


def describe_http(e: BaseException) -> str:
    import urllib.error
    if isinstance(e, urllib.error.HTTPError):
        if e.code == 429:
            return "HTTP 429: Yahoo rate limit, backing off"
        return f"HTTP {e.code} from Yahoo"
    if isinstance(e, urllib.error.URLError):
        return f"network error: {e.reason}"
    return describe_error(e)


async def fetch_equity_chart(symbol: str) -> EquityChart:
    last: Exception | None = None
    for host in YAHOO_HOSTS:
        try:
            data = await asyncio.to_thread(
                http_json, f"{host}/v8/finance/chart/{urllib.parse.quote(symbol)}?interval=1m&range=1d")
            return parse_chart(data)
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(describe_http(last)) if last else RuntimeError("no data")
