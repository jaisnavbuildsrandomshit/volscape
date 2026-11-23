"""Background runner that bridges the asyncio live feeds into Streamlit's rerun model.

One daemon thread owns an asyncio loop that runs
  * a Binance trade+depth `Feed` for the selected symbol, feeding a `MarketState`, and
  * a `MarketBoard` (crypto mini-tickers real time, equities via Yahoo, delayed).
All state mutation happens on that loop under `self.lock`; Streamlit only ever reads a plain,
immutable `snapshot()` taken under the same lock. Nothing connects until `start()` is called.
"""
from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from .feed import Feed, Status
from .markets import MarketBoard
from .state import SYMBOLS, BookSnapshot, MarketState, Trade

EVENT_LOG_MAX = 60


def _f(x):
    return None if x is None else float(x)


@dataclass
class LiveSnapshot:
    connected: bool
    symbol: str
    status: str
    detail: str
    receive_age_s: float | None
    msg_rate: float
    last: float | None
    mid: float | None
    spread: float | None
    spread_bps: float | None
    imbalance: float | None
    vwap: float | None
    signed_flow_60s: float | None
    trade_rate: float | None
    window_s: float
    bids: list = field(default_factory=list)       # [(price, qty)] best first
    asks: list = field(default_factory=list)
    trades: list = field(default_factory=list)     # newest first: (time_ms, price, qty, side, notional)
    price_series: list = field(default_factory=list)
    flow_series: list = field(default_factory=list)
    duplicates: int = 0
    missed: int = 0
    gaps: int = 0
    quotes: list = field(default_factory=list)     # watchlist rows (dicts)
    crypto_status: str = "OFF"
    yahoo_status: str = "OFF"
    yahoo_error: str = ""
    events: list = field(default_factory=list)


class LiveHub:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.thread: threading.Thread | None = None
        self.symbol = "BTCUSDT"
        self.state = MarketState(self.symbol)
        self.gen = 0
        self.status, self.detail = "OFF", ""
        self.feed_task: asyncio.Task | None = None
        self.board: MarketBoard | None = None
        self.events: deque[tuple[float, str]] = deque(maxlen=EVENT_LOG_MAX)

    # ---------------------------------------------------------------- lifecycle
    @property
    def connected(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, symbol: str) -> None:
        if self.connected:
            self.set_symbol(symbol)
            return
        self.symbol = symbol if symbol in SYMBOLS else "BTCUSDT"
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run, name="volscape-live", daemon=True)
        self.thread.start()
        self._call(self._start_all)

    def stop(self) -> None:
        if not self.connected:
            return
        fut = asyncio.run_coroutine_threadsafe(self._stop_all(), self.loop)
        try:
            fut.result(timeout=5)
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)
        self.thread = None
        with self.lock:
            self.status, self.detail = "OFF", "disconnected by user"
            self._log("disconnected")

    def set_symbol(self, symbol: str) -> None:
        if symbol == self.symbol or symbol not in SYMBOLS:
            return
        self.symbol = symbol
        if self.connected:
            self._call(self._restart_feed)

    # ---------------------------------------------------------------- loop side
    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _call(self, fn) -> None:
        self.loop.call_soon_threadsafe(fn)

    def _log(self, msg: str) -> None:
        # dedupe consecutive identical diagnostics so reconnect storms don't flood the log
        if self.events and self.events[-1][1] == msg:
            return
        self.events.append((time.time(), msg))

    def _start_all(self) -> None:
        self.board = MarketBoard(on_error=lambda m: self._locked_log(m))
        self.board.start()
        self._restart_feed()

    def _locked_log(self, msg: str) -> None:
        with self.lock:
            self._log(msg[:160])

    def _restart_feed(self) -> None:
        if self.feed_task is not None:
            self.feed_task.cancel()
        with self.lock:
            self.gen += 1
            self.state.reset(self.symbol)
            self.status, self.detail = "CONNECTING", ""
            self._log(f"subscribe {self.symbol}")
        gen = self.gen
        feed = Feed(self.symbol, gen, self._on_event, self._on_status, self._on_error)
        self.feed_task = asyncio.ensure_future(feed.run())

    async def _stop_all(self) -> None:
        if self.feed_task is not None:
            self.feed_task.cancel()
            try:
                await asyncio.wait_for(self.feed_task, 3)
            except BaseException:  # noqa: BLE001
                pass
        if self.board is not None:
            await self.board.stop()

    def _on_event(self, gen: int, ev) -> None:
        now = time.monotonic()
        with self.lock:
            if gen != self.gen:
                return  # stale feed generation can never contaminate new state
            self.state.note_message(now)
            if isinstance(ev, BookSnapshot):
                self.state.apply_depth(ev, now)
            elif isinstance(ev, Trade):
                self.state.apply_trade(ev, now)

    def _on_status(self, gen: int, status: Status, detail: str) -> None:
        with self.lock:
            if gen != self.gen:
                return
            if status in (Status.RECONNECTING, Status.OFFLINE) and self.status == "LIVE":
                self.state.on_reconnect(time.monotonic())
            self.status, self.detail = status.value, detail
            self._log(f"{self.symbol} {status.value} {detail}".strip())

    def _on_error(self, gen: int, msg: str) -> None:
        with self.lock:
            if gen == self.gen:
                self._log(msg[:160])

    # ---------------------------------------------------------------- read side
    def snapshot(self, n_trades: int = 18, n_secs: int = 300) -> LiveSnapshot:
        now, wall = time.monotonic(), int(time.time())
        with self.lock:
            s = self.state
            quotes = []
            if self.board is not None:
                for q in self.board.quotes.values():
                    quotes.append(dict(symbol=q.symbol, name=q.name, group=q.group, source=q.source,
                                       price=q.price, change_pct=q.change_pct, high=q.high, low=q.low,
                                       currency=q.currency, market_time=q.market_time,
                                       age_s=None if q.received is None else now - q.received,
                                       series=list(q.series)))
            trades = [(t.trade_time_ms, float(t.price), float(t.qty), t.side, float(t.notional))
                      for t in list(s.trades)[-n_trades:]][::-1]
            return LiveSnapshot(
                connected=self.connected, symbol=self.symbol, status=self.status, detail=self.detail,
                receive_age_s=s.receive_age(now), msg_rate=s.message_rate(now),
                last=_f(s.last_price), mid=_f(s.midprice()), spread=_f(s.spread()),
                spread_bps=_f(s.spread_bps()), imbalance=_f(s.imbalance()), vwap=_f(s.vwap()),
                signed_flow_60s=_f(s.signed_flow(now)) if s.window_coverage(now) > 0 else None,
                trade_rate=s.trade_rate(now), window_s=s.window_coverage(now),
                bids=[(float(p), float(q)) for p, q in s.bids], asks=[(float(p), float(q)) for p, q in s.asks],
                trades=trades,
                price_series=[float(x) for x in s.price_series(n_secs, wall)],
                flow_series=[float(x) for x in s.flow_series(min(n_secs, 120), wall)],
                duplicates=s.duplicates, missed=s.missed_trades, gaps=s.gaps, quotes=quotes,
                crypto_status=self.board.crypto_status if self.board else "OFF",
                yahoo_status=self.board.yahoo_status if self.board else "OFF",
                yahoo_error=self.board.yahoo_error if self.board else "",
                events=list(self.events)[-12:][::-1])
