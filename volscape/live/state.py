"""Market state and analytics. Pure, synchronous, and driven by injected clocks.

All prices/quantities are Decimal (parsed from Binance's string fields) so sums
and differences are exact. `now` arguments are time.monotonic() seconds; wall
clock values are epoch milliseconds.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

ZERO = Decimal(0)
DEPTH_LEVELS = 20
FLOW_WINDOW_S = 60.0
RATE_WINDOW_S = 10.0
MSG_RATE_WINDOW_S = 5.0
TAPE_MAX = 5000          # bounded trade history (tape + CSV export)
PRICE_HISTORY_S = 600    # one close per second, bounded
ALERTS_MAX = 200


@dataclass(frozen=True)
class SymbolSpec:
    symbol: str
    base: str
    quote: str
    price_dp: int
    qty_dp: int
    large_quote: Decimal  # default large-trade threshold in quote currency


def _spec(base: str, price_dp: int, qty_dp: int, large: int) -> SymbolSpec:
    return SymbolSpec(base + "USDT", base, "USDT", price_dp, qty_dp, Decimal(large))


# Binance Spot USDT pairs. Display precision only; values are kept exact.
SYMBOLS: dict[str, SymbolSpec] = {s.symbol: s for s in (
    _spec("BTC", 2, 5, 250_000), _spec("ETH", 2, 4, 100_000), _spec("SOL", 2, 3, 50_000),
    _spec("BNB", 2, 3, 50_000), _spec("XRP", 4, 1, 50_000), _spec("DOGE", 5, 0, 25_000),
    _spec("ADA", 4, 1, 25_000), _spec("AVAX", 2, 2, 25_000), _spec("LINK", 2, 2, 25_000),
    _spec("DOT", 3, 2, 25_000), _spec("TRX", 4, 1, 25_000), _spec("LTC", 2, 3, 25_000),
    _spec("BCH", 1, 3, 25_000), _spec("NEAR", 3, 1, 25_000), _spec("APT", 3, 2, 25_000),
    _spec("SUI", 4, 1, 25_000), _spec("ARB", 4, 1, 25_000), _spec("OP", 3, 2, 25_000),
    _spec("UNI", 3, 2, 25_000), _spec("ATOM", 3, 2, 25_000),
    _spec("FIL", 3, 2, 25_000), _spec("PEPE", 8, 0, 25_000), _spec("SHIB", 8, 0, 25_000),
)}


@dataclass(frozen=True)
class Trade:
    symbol: str
    trade_id: int
    price: Decimal
    qty: Decimal
    trade_time_ms: int
    buyer_is_maker: bool
    recv_wall_ms: int = 0

    @property
    def is_buy(self) -> bool:
        """Aggressor side. Binance `m` = "is the buyer the market maker?".
        Buyer is maker -> the seller crossed the spread -> seller-initiated."""
        return not self.buyer_is_maker

    @property
    def side(self) -> str:
        return "BUY" if self.is_buy else "SELL"

    @property
    def notional(self) -> Decimal:
        return self.price * self.qty


@dataclass(frozen=True)
class BookSnapshot:
    symbol: str
    last_update_id: int
    bids: tuple[tuple[Decimal, Decimal], ...]
    asks: tuple[tuple[Decimal, Decimal], ...]


class ParseError(ValueError):
    pass


def _levels(raw) -> list[tuple[Decimal, Decimal]]:
    out = []
    for lvl in raw:
        p, q = Decimal(lvl[0]), Decimal(lvl[1])
        if not (p.is_finite() and q.is_finite()) or p <= 0 or q < 0:
            raise ParseError(f"bad level {lvl!r}")
        out.append((p, q))
    return out


def parse_message(raw: str | bytes, recv_wall_ms: int = 0) -> Trade | BookSnapshot | None:
    """Parse one combined-stream message: {"stream": "...", "data": {...}}.

    Returns None for non-market messages (e.g. subscription acks).
    Raises ParseError for malformed market messages.
    """
    try:
        msg = json.loads(raw)
    except (ValueError, TypeError) as e:
        raise ParseError(f"invalid JSON: {e}") from None
    if not isinstance(msg, dict) or "stream" not in msg:
        return None
    stream, data = msg.get("stream"), msg.get("data")
    if not isinstance(stream, str) or not isinstance(data, dict):
        raise ParseError("missing stream/data")
    sym, _, kind = stream.partition("@")
    sym = sym.upper()
    try:
        if kind == "trade":
            if data.get("e") != "trade" or str(data.get("s", "")).upper() != sym:
                raise ParseError(f"trade event mismatch on {stream}")
            price, qty = Decimal(data["p"]), Decimal(data["q"])
            if not (price.is_finite() and qty.is_finite()) or price <= 0 or qty <= 0:
                raise ParseError("non-positive trade price/qty")
            return Trade(sym, int(data["t"]), price, qty, int(data["T"]),
                         bool(data["m"]), recv_wall_ms)
        if kind.startswith("depth"):
            bids = _levels(data["bids"])
            asks = _levels(data["asks"])
            return BookSnapshot(sym, int(data["lastUpdateId"]), tuple(bids), tuple(asks))
    except (KeyError, TypeError, IndexError, InvalidOperation) as e:
        raise ParseError(f"{stream}: {type(e).__name__} {e}") from None
    return None


class RollingSum:
    """Time-windowed sum with O(1) amortised add/expire. Expires on read, so
    quiet periods drain the window even when no new events arrive."""

    def __init__(self, window_s: float, maxlen: int = 200_000):
        self.window_s = window_s
        self._q: deque[tuple[float, Decimal]] = deque()
        self._sum = ZERO
        self._maxlen = maxlen

    def add(self, now: float, value: Decimal) -> None:
        self._q.append((now, value))
        self._sum += value
        if len(self._q) > self._maxlen:  # hard memory bound
            _, v = self._q.popleft()
            self._sum -= v
        self.expire(now)

    def expire(self, now: float) -> None:
        cutoff = now - self.window_s
        q = self._q
        while q and q[0][0] <= cutoff:
            _, v = q.popleft()
            self._sum -= v

    def total(self, now: float) -> Decimal:
        self.expire(now)
        return self._sum

    def count(self, now: float) -> int:
        self.expire(now)
        return len(self._q)

    def clear(self) -> None:
        self._q.clear()
        self._sum = ZERO


@dataclass
class Alert:
    wall_ms: int
    kind: str      # LARGE / CONN / STALE / INFO / EXPORT / ERROR
    text: str
    side: str = ""  # BUY / SELL for large trades


def safe_div(n: Decimal, d: Decimal) -> Decimal | None:
    return None if d == 0 else n / d


@dataclass
class MarketState:
    symbol: str
    large_quote: Decimal = field(default=Decimal(0))

    def __post_init__(self) -> None:
        if not self.large_quote:
            self.large_quote = SYMBOLS[self.symbol].large_quote
        self.alerts: deque[Alert] = deque(maxlen=ALERTS_MAX)
        self.msg_rate = RollingSum(MSG_RATE_WINDOW_S)
        self.reset(self.symbol)

    # ---- lifecycle -------------------------------------------------
    def reset(self, symbol: str) -> None:
        """Clear every symbol-specific value (used on symbol change)."""
        self.symbol = symbol
        self.spec = SYMBOLS[symbol]
        self.bids: list[tuple[Decimal, Decimal]] = []
        self.asks: list[tuple[Decimal, Decimal]] = []
        self.book_update_id: int | None = None
        self.trades: deque[Trade] = deque(maxlen=TAPE_MAX)
        self.last_trade_id: int | None = None
        self.vwap_pv = ZERO
        self.vwap_q = ZERO
        self.session_start_ms: int | None = None
        self.flow = RollingSum(FLOW_WINDOW_S)
        self.buy_vol = RollingSum(FLOW_WINDOW_S)
        self.sell_vol = RollingSum(FLOW_WINDOW_S)
        self.rate = RollingSum(RATE_WINDOW_S)
        self.window_start: float | None = None   # continuity start for rolling windows
        self.price_secs: deque[tuple[int, Decimal]] = deque(maxlen=PRICE_HISTORY_S)
        self.flow_secs: deque[tuple[int, Decimal]] = deque(maxlen=PRICE_HISTORY_S)  # net quote flow / second
        self.duplicates = 0
        self.missed_trades = 0
        self.gaps = 0
        self.parse_errors = 0
        self.reconnects = 0
        self.last_msg_at: float | None = None
        self.msg_rate.clear()

    def on_reconnect(self, now: float) -> None:
        """A connection gap occurred: drop the (possibly stale) book and restart
        continuity-sensitive rolling windows. VWAP/tape persist, and the gap is
        counted so the UI can flag it."""
        self.reconnects += 1
        self.gaps += 1
        self.bids, self.asks, self.book_update_id = [], [], None
        for w in (self.flow, self.buy_vol, self.sell_vol, self.rate):
            w.clear()
        self.window_start = None
        self.last_msg_at = None

    # ---- ingestion -------------------------------------------------
    def note_message(self, now: float) -> None:
        self.last_msg_at = now
        self.msg_rate.add(now, Decimal(1))

    def apply_depth(self, snap: BookSnapshot, now: float) -> bool:
        if snap.symbol != self.symbol:
            return False
        if self.book_update_id is not None and snap.last_update_id < self.book_update_id:
            return False  # out-of-order snapshot
        # Partial-depth messages are full replacements of the top N, not diffs.
        self.bids = sorted((l for l in snap.bids if l[1] > 0), key=lambda l: l[0], reverse=True)[:DEPTH_LEVELS]
        self.asks = sorted((l for l in snap.asks if l[1] > 0), key=lambda l: l[0])[:DEPTH_LEVELS]
        self.book_update_id = snap.last_update_id
        return True

    def apply_trade(self, tr: Trade, now: float) -> bool:
        if tr.symbol != self.symbol:
            return False
        if self.last_trade_id is not None:
            if tr.trade_id <= self.last_trade_id:
                self.duplicates += 1  # replay across reconnect or duplicate frame
                return False
            if tr.trade_id > self.last_trade_id + 1:
                self.missed_trades += tr.trade_id - self.last_trade_id - 1
        self.last_trade_id = tr.trade_id
        if self.window_start is None:
            self.window_start = now
        if self.session_start_ms is None:
            self.session_start_ms = tr.trade_time_ms
        notional = tr.notional
        self.trades.append(tr)
        self.vwap_pv += notional
        self.vwap_q += tr.qty
        self.flow.add(now, notional if tr.is_buy else -notional)
        (self.buy_vol if tr.is_buy else self.sell_vol).add(now, notional)
        self.rate.add(now, Decimal(1))
        sec = tr.trade_time_ms // 1000
        if self.price_secs and self.price_secs[-1][0] == sec:
            self.price_secs[-1] = (sec, tr.price)
        elif not self.price_secs or sec > self.price_secs[-1][0]:
            self.price_secs.append((sec, tr.price))
        signed = notional if tr.is_buy else -notional
        if self.flow_secs and self.flow_secs[-1][0] == sec:
            self.flow_secs[-1] = (sec, self.flow_secs[-1][1] + signed)
        elif not self.flow_secs or sec > self.flow_secs[-1][0]:
            self.flow_secs.append((sec, signed))
        if notional >= self.large_quote > 0:
            q = self.spec
            self.alerts.appendleft(Alert(
                tr.recv_wall_ms or tr.trade_time_ms, "LARGE",
                f"{tr.side:<4} {tr.qty:,.{q.qty_dp}f} {q.base} @ {tr.price:,.{q.price_dp}f}"
                f" = {notional:,.0f} {q.quote}", tr.side))
        return True

    def alert(self, wall_ms: int, kind: str, text: str) -> None:
        self.alerts.appendleft(Alert(wall_ms, kind, text))

    # ---- analytics -------------------------------------------------
    @property
    def best_bid(self) -> Decimal | None:
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> Decimal | None:
        return self.asks[0][0] if self.asks else None

    @property
    def last_price(self) -> Decimal | None:
        return self.trades[-1].price if self.trades else None

    def midprice(self) -> Decimal | None:
        b, a = self.best_bid, self.best_ask
        return None if b is None or a is None else (a + b) / 2

    def spread(self) -> Decimal | None:
        b, a = self.best_bid, self.best_ask
        return None if b is None or a is None else a - b

    def spread_bps(self) -> Decimal | None:
        s, m = self.spread(), self.midprice()
        if s is None or m is None:
            return None
        r = safe_div(s, m)
        return None if r is None else r * 10_000

    def vwap(self) -> Decimal | None:
        return safe_div(self.vwap_pv, self.vwap_q)

    def depth_totals(self) -> tuple[Decimal, Decimal]:
        return sum((q for _, q in self.bids), ZERO), sum((q for _, q in self.asks), ZERO)

    def imbalance(self) -> Decimal | None:
        """Top-20 displayed-quantity imbalance in [-1, 1]."""
        if not self.bids or not self.asks:
            return None
        b, a = self.depth_totals()
        return safe_div(b - a, b + a)

    def cumulative_depth(self) -> tuple[list[Decimal], list[Decimal]]:
        def cum(levels):
            out, run = [], ZERO
            for _, q in levels:
                run += q
                out.append(run)
            return out
        return cum(self.bids), cum(self.asks)

    def window_coverage(self, now: float) -> float:
        """Seconds of continuous data behind rolling windows (for warming-up)."""
        return 0.0 if self.window_start is None else now - self.window_start

    def signed_flow(self, now: float) -> Decimal:
        return self.flow.total(now)

    def trade_rate(self, now: float) -> float | None:
        cov = self.window_coverage(now)
        if cov <= 0:
            return None
        return self.rate.count(now) / min(RATE_WINDOW_S, max(cov, 1.0))

    def message_rate(self, now: float) -> float:
        return self.msg_rate.count(now) / MSG_RATE_WINDOW_S

    def receive_age(self, now: float) -> float | None:
        return None if self.last_msg_at is None else now - self.last_msg_at

    def flow_series(self, n: int, now_wall_s: int) -> list[Decimal]:
        """Net signed quote flow per second for the last n seconds (0 when quiet)."""
        if n <= 0:
            return []
        by_sec = dict(self.flow_secs)
        return [by_sec.get(s, ZERO) for s in range(now_wall_s - n + 1, now_wall_s + 1)]

    def price_series(self, n: int, now_wall_s: int | None = None) -> list[Decimal]:
        """Per-second closing trade price, forward-filled over quiet seconds."""
        if not self.price_secs or n <= 0:
            return []
        end = self.price_secs[-1][0] if now_wall_s is None else max(now_wall_s, self.price_secs[-1][0])
        start = end - n + 1
        out: list[Decimal] = []
        items = list(self.price_secs)
        i, last = 0, None
        while i < len(items) and items[i][0] < start:
            last = items[i][1]
            i += 1
        for s in range(start, end + 1):
            while i < len(items) and items[i][0] <= s:
                last = items[i][1]
                i += 1
            if last is not None:
                out.append(last)
        return out
