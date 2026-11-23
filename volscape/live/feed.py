"""Binance Spot public market-data WebSocket transport.

One combined-stream connection per symbol: <sym>@trade + <sym>@depth20@100ms.
Responsibilities: connect, keepalive (the `websockets` library answers server
pings automatically and sends its own pings), proactive rollover before
Binance's 24h connection limit, bounded exponential backoff with jitter,
endpoint fallback, cancellation. Parsed events are handed to callbacks tagged
with the feed's generation so stale feeds can never contaminate new state.
"""
from __future__ import annotations

import asyncio
import logging
import random
import socket
import time
from enum import Enum
from typing import Callable

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidStatus, InvalidURI

from .state import BookSnapshot, ParseError, Trade, parse_message

log = logging.getLogger("orderflow.feed")

ENDPOINTS = (
    "wss://data-stream.binance.vision",   # market-data-only endpoint
    "wss://stream.binance.com:9443",
    "wss://stream.binance.com:443",
)
ROLLOVER_S = 23.5 * 3600      # Binance closes connections at 24h
RECV_TIMEOUT_S = 30.0         # no frames at all -> treat as dead and reconnect
BACKOFF_BASE_S = 1.0
BACKOFF_MAX_S = 30.0
OFFLINE_AFTER = 3             # consecutive failures before showing OFFLINE


class Status(str, Enum):
    CONNECTING = "CONNECTING"
    LIVE = "LIVE"
    RECONNECTING = "RECONNECTING"
    OFFLINE = "OFFLINE"
    STOPPED = "STOPPED"


def stream_url(base: str, symbol: str) -> str:
    s = symbol.lower()
    return f"{base}/stream?streams={s}@trade/{s}@depth20@100ms"


def backoff_delay(attempt: int, rng: random.Random | None = None) -> float:
    """Exponential backoff capped at BACKOFF_MAX_S, with 'equal jitter'."""
    r = rng or random
    cap = min(BACKOFF_MAX_S, BACKOFF_BASE_S * (2 ** max(0, attempt - 1)))
    return cap / 2 + r.uniform(0, cap / 2)


def describe_error(e: BaseException) -> str:
    if isinstance(e, InvalidStatus):
        code = e.response.status_code
        if code == 451:
            return "HTTP 451: Binance is unavailable from this location (restricted region)"
        if code in (403, 418, 429):
            return f"HTTP {code}: blocked or rate-limited by Binance"
        return f"HTTP {code} during WebSocket handshake"
    if isinstance(e, socket.gaierror):
        return "DNS lookup failed (offline, or DNS blocks the exchange)"
    if isinstance(e, (TimeoutError, asyncio.TimeoutError)):
        return "timed out (firewall, proxy, or network outage?)"
    if isinstance(e, ConnectionRefusedError):
        return "connection refused"
    if isinstance(e, ConnectionClosed):
        return f"connection closed (code {e.rcvd.code if e.rcvd else 'none'})"
    if isinstance(e, OSError):
        return f"network error: {e.strerror or e}"
    return f"{type(e).__name__}: {e}"


class Feed:
    def __init__(
        self,
        symbol: str,
        generation: int,
        on_event: Callable[[int, Trade | BookSnapshot], None],
        on_status: Callable[[int, Status, str], None],
        on_error: Callable[[int, str], None],
        endpoints: tuple[str, ...] = ENDPOINTS,
    ) -> None:
        self.symbol = symbol
        self.gen = generation
        self.on_event, self.on_status, self.on_error = on_event, on_status, on_error
        self.endpoints = endpoints
        self.status = Status.CONNECTING
        self.retry_at: float | None = None
        self.endpoint: str | None = None

    def _set(self, status: Status, detail: str = "") -> None:
        self.status = status
        self.on_status(self.gen, status, detail)

    async def run(self) -> None:
        failures = 0
        ep_idx = 0
        ever_live = False
        try:
            while True:
                base = self.endpoints[ep_idx % len(self.endpoints)]
                url = stream_url(base, self.symbol)
                self._set(Status.RECONNECTING if ever_live or failures else Status.CONNECTING, base)
                try:
                    rolled = await self._session(url, base)
                    failures = 0
                    if rolled:
                        continue  # planned rollover: reconnect immediately
                    raise ConnectionError("server closed the stream")
                except asyncio.CancelledError:
                    raise
                except InvalidURI as e:
                    self.on_error(self.gen, f"invalid endpoint {base}: {e}")
                    failures += 1
                except Exception as e:  # noqa: BLE001 - every transport fault is retried
                    failures += 1
                    msg = f"{base}: {describe_error(e)}"
                    log.warning(msg)
                    self.on_error(self.gen, msg)
                ever_live = ever_live or self.endpoint is not None
                if failures and failures % 2 == 0:
                    ep_idx += 1  # rotate endpoint after repeated failures
                delay = backoff_delay(failures)
                self.retry_at = time.monotonic() + delay
                self._set(Status.OFFLINE if failures >= OFFLINE_AFTER else Status.RECONNECTING,
                          f"retry in {delay:.1f}s")
                await asyncio.sleep(delay)
                self.retry_at = None
        finally:
            self.status = Status.STOPPED

    async def _session(self, url: str, base: str) -> bool:
        """Run one connection. Returns True on planned rollover."""
        async with connect(url, open_timeout=10, ping_interval=20, ping_timeout=20,
                           close_timeout=2, max_size=2**20, max_queue=256) as ws:
            opened = time.monotonic()
            first = True
            while True:
                remaining = ROLLOVER_S - (time.monotonic() - opened)
                if remaining <= 0:
                    self.on_error(self.gen, "24h connection rollover: reconnecting")
                    return True
                raw = await asyncio.wait_for(ws.recv(), timeout=min(remaining, RECV_TIMEOUT_S))
                try:
                    ev = parse_message(raw, int(time.time() * 1000))
                except ParseError as e:
                    self.on_error(self.gen, f"parse: {e}"[:160])
                    continue
                if ev is None:
                    continue
                if first:
                    first = False
                    self.endpoint = base
                    self._set(Status.LIVE, base)
                self.on_event(self.gen, ev)
