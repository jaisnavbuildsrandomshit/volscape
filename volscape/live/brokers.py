"""Stock / index option-chain snapshots via the user's OWN API credentials.

* Upstox (NSE, India): GET https://api.upstox.com/v2/option/contract and /v2/option/chain
  Auth: Authorization: Bearer <access token>   (env UPSTOX_ACCESS_TOKEN; Upstox tokens expire daily)
  NSE index and stock options are European; expiry taken as 15:30 IST (10:00 UTC) on the expiry date.
* Alpaca (US): GET https://data.alpaca.markets/v1beta1/options/snapshots/{underlying}?feed=indicative
  plus GET https://data.alpaca.markets/v2/stocks/trades/latest?symbols=..&feed=iex for spot.
  Auth headers APCA-API-KEY-ID / APCA-API-SECRET-KEY (env ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY).
  US single-stock and ETF options are AMERICAN; rows are marked so, and validation then requires
  the explicit European-approximation opt-in. Expiry taken as 16:00 America/New_York.
  The free 'indicative' feed is delayed and modified by Alpaca (their docs); labelled as such.

Credentials are read from environment variables only, sent only to the provider's own host, and
never logged, displayed or written to exports.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from .chainbuild import build_chain_csv

UPSTOX = "https://api.upstox.com/v2"
ALPACA = "https://data.alpaca.markets"

# Upstox instrument keys (segment|identifier). Stocks use the NSE_EQ|<ISIN> form.
NSE_UNDERLYINGS = {
    "NIFTY 50": "NSE_INDEX|Nifty 50",
    "NIFTY BANK": "NSE_INDEX|Nifty Bank",
    "RELIANCE": "NSE_EQ|INE002A01018",
    "TCS": "NSE_EQ|INE467B01029",
    "HDFCBANK": "NSE_EQ|INE040A01034",
    "INFY": "NSE_EQ|INE009A01021",
    "ICICIBANK": "NSE_EQ|INE090A01021",
    "SBIN": "NSE_EQ|INE062A01020",
}
US_UNDERLYINGS = ("SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA", "GOOGL", "IWM")


class MissingCredentials(RuntimeError):
    pass


def _get(url: str, headers: dict, timeout: float = 20.0):
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "volscape/1.0", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        hint = {401: "credentials rejected or expired", 403: "your plan does not cover this data",
                429: "rate limited, retry shortly"}.get(e.code, "")
        raise RuntimeError(f"HTTP {e.code} from {urllib.parse.urlsplit(url).netloc} {hint}".strip()) from None


# ------------------------------------------------------------------------------------------ Upstox (NSE)
def upstox_credentials() -> dict:
    tok = os.environ.get("UPSTOX_ACCESS_TOKEN", "").strip()
    if not tok:
        raise MissingCredentials("UPSTOX_ACCESS_TOKEN is not set")
    return {"Authorization": f"Bearer {tok}"}


def upstox_rows(chain_rows: list[dict]) -> pd.DataFrame:
    """Documented /v2/option/chain rows -> quotes frame (one row per call/put)."""
    out = []
    for r in chain_rows:
        exp = pd.Timestamp(r["expiry"] + " 15:30", tz="Asia/Kolkata")
        for typ, key in (("C", "call_options"), ("P", "put_options")):
            md = (r.get(key) or {}).get("market_data") or {}
            out.append(dict(expiry=exp, strike=float(r["strike_price"]), option_type=typ,
                            bid=md.get("bid_price") or None, ask=md.get("ask_price") or None,
                            volume=md.get("volume"), open_interest=md.get("oi")))
    return pd.DataFrame(out)


def fetch_upstox(name: str, max_expiries: int = 6) -> tuple[bytes, dict]:
    key = NSE_UNDERLYINGS.get(name, name)
    h = upstox_credentials()
    contracts = _get(f"{UPSTOX}/option/contract?instrument_key={urllib.parse.quote(key)}", h).get("data") or []
    expiries = sorted({c["expiry"] for c in contracts if c.get("expiry")})[:max_expiries]
    if not expiries:
        raise RuntimeError(f"Upstox returned no option contracts for {key}")
    rows, spot = [], None
    for e in expiries:
        data = _get(f"{UPSTOX}/option/chain?instrument_key={urllib.parse.quote(key)}&expiry_date={e}", h).get("data") or []
        rows += data
        spot = spot or next((d.get("underlying_spot_price") for d in data if d.get("underlying_spot_price")), None)
    if not rows or not spot:
        raise RuntimeError("Upstox option chain was empty")
    now = pd.Timestamp.now(tz="UTC")
    csv, notes = build_chain_csv(upstox_rows(rows), float(spot), now, f"{name} (NSE)", "INR", "european")
    notes.update(source="Upstox API v2 option chain (your account)", underlying_key=key, fetched_at=now.isoformat())
    return csv, notes


# ------------------------------------------------------------------------------------------ Alpaca (US)
def alpaca_credentials() -> dict:
    kid = os.environ.get("ALPACA_API_KEY_ID", "").strip()
    sec = os.environ.get("ALPACA_API_SECRET_KEY", "").strip()
    if not kid or not sec:
        raise MissingCredentials("ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY are not set")
    return {"APCA-API-KEY-ID": kid, "APCA-API-SECRET-KEY": sec}


def parse_occ(sym: str):
    """'AAPL240426C00162500' -> ('AAPL', expiry 16:00 NY, 'C', 162.5)."""
    root, ymd, typ, strike = sym[:-15], sym[-15:-9], sym[-9], int(sym[-8:]) / 1000.0
    exp = pd.Timestamp(f"20{ymd[:2]}-{ymd[2:4]}-{ymd[4:]} 16:00", tz="America/New_York")
    return root, exp, typ, strike


def alpaca_rows(snapshots: dict) -> pd.DataFrame:
    out = []
    for sym, snap in snapshots.items():
        try:
            _, exp, typ, K = parse_occ(sym)
        except (ValueError, IndexError):
            continue
        qt = snap.get("latestQuote") or {}
        out.append(dict(expiry=exp, strike=K, option_type=typ, bid=qt.get("bp") or None, ask=qt.get("ap") or None,
                        volume=None, open_interest=None))
    return pd.DataFrame(out)


def fetch_alpaca(symbol: str, max_pages: int = 30) -> tuple[bytes, dict]:
    h = alpaca_credentials()
    trade = _get(f"{ALPACA}/v2/stocks/trades/latest?symbols={urllib.parse.quote(symbol)}&feed=iex", h)
    spot = ((trade.get("trades") or {}).get(symbol) or {}).get("p")
    if not spot:
        raise RuntimeError(f"no latest IEX trade for {symbol}")
    snaps, token, pages = {}, None, 0
    while pages < max_pages:
        url = f"{ALPACA}/v1beta1/options/snapshots/{urllib.parse.quote(symbol)}?feed=indicative&limit=1000"
        if token:
            url += f"&page_token={urllib.parse.quote(token)}"
        page = _get(url, h)
        snaps.update(page.get("snapshots") or {})
        token = page.get("next_page_token")
        pages += 1
        if not token:
            break
    if not snaps:
        raise RuntimeError(f"Alpaca returned no option snapshots for {symbol}")
    now = pd.Timestamp.now(tz="UTC")
    csv, notes = build_chain_csv(alpaca_rows(snaps), float(spot), now, f"{symbol} (US)", "USD", "american")
    notes.update(source="Alpaca options snapshots · indicative feed (delayed/modified) · spot = latest IEX trade",
                 fetched_at=now.isoformat(), truncated=bool(token))
    return csv, notes
