"""Live option-chain snapshots from Deribit's public REST API (no keys, no account).

Endpoint: GET https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option
Each row carries best bid/ask in units of the BASE coin (BTC/ETH per 1 unit of underlying), the
instrument's underlying (futures or synthetic) price, the index price and Deribit's own mark IV.

Conversion into VOLSCAPE's CSV schema (verified empirically: re-solving IV from Deribit mark prices
under this convention reproduces Deribit `mark_iv`, OTM median gap <= 0.2 vol pt, longer tenors <= 0.03):
  * F_T      = median `underlying_price` over that expiry's instruments (forward for the expiry)
  * S        = index price (`estimated_delivery_price`), one per snapshot
  * USD quote = coin quote * F_T          (Deribit options are European, priced off the forward)
  * r = 0 and q_T = -ln(F_T / S) / T      so that F = S e^{(r-q)T} reproduces F_T exactly
  * expiry   = Deribit expiry date at 08:00 UTC; T from the snapshot fetch time, ACT/365
Rows without an ask are skipped and counted (no offer -> no two-sided quote). A missing bid is
written as 0 (no buyer at a positive price) and is flagged ZERO_BID by validation. Nothing else is
altered; validation, IV solving and fitting then run unchanged.
"""
from __future__ import annotations

import io
import json
import math
import statistics
import urllib.request
from datetime import datetime, timezone

import pandas as pd

API = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency={cur}&kind=option"
HEADERS = {"User-Agent": "volscape/1.0", "Accept": "application/json"}
CURRENCIES = ("BTC", "ETH")
_MON = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}


def fetch_book(currency: str, timeout: float = 15.0) -> list[dict]:
    if currency not in CURRENCIES:
        raise ValueError(f"unsupported currency {currency}")
    req = urllib.request.Request(API.format(cur=currency), headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.loads(r.read().decode("utf-8"))
    if "result" not in payload:
        raise RuntimeError(f"unexpected Deribit response: {str(payload)[:200]}")
    return payload["result"]


def parse_expiry(code: str) -> datetime:
    """'27NOV26' -> 2026-11-27 08:00 UTC (Deribit option expiry time)."""
    day, mon, yr = int(code[:-5]), _MON[code[-5:-2]], 2000 + int(code[-2:])
    return datetime(yr, mon, day, 8, 0, tzinfo=timezone.utc)


def to_chain(rows: list[dict], fetched_at: datetime, currency: str) -> tuple[bytes, dict]:
    """Deribit book summary -> VOLSCAPE CSV bytes + conversion notes (counts, index, forwards)."""
    parsed, no_ask, bad = [], 0, 0
    for x in rows:
        try:
            _, exp_code, strike, typ = x["instrument_name"].split("-")
            exp = parse_expiry(exp_code)
            K = float(strike)
        except (KeyError, ValueError):
            bad += 1
            continue
        if x.get("ask_price") in (None, 0):
            no_ask += 1
            continue
        parsed.append(dict(exp=exp, K=K, typ=typ, bid=x.get("bid_price") or 0.0, ask=x["ask_price"],
                           F=x.get("underlying_price"), S=x.get("estimated_delivery_price"),
                           vol=x.get("volume"), oi=x.get("open_interest")))
    if not parsed:
        raise RuntimeError("Deribit returned no two-sided option quotes")
    S = statistics.median(p["S"] for p in parsed if p["S"])
    fwd = {}
    for p in parsed:
        if p["F"]:
            fwd.setdefault(p["exp"], []).append(p["F"])
    fwd = {e: statistics.median(v) for e, v in fwd.items()}
    out = []
    vt = fetched_at.astimezone(timezone.utc).replace(microsecond=0)
    for p in parsed:
        F = fwd.get(p["exp"])
        if F is None:
            bad += 1
            continue
        T = (p["exp"] - vt).total_seconds() / (365 * 86400)
        q = -math.log(F / S) / T if T > 0 else 0.0   # expired rows are rejected by validation anyway
        out.append(dict(valuation_time=vt.isoformat(), expiry=p["exp"].isoformat(), strike=p["K"],
                        option_type=p["typ"], bid=round(p["bid"] * F, 6), ask=round(p["ask"] * F, 6),
                        spot=S, rate=0.0, dividend_yield=q, underlying=f"{currency} (Deribit index)",
                        currency="USD", volume=p["vol"], open_interest=p["oi"], exercise_style="european"))
    df = pd.DataFrame(out).sort_values(["expiry", "strike", "option_type"])
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    notes = dict(source="Deribit public API get_book_summary_by_currency", currency=currency,
                 fetched_at=vt.isoformat(), index_price=S, n_raw=len(rows), n_no_ask=no_ask, n_unparsed=bad,
                 n_written=len(df), forwards={e.date().isoformat(): f for e, f in sorted(fwd.items())})
    return buf.getvalue().encode(), notes
