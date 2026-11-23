"""Shared conversion of raw option quotes into VOLSCAPE's CSV schema for providers that do NOT
publish a forward or a rate (Upstox / NSE, Alpaca / US).

Forward and discount factor per expiry come from the market itself (put-call parity), not from an
assumed rate. For European options:   C - P = DF * F - DF * K.
A least-squares line through (K, C_mid - P_mid) over the near-the-money pairs gives slope = -DF and
intercept = DF * F. Then r_T = -ln(DF)/T and q_T = r_T - ln(F/S)/T, so that F = S e^{(r-q)T}.
Expiries with fewer than MIN_PAIRS usable pairs, or an implausible DF, are skipped and counted:
the forward is never guessed. For American options parity is only approximate (early exercise);
callers flag such chains as American so validation requires the explicit European opt-in.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_PAIRS = 3
N_NEAR = 10            # pairs closest to spot used in the parity regression
DF_RANGE = (0.80, 1.02)


@dataclass
class ParityFit:
    df: float
    forward: float
    n_pairs: int
    rmse: float


def parity_forward(K, c_mid, p_mid, spot: float, n_near: int = N_NEAR) -> ParityFit | None:
    K, c, p = (np.asarray(x, float) for x in (K, c_mid, p_mid))
    ok = np.isfinite(K) & np.isfinite(c) & np.isfinite(p) & (c > 0) & (p > 0)
    K, y = K[ok], (c - p)[ok]
    if len(K) < MIN_PAIRS:
        return None
    near = np.argsort(np.abs(K - spot))[:n_near]
    K, y = K[near], y[near]
    if len(np.unique(K)) < MIN_PAIRS:
        return None
    A = np.column_stack([np.ones_like(K), -K])
    (a, df), *_ = np.linalg.lstsq(A, y, rcond=None)
    if not (DF_RANGE[0] < df <= DF_RANGE[1]):
        return None
    F = a / df
    if not np.isfinite(F) or F <= 0:
        return None
    rmse = float(np.sqrt(np.mean((A @ np.array([a, df]) - y) ** 2)))
    return ParityFit(float(df), float(F), int(len(K)), rmse)


def build_chain_csv(quotes: pd.DataFrame, spot: float, valuation: pd.Timestamp, underlying: str,
                    currency: str, exercise: str) -> tuple[bytes, dict]:
    """quotes columns: expiry (tz-aware Timestamp), strike, option_type ('C'/'P'), bid, ask, [volume, open_interest].
    Rows without an ask are dropped and counted; a missing bid becomes 0 (flagged ZERO_BID later)."""
    q = quotes.copy()
    n_raw = len(q)
    no_ask = int((~(q.ask.astype(float) > 0)).sum())
    q = q[q.ask.astype(float) > 0].copy()
    q["bid"] = q.bid.fillna(0.0).astype(float)
    q["mid"] = (q.bid + q.ask) / 2
    vt = valuation.tz_convert("UTC").floor("s")
    out, fits, skipped = [], {}, {}
    for exp, g in q.groupby("expiry"):
        T = (exp - vt).total_seconds() / (365 * 86400)
        if T <= 0:
            skipped[exp.date().isoformat()] = "expired"
            continue
        piv = g.pivot_table(index="strike", columns="option_type", values="mid", aggfunc="first")
        if not {"C", "P"} <= set(piv.columns):
            skipped[exp.date().isoformat()] = "no call/put pairs"
            continue
        fit = parity_forward(piv.index.to_numpy(), piv["C"].to_numpy(), piv["P"].to_numpy(), spot)
        if fit is None:
            skipped[exp.date().isoformat()] = f"parity forward not identified (<{MIN_PAIRS} two-sided pairs or implausible DF)"
            continue
        r = -math.log(fit.df) / T
        qy = r - math.log(fit.forward / spot) / T
        fits[exp.date().isoformat()] = dict(forward=fit.forward, discount_factor=fit.df, rate=r, div_yield=qy,
                                            n_pairs=fit.n_pairs, parity_rmse=fit.rmse)
        for row in g.itertuples(index=False):
            out.append(dict(valuation_time=vt.isoformat(), expiry=exp.isoformat(), strike=float(row.strike),
                            option_type=row.option_type, bid=float(row.bid), ask=float(row.ask), spot=float(spot),
                            rate=r, dividend_yield=qy, underlying=underlying, currency=currency,
                            volume=getattr(row, "volume", None), open_interest=getattr(row, "open_interest", None),
                            exercise_style=exercise))
    if not out:
        raise RuntimeError("no expiry had enough two-sided call/put pairs to identify a forward: " + str(skipped))
    df = pd.DataFrame(out).sort_values(["expiry", "strike", "option_type"])
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    notes = dict(n_raw=n_raw, n_no_ask=no_ask, n_written=len(df), spot=spot, forwards=fits, skipped_expiries=skipped)
    return buf.getvalue().encode(), notes
