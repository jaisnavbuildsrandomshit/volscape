"""Implied-volatility inversion by bracketed Brent root finding.

Statuses (distinct, never collapsed):
  OK             converged, |model - target| within price tolerance
  INVALID_INPUT  nonfinite / nonpositive S, K, or nonfinite T, r, q, price
  EXPIRED        T <= 0: no time value, IV undefined
  INVALID_PRICE  target price outside model-free bounds [lower, upper]
  NO_BRACKET     price inside bounds but not attained for sigma in [sigma_lo, sigma_hi]
                 (e.g. price at intrinsic -> would need sigma below sigma_lo)
  NOT_CONVERGED  Brent failed, or final price residual exceeds tolerance
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
from scipy.optimize import brentq

from .pricing import bsm_price, greeks, price_bounds

OK = "OK"
INVALID_INPUT = "INVALID_INPUT"
EXPIRED = "EXPIRED"
INVALID_PRICE = "INVALID_PRICE"
NO_BRACKET = "NO_BRACKET"
NOT_CONVERGED = "NOT_CONVERGED"
STATUSES = (OK, INVALID_INPUT, EXPIRED, INVALID_PRICE, NO_BRACKET, NOT_CONVERGED)


def _ncdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


@dataclass(frozen=True)
class IVSettings:
    sigma_lo: float = 1e-4
    sigma_hi: float = 5.0
    xtol: float = 1e-12
    max_iter: int = 200
    price_rtol: float = 1e-9     # residual tolerance relative to max(price, 1e-8*S)
    bound_tol: float = 1e-10     # absolute slack on bound checks (currency)

    def to_dict(self):
        return asdict(self)


def implied_vol(price, S, K, T, r, q, is_call, settings: IVSettings = IVSettings()):
    """Scalar IV. Returns (iv or nan, status)."""
    if not all(np.isfinite(v) for v in (price, S, K, T, r, q)) or S <= 0 or K <= 0:
        return np.nan, INVALID_INPUT
    if T <= 0:
        return np.nan, EXPIRED
    lo_b, hi_b = (float(x) for x in price_bounds(S, K, T, r, q, is_call))
    if price < 0 or price < lo_b - settings.bound_tol or price > hi_b + settings.bound_tol:
        return np.nan, INVALID_PRICE

    F = S * math.exp((r - q) * T)
    df = math.exp(-r * T)
    sqrtT = math.sqrt(T)
    lnFK = math.log(F / K)

    def f(sig):
        # scalar BSM (identical formula to pricing.bsm_price, sigma > 0 here)
        vs = sig * sqrtT
        d1 = lnFK / vs + 0.5 * vs
        d2 = d1 - vs
        if is_call:
            v = df * (F * _ncdf(d1) - K * _ncdf(d2))
        else:
            v = df * (K * _ncdf(-d2) - F * _ncdf(-d1))
        return v - price

    f_lo, f_hi = f(settings.sigma_lo), f(settings.sigma_hi)
    if f_lo == 0.0:
        return settings.sigma_lo, OK
    if f_hi == 0.0:
        return settings.sigma_hi, OK
    if f_lo > 0 or f_hi < 0:
        return np.nan, NO_BRACKET
    try:
        sig, info = brentq(f, settings.sigma_lo, settings.sigma_hi, xtol=settings.xtol,
                           rtol=4 * np.finfo(float).eps, maxiter=settings.max_iter,
                           full_output=True, disp=False)
    except (RuntimeError, ValueError):
        return np.nan, NOT_CONVERGED
    tol = max(settings.price_rtol * max(price, 1e-8 * S), 1e-12)
    if not info.converged or abs(f(sig)) > tol:
        return np.nan, NOT_CONVERGED
    return float(sig), OK


def implied_vol_array(price, S, K, T, r, q, is_call, settings: IVSettings = IVSettings()):
    arrs = np.broadcast_arrays(*(np.asarray(x, float) for x in (price, S, K, T, r, q)))
    is_call = np.broadcast_to(np.asarray(is_call, bool), arrs[0].shape)
    iv = np.full(arrs[0].shape, np.nan)
    status = np.empty(arrs[0].shape, dtype=object)
    for i in np.ndindex(arrs[0].shape):
        iv[i], status[i] = implied_vol(*(float(a[i]) for a in arrs), bool(is_call[i]), settings)
    return iv, status


def solve_chain_iv(df, settings: IVSettings = IVSettings(), min_vega_volpt: float = 0.005):
    """Solve bid/mid/ask IV for an accepted chain (columns from data.validate_chain).

    Adds iv_bid/iv_mid/iv_ask and their statuses, vega at mid IV per vol point, an IV
    half-width uncertainty = (ask - bid)/2 / vega (vol decimal, first order), and a LOW_VEGA
    flag when vega per vol point < ``min_vega_volpt`` currency units: there, one price tick
    moves IV by many vol points and the inversion is numerically unreliable.
    """
    out = df.copy()
    is_call = (out["option_type"] == "C").to_numpy()
    args = [out[c].to_numpy(float) for c in ("spot", "strike", "T", "rate", "dividend_yield")]
    for side in ("bid", "mid", "ask"):
        iv, st = implied_vol_array(out[side].to_numpy(float), *args, is_call, settings)
        out[f"iv_{side}"] = iv
        out[f"iv_{side}_status"] = st
    g = greeks(*args, out["iv_mid"].to_numpy(float), is_call)
    vega_pt = g["vega"] / 100.0
    out["vega_volpt"] = vega_pt
    half = (out["ask"] - out["bid"]).to_numpy(float) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        out["iv_halfwidth"] = np.where(vega_pt > 0, half / (vega_pt * 100.0), np.nan)
    out["low_vega"] = ~(vega_pt >= min_vega_volpt)
    return out
