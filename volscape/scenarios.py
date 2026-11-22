"""European options portfolio and scenario repricing.

Accounting conventions (explicit):
* quantity  : signed number of contracts (+ long, - short).
* premium   : price per ONE unit of underlying paid (long) / received (short) at inception.
* multiplier: units of underlying per contract (default 100).
* initial cost = sum(quantity * premium * multiplier)   (> 0 = net debit paid).
* scenario value = sum(quantity * V_scenario * multiplier).
* P&L = scenario value - initial cost. It INCLUDES the premium paid/received and the
  multiplier. It EXCLUDES financing/carry on the premium cash, transaction costs, margin,
  and dividends paid on any hedge.

Scenario definition: spot S -> S*(1+ds), every leg's implied vol shifted in parallel by dv
(absolute vol points), calendar time advanced by h days. Each leg is fully repriced with BSM at
remaining T' = T - h/365; a leg with T' <= 0 settles at intrinsic value on the scenario spot.
Rates and dividend yields are held at the snapshot values of the leg's expiry.

Surface dynamics (how the leg's base vol responds to the spot move):
* sticky-strike         : sigma_leg = sigma_surface(k0 = ln(K / F0(T)), T) - the vol of a fixed
                          strike does not change when spot moves.
* sticky-log-moneyness  : sigma_leg = sigma_surface(k1 = ln(K / F1(T)), T) with F1 built from the
                          shocked spot - the smile moves with the forward, so a fixed strike
                          slides along the smile.
In both, the vol is read at the leg's ORIGINAL maturity T (no roll-down along the term
structure as time passes). A scenario is N/A when the required surface point is undefined (outside
observed coverage with extrapolation off) or when the shocked vol is <= 0.

Greeks are BSM Greeks at the leg's surface vol with the same conventions as pricing.greeks,
scaled by quantity * multiplier: delta in underlying units, gamma per 1 unit of spot,
vega per vol point, theta per calendar day, rho per 1 % rate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .pricing import bsm_price, greeks
from .surface import SurfaceModel

STICKY_STRIKE = "sticky-strike"
STICKY_MONEYNESS = "sticky-log-moneyness"


@dataclass
class Leg:
    option_type: str     # "C" / "P"
    strike: float
    T: float             # years to expiry at valuation (ACT/365)
    quantity: float
    premium: float       # per unit of underlying
    multiplier: float = 100.0
    rate: float = 0.0
    dividend_yield: float = 0.0
    label: str = ""


def leg_vol(leg: Leg, model: SurfaceModel, S0: float, S1: float, dynamics: str, extrapolate: bool):
    F_ref = (S0 if dynamics == STICKY_STRIKE else S1) * np.exp((leg.rate - leg.dividend_yield) * leg.T)
    k = np.log(leg.strike / F_ref)
    return model.iv(k, leg.T, extrapolate=extrapolate)


def initial_cost(legs: list[Leg]) -> float:
    return float(sum(l.quantity * l.premium * l.multiplier for l in legs))


def portfolio_greeks(legs: list[Leg], model: SurfaceModel, S0: float, extrapolate: bool = False) -> dict:
    tot = dict(delta=0.0, gamma=0.0, vega_volpt=0.0, theta_day=0.0, rho_pct=0.0, model_value=0.0)
    for l in legs:
        sig = float(leg_vol(l, model, S0, S0, STICKY_STRIKE, extrapolate))
        if not np.isfinite(sig) or sig <= 0 or l.T <= 0:
            return {k: np.nan for k in tot}
        g = greeks(S0, l.strike, l.T, l.rate, l.dividend_yield, sig, l.option_type == "C")
        n = l.quantity * l.multiplier
        tot["delta"] += n * float(g["delta"])
        tot["gamma"] += n * float(g["gamma"])
        tot["vega_volpt"] += n * float(g["vega"]) / 100
        tot["theta_day"] += n * float(g["theta"]) / 365
        tot["rho_pct"] += n * float(g["rho"]) / 100
        tot["model_value"] += n * float(bsm_price(S0, l.strike, l.T, l.rate, l.dividend_yield, sig, l.option_type == "C"))
    return tot


def scenario_pnl(legs: list[Leg], model: SurfaceModel, S0: float, spot_shocks, vol_shocks_volpts,
                 horizon_days: float = 0.0, dynamics: str = STICKY_STRIKE, extrapolate: bool = False):
    """P&L matrix of shape (len(vol_shocks), len(spot_shocks)); NaN = N/A (not reliably evaluable)."""
    ds = np.asarray(spot_shocks, float)
    dv = np.asarray(vol_shocks_volpts, float) / 100
    S1 = S0 * (1 + ds)
    value = np.zeros((len(dv), len(ds)))
    h = horizon_days / 365.0
    for l in legs:
        Tr = l.T - h
        is_call = l.option_type == "C"
        n = l.quantity * l.multiplier
        if Tr <= 0:
            intrinsic = np.maximum(S1 - l.strike, 0) if is_call else np.maximum(l.strike - S1, 0)
            value += n * intrinsic[None, :]
            continue
        base = np.array([float(leg_vol(l, model, S0, s1, dynamics, extrapolate)) for s1 in S1])
        sig = base[None, :] + dv[:, None]
        bad = ~np.isfinite(sig) | (sig <= 0)
        px = bsm_price(np.broadcast_to(S1, sig.shape), l.strike, Tr, l.rate, l.dividend_yield,
                       np.where(bad, 0.0, sig), is_call)
        value += n * np.where(bad, np.nan, px)
    return value - initial_cost(legs)


def legs_from_frame(df: pd.DataFrame, chain: pd.DataFrame) -> tuple[list[Leg], list[str]]:
    """Build legs from an editor table (columns: type, expiry, strike, quantity, premium, multiplier).
    Rows with invalid fields are skipped and reported - never guessed."""
    legs, errors = [], []
    exp_info = chain.groupby("expiry").agg(T=("T", "first"), rate=("rate", "first"), q=("dividend_yield", "first"))
    exp_map = {e.strftime("%Y-%m-%d"): e for e in exp_info.index}
    for i, r in df.reset_index(drop=True).iterrows():
        try:
            typ = str(r["type"]).upper()[:1]
            if typ not in ("C", "P"):
                raise ValueError("type must be Call or Put")
            e = exp_map.get(str(r["expiry"]))
            if e is None:
                raise ValueError(f"expiry {r['expiry']} not in snapshot")
            K, qty, prem, mult = (float(r[c]) for c in ("strike", "quantity", "premium", "multiplier"))
            if not all(np.isfinite([K, qty, prem, mult])) or K <= 0 or mult <= 0 or prem < 0:
                raise ValueError("strike/multiplier must be > 0, premium >= 0, all finite")
            info = exp_info.loc[e]
            legs.append(Leg(typ, K, float(info["T"]), qty, prem, mult, float(info["rate"]), float(info["q"]),
                            f"{qty:+g} {typ} {K:g} {str(r['expiry'])}"))
        except (ValueError, TypeError, KeyError) as exc:
            errors.append(f"leg {i + 1}: {exc}")
    return legs, errors
