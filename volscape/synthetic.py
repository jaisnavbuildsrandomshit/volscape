"""Seeded SYNTHETIC option-chain generator (explicitly disclosed model; NOT market data).

Generating model - SSVI (Gatheral & Jacquier 2014) total variance:
    w(k, T) = theta_T / 2 * (1 + rho*phi*k + sqrt((phi*k + rho)^2 + 1 - rho^2))
    theta_T = atm_vol(T)^2 * T,  atm_vol(T) = v_inf + (v0 - v_inf) * exp(-T / tau)
    phi(theta) = eta / (theta^gamma * (1 + theta)^(1 - gamma))      (power-law)
With eta*(1+|rho|) <= 2 and gamma in (0, 0.5], SSVI is free of static arbitrage, and each
slice equals a raw SVI slice exactly (see ``ssvi_to_raw``), which makes clean data an
identified recovery target for the SVI calibrator.

Quotes: mid = BSM(model IV [+ seeded Gaussian IV noise]); half-spread =
max(min_half_tick*tick, spread_vega_volpts * vega_per_volpt + spread_price_frac * price),
bid/ask rounded outward to the tick, bid floored at 0 by construction. These widths are
stylised and make no claim of resembling any real market.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd

from .pricing import bsm_price, greeks


@dataclass(frozen=True)
class SyntheticSpec:
    seed: int = 7
    valuation_time: str = "2026-06-15T20:00:00+00:00"
    underlying: str = "SYNTH-IDX"
    currency: str = "USD"
    spot: float = 100.0
    rate: float = 0.04
    dividend_yield: float = 0.015
    expiry_days: tuple = (7, 14, 30, 60, 91, 137, 182, 273, 365, 548, 730)
    v0: float = 0.24          # short-end ATM vol
    v_inf: float = 0.19       # long-end ATM vol
    tau: float = 0.35         # term-structure decay (years)
    rho: float = -0.55
    eta: float = 1.05
    gamma: float = 0.42
    n_std: float = 2.6        # strike span in ATM std-devs each side
    n_strikes: int = 21
    iv_noise_volpts: float = 0.0
    tick: float = 0.01
    min_half_ticks: float = 1.0
    spread_vega_volpts: float = 0.35
    spread_price_frac: float = 0.02

    def to_dict(self):
        d = asdict(self)
        d["expiry_days"] = list(self.expiry_days)
        return d


def atm_total_variance(T, spec: SyntheticSpec):
    v = spec.v_inf + (spec.v0 - spec.v_inf) * np.exp(-np.asarray(T) / spec.tau)
    return v * v * T


def phi(theta, spec: SyntheticSpec):
    return spec.eta / (theta ** spec.gamma * (1.0 + theta) ** (1.0 - spec.gamma))


def ssvi_w(k, T, spec: SyntheticSpec):
    th = atm_total_variance(T, spec)
    p = phi(th, spec)
    r = spec.rho
    return th / 2.0 * (1 + r * p * k + np.sqrt((p * k + r) ** 2 + 1 - r * r))


def ssvi_to_raw(T, spec: SyntheticSpec):
    """Exact raw-SVI parameters (a, b, rho, m, sigma) of the SSVI slice at T."""
    th = float(atm_total_variance(T, spec))
    p = float(phi(th, spec))
    r = spec.rho
    return dict(a=th / 2 * (1 - r * r), b=th * p / 2, rho=r, m=-r / p, sigma=np.sqrt(1 - r * r) / p)


def _round_strike(K, F):
    step = 0.5 if F < 50 else (1.0 if F < 250 else 5.0)
    return np.round(K / step) * step


def generate_chain(spec: SyntheticSpec = SyntheticSpec()) -> pd.DataFrame:
    rng = np.random.default_rng(spec.seed)
    vt = pd.Timestamp(spec.valuation_time)
    rows = []
    for d in spec.expiry_days:
        exp = vt + pd.Timedelta(days=d)
        T = d / 365.0
        F = spec.spot * np.exp((spec.rate - spec.dividend_yield) * T)
        atm_sd = np.sqrt(float(atm_total_variance(T, spec)))
        ks = np.linspace(-spec.n_std * atm_sd * 1.15, spec.n_std * atm_sd * 0.85, spec.n_strikes)
        strikes = np.unique(_round_strike(F * np.exp(ks), F))
        for K in strikes:
            k = np.log(K / F)
            iv_true = np.sqrt(ssvi_w(k, T, spec) / T)
            for typ in ("C", "P"):
                iv = iv_true + (rng.normal(0.0, spec.iv_noise_volpts / 100.0) if spec.iv_noise_volpts > 0 else 0.0)
                is_call = typ == "C"
                mid = float(bsm_price(spec.spot, K, T, spec.rate, spec.dividend_yield, iv, is_call))
                vega = float(greeks(spec.spot, K, T, spec.rate, spec.dividend_yield, iv, is_call)["vega"]) / 100
                half = max(spec.min_half_ticks * spec.tick,
                           spec.spread_vega_volpts * vega + spec.spread_price_frac * mid)
                bid = max(np.floor((mid - half) / spec.tick) * spec.tick, 0.0)
                ask = np.ceil((mid + half) / spec.tick) * spec.tick
                vol_seed = rng.integers(0, 5000)
                rows.append(dict(
                    valuation_time=vt.isoformat(), expiry=exp.isoformat(), strike=float(K),
                    option_type=typ, bid=round(bid, 6), ask=round(ask, 6), spot=spec.spot,
                    rate=spec.rate, dividend_yield=spec.dividend_yield,
                    underlying=spec.underlying, currency=spec.currency,
                    volume=int(vol_seed * np.exp(-abs(k) * 3)),
                    open_interest=int(vol_seed * 7 * np.exp(-abs(k) * 2)),
                ))
    return pd.DataFrame(rows)


def make_dirty_chain(spec: SyntheticSpec = SyntheticSpec()) -> tuple[pd.DataFrame, list[str]]:
    """Clean synthetic subset plus deliberately injected defects (documented per line)."""
    base = generate_chain(spec)
    base = base[base["expiry"].isin(sorted(base["expiry"].unique())[2:6])].reset_index(drop=True)
    df = base.astype(object).copy()
    notes = []

    def inject(i, col, val, why):
        df.at[i, col] = val
        notes.append(f"line {i + 2}: {why}")

    inject(3, "bid", "", "missing bid")
    inject(8, "valuation_time", "2026-06-15 20:00:00", "timestamp without timezone")
    inject(12, "strike", -95.0, "negative strike")
    inject(17, "ask", float(df.at[17, "bid"]) - 0.05 if float(df.at[17, "bid"]) > 0.05 else 0.0, "ask below bid")
    inject(21, "option_type", "X", "invalid option type")
    inject(26, "spot", 101.25, "spot inconsistent with snapshot")
    inject(30, "expiry", "2026-06-10T20:00:00+00:00", "expired contract")
    inject(35, "bid", "abc", "non-numeric bid")
    inject(40, "rate", 0.09, "rate inconsistent within expiry")
    up = float(df.at[44, "spot"]) * 1.2
    inject(44, "bid", round(up, 2), "call bid above upper bound S*e^{-qT}")
    df.at[44, "ask"] = round(up + 0.5, 2)
    inject(50, "ask", round(float(df.at[50, "ask"]) * 6 + 1, 2), "suspiciously wide spread (flag, kept)")
    df["exercise_style"] = "european"
    inject(55, "exercise_style", "american", "American-style contract")
    dup = df.iloc[[60]].copy()
    dup["bid"] = float(dup["bid"].iloc[0]) + 0.02
    df = pd.concat([df, dup], ignore_index=True)
    notes.append(f"line 62 and line {len(df) + 1}: duplicate contract with different quote")
    return df, notes
