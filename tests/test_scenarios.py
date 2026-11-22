import numpy as np
import pandas as pd
import pytest

from volscape.pricing import bsm_price, greeks
from volscape.scenarios import (STICKY_MONEYNESS, STICKY_STRIKE, Leg, initial_cost,
                                portfolio_greeks, scenario_pnl)
from volscape.surface import BaselineSlice, SurfaceModel

TZ = "UTC"
S0, R, Q = 100.0, 0.03, 0.01


def _surface(skew=0.0):
    """Two-slice baseline surface; IV(k) = 0.2 + skew*k (skew<0 = equity-style)."""
    k = np.linspace(-0.6, 0.6, 61)
    sl = []
    for e, T in ((pd.Timestamp("2026-09-01", tz=TZ), 0.25), (pd.Timestamp("2027-06-01", tz=TZ), 1.0)):
        sl.append(BaselineSlice(e, T, k, (0.2 + skew * k) ** 2 * T))
    return SurfaceModel("baseline", sl)


def _leg(typ="C", K=100.0, T=0.25, qty=1.0, prem=None, mult=100.0):
    if prem is None:
        prem = float(bsm_price(S0, K, T, R, Q, 0.2 + 0 * K, typ == "C"))
    return Leg(typ, K, T, qty, prem, mult, R, Q)


def test_cost_pnl_premium_and_multiplier_accounting():
    m = _surface()
    leg = _leg(prem=5.0, qty=2, mult=100)
    assert initial_cost([leg]) == 1000.0
    pnl = scenario_pnl([leg], m, S0, [0.0], [0.0])
    fair = float(bsm_price(S0, 100, 0.25, R, Q, 0.2, True))
    assert pnl[0, 0] == pytest.approx(2 * 100 * (fair - 5.0), rel=1e-12)
    # linear in multiplier and quantity sign
    short = _leg(prem=5.0, qty=-2, mult=10)
    assert scenario_pnl([short], m, S0, [0.0], [0.0])[0, 0] == pytest.approx(-pnl[0, 0] / 10, rel=1e-12)


def test_portfolio_greek_signs_and_units():
    m = _surface()
    straddle = [_leg("C"), _leg("P")]
    g = portfolio_greeks(straddle, m, S0)
    assert g["gamma"] > 0 and g["vega_volpt"] > 0 and g["theta_day"] < 0
    single = greeks(S0, 100, 0.25, R, Q, 0.2, True)
    g1 = portfolio_greeks([_leg("C")], m, S0)
    assert g1["delta"] == pytest.approx(100 * float(single["delta"]))
    assert g1["vega_volpt"] == pytest.approx(100 * float(single["vega"]) / 100)
    assert g1["theta_day"] == pytest.approx(100 * float(single["theta"]) / 365)
    short_put = portfolio_greeks([_leg("P", qty=-1)], m, S0)
    assert short_put["delta"] > 0 and short_put["gamma"] < 0


def test_expiry_settles_at_intrinsic_on_scenario_spot():
    m = _surface()
    leg = _leg("P", K=100, T=0.25, prem=3.0, qty=1, mult=100)
    shocks = np.array([-0.2, 0.0, 0.2])
    pnl = scenario_pnl([leg], m, S0, shocks, [0.0, 5.0], horizon_days=120)   # beyond 91.25d expiry
    intrinsic = np.maximum(100 - S0 * (1 + shocks), 0)
    expected = 100 * (intrinsic - 3.0)
    assert np.allclose(pnl, expected[None, :])   # vol shock irrelevant after expiry


def test_time_forward_reduces_long_option_value():
    m = _surface()
    leg = _leg("C")
    p0 = scenario_pnl([leg], m, S0, [0.0], [0.0], 0)[0, 0]
    p30 = scenario_pnl([leg], m, S0, [0.0], [0.0], 30)[0, 0]
    assert p30 < p0


def test_sticky_rules_identical_on_flat_and_differ_on_skew():
    leg = _leg("C", K=100)
    flat = _surface(0.0)
    a = scenario_pnl([leg], flat, S0, [-0.1, 0.1], [0.0], 0, STICKY_STRIKE)
    b = scenario_pnl([leg], flat, S0, [-0.1, 0.1], [0.0], 0, STICKY_MONEYNESS)
    assert np.allclose(a, b)
    sk = _surface(-0.3)
    a = scenario_pnl([leg], sk, S0, [-0.1], [0.0], 0, STICKY_STRIKE)
    b = scenario_pnl([leg], sk, S0, [-0.1], [0.0], 0, STICKY_MONEYNESS)
    # spot down: under sticky-moneyness the 100 strike becomes OTM-call side (k>0) where
    # negative skew gives LOWER vol, so the long call is worth less than under sticky-strike.
    assert b[0, 0] < a[0, 0]


def test_unevaluable_scenarios_are_na():
    m = _surface()
    leg = _leg("C", K=100)
    pnl = scenario_pnl([leg], m, S0, [0.0, -0.6], [0.0, -25.0], 0, STICKY_MONEYNESS)
    assert np.isnan(pnl[1, 0])            # vol shock drives IV below zero
    assert np.isnan(pnl[0, 1])            # k = ln(100/40F) outside observed range, no extrapolation
    assert np.isfinite(pnl[0, 0])
    far = _leg("C", K=400)                # outside surface coverage -> Greeks N/A
    assert np.isnan(portfolio_greeks([far], m, S0)["delta"])
