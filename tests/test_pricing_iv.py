import numpy as np
import pytest

from volscape.iv import (EXPIRED, INVALID_INPUT, INVALID_PRICE, NO_BRACKET, OK, IVSettings,
                         implied_vol, implied_vol_array)
from volscape.pricing import bsm_price, greeks, parity_residual, price_bounds


# ---------------------------------------------------------------- known answers (textbook)
def test_hull_call_put_known_values():
    # Hull, Options Futures & Other Derivatives: S=42, K=40, r=10%, sigma=20%, T=0.5
    assert bsm_price(42, 40, 0.5, 0.10, 0.0, 0.20, True) == pytest.approx(4.7594, abs=5e-5)
    assert bsm_price(42, 40, 0.5, 0.10, 0.0, 0.20, False) == pytest.approx(0.8086, abs=5e-5)


def test_haug_merton_dividend_put():
    # Haug, Complete Guide to Option Pricing Formulas: generalized BSM put with continuous yield
    assert bsm_price(100, 95, 0.5, 0.10, 0.05, 0.20, False) == pytest.approx(2.4648, abs=5e-5)


def test_hull_greeks_known_values():
    # Hull example: S=49, K=50, r=5%, sigma=20%, T=20 weeks (0.3846y)
    g = greeks(49, 50, 0.3846, 0.05, 0.0, 0.20, True)
    assert float(g["delta"]) == pytest.approx(0.522, abs=1e-3)
    assert float(g["gamma"]) == pytest.approx(0.066, abs=1e-3)
    assert float(g["vega"]) == pytest.approx(12.1, abs=0.05)      # per 1.00 vol
    assert float(g["theta"]) == pytest.approx(-4.31, abs=0.01)    # per year
    assert float(g["rho"]) == pytest.approx(8.91, abs=0.01)       # per 1.00 rate


# ---------------------------------------------------------------- parity, bounds, limits
GRID = [(S, K, T, r, q, s) for S in (50.0, 100.0) for K in (40.0, 100.0, 180.0)
        for T in (0.01, 0.5, 3.0) for r in (-0.01, 0.05) for q in (0.0, 0.03) for s in (0.05, 0.3, 1.2)]


@pytest.mark.parametrize("S,K,T,r,q,s", GRID)
def test_put_call_parity_and_bounds(S, K, T, r, q, s):
    c = bsm_price(S, K, T, r, q, s, True)
    p = bsm_price(S, K, T, r, q, s, False)
    assert parity_residual(c, p, S, K, T, r, q) == pytest.approx(0.0, abs=1e-11 * max(S, K))
    for val, is_call in ((c, True), (p, False)):
        lo, hi = price_bounds(S, K, T, r, q, is_call)
        assert lo - 1e-12 <= val <= hi + 1e-12


def test_expiry_and_zero_vol_limits():
    # at expiry: undiscounted intrinsic
    assert bsm_price(105, 100, 0.0, 0.05, 0.02, 0.3, True) == 5.0
    assert bsm_price(95, 100, -0.1, 0.05, 0.02, 0.3, False) == 5.0
    # zero vol: discounted forward intrinsic, continuous with tiny vol
    S, K, T, r, q = 100, 95, 1.0, 0.05, 0.02
    F = S * np.exp((r - q) * T)
    assert bsm_price(S, K, T, r, q, 0.0, True) == pytest.approx(np.exp(-r * T) * (F - K), rel=1e-14)
    assert bsm_price(S, K, T, r, q, 0.0, False) == 0.0
    assert bsm_price(S, K, T, r, q, 1e-7, True) == pytest.approx(bsm_price(S, K, T, r, q, 0.0, True), rel=1e-12)
    # near expiry converges to intrinsic, and Greeks are NaN (undefined) rather than silently 0
    assert bsm_price(105, 100, 1e-10, 0.05, 0.0, 0.3, True) == pytest.approx(5.0, abs=1e-6)
    assert np.isnan(greeks(100, 100, 0.0, 0.05, 0.0, 0.2, True)["delta"])
    # vol -> large: call approaches upper bound S e^{-qT}
    assert bsm_price(S, K, T, r, q, 50.0, True) == pytest.approx(S * np.exp(-q * T), rel=1e-6)


# ---------------------------------------------------------------- finite-difference Greeks
@pytest.mark.parametrize("S,K,T,r,q,s,call", [
    (100, 100, 0.25, 0.03, 0.01, 0.2, True), (100, 80, 1.5, 0.05, 0.02, 0.35, False),
    (100, 130, 0.08, 0.01, 0.0, 0.6, True), (2500, 2400, 2.0, 0.04, 0.015, 0.18, False)])
def test_greeks_match_central_differences(S, K, T, r, q, s, call):
    g = greeks(S, K, T, r, q, s, call)
    P = lambda **kw: bsm_price(kw.get("S", S), K, kw.get("T", T), kw.get("r", r), q, kw.get("s", s), call)
    hS, hs, hT, hr = 1e-4 * S, 1e-5, 1e-6, 1e-6
    fd = {
        "delta": (P(S=S + hS) - P(S=S - hS)) / (2 * hS),
        "gamma": (P(S=S + hS) - 2 * P() + P(S=S - hS)) / hS ** 2,
        "vega": (P(s=s + hs) - P(s=s - hs)) / (2 * hs),
        "theta": -(P(T=T + hT) - P(T=T - hT)) / (2 * hT),
        "rho": (P(r=r + hr) - P(r=r - hr)) / (2 * hr),
    }
    scale = {"delta": 1.0, "gamma": 1.0 / S, "vega": S, "theta": S, "rho": K * T}
    for k, v in fd.items():
        assert float(g[k]) == pytest.approx(v, rel=1e-5, abs=1e-7 * scale[k]), k


# ---------------------------------------------------------------- implied vol
@pytest.mark.parametrize("T", [0.02, 0.25, 1.0, 3.0])
@pytest.mark.parametrize("sigma", [0.05, 0.2, 0.6, 1.5])
def test_iv_round_trip(T, sigma):
    S, r, q = 100.0, 0.03, 0.01
    F = S * np.exp((r - q) * T)
    sd = sigma * np.sqrt(T)
    for k in np.linspace(-2.5 * sd, 2.5 * sd, 11):
        K = F * np.exp(k)
        for call in (True, False):
            price = bsm_price(S, K, T, r, q, sigma, call)
            iv, st = implied_vol(price, S, K, T, r, q, call)
            vega = float(greeks(S, K, T, r, q, sigma, call)["vega"])
            assert st == OK, (k, call)
            # recoverable accuracy is limited by price precision / vega
            tol = max(1e-9, 1e-13 * S / max(vega, 1e-300))
            assert abs(iv - sigma) < tol, (k, call, iv, sigma)


def test_iv_status_codes_are_distinct():
    S, K, T, r, q = 100.0, 100.0, 0.5, 0.05, 0.0
    lo, hi = (float(x) for x in price_bounds(S, K, T, r, q, True))
    assert implied_vol(lo - 0.01, S, K, T, r, q, True)[1] == INVALID_PRICE
    assert implied_vol(hi + 0.01, S, K, T, r, q, True)[1] == INVALID_PRICE
    assert implied_vol(-1.0, S, K, T, r, q, True)[1] == INVALID_PRICE
    # exactly the intrinsic lower bound needs sigma below sigma_lo -> no bracket
    assert implied_vol(lo, S, K, T, r, q, True)[1] == NO_BRACKET
    # price requiring sigma=1.0 with sigma_hi=0.5 -> no bracket
    p = bsm_price(S, K, T, r, q, 1.0, True)
    assert implied_vol(p, S, K, T, r, q, True, IVSettings(sigma_hi=0.5))[1] == NO_BRACKET
    assert implied_vol(5.0, S, K, 0.0, r, q, True)[1] == EXPIRED
    assert implied_vol(np.nan, S, K, T, r, q, True)[1] == INVALID_INPUT
    assert implied_vol(5.0, S, -K, T, r, q, True)[1] == INVALID_INPUT
    iv, st = implied_vol_array([p, np.nan], S, K, T, r, q, True)
    assert list(st) == [OK, INVALID_INPUT] and iv[0] == pytest.approx(1.0, abs=1e-10)
