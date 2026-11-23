"""Upstox / Alpaca adapters: offline. Payloads follow each provider's documented response shape and
are priced with BSM from known (r, q, vol), so the conversion must recover them exactly."""
import math

import numpy as np
import pandas as pd
import pytest

from volscape.live import brokers
from volscape.live.chainbuild import build_chain_csv, parity_forward
from volscape.pipeline import run_pipeline
from volscape.data import ValidationSettings
from volscape.pipeline import PipelineSettings
from volscape.pricing import bsm_price
from volscape.surface import svi_w

NOW = pd.Timestamp("2026-11-23 05:00", tz="UTC")


def test_parity_forward_recovers_discount_and_forward():
    S, T, r, q = 24000.0, 0.25, 0.065, 0.012
    K = np.arange(22000, 26001, 250.0)
    c = bsm_price(S, K, T, r, q, 0.15, True)
    p = bsm_price(S, K, T, r, q, 0.15, False)
    fit = parity_forward(K, c, p, S)
    assert fit.df == pytest.approx(math.exp(-r * T), rel=1e-10)
    assert fit.forward == pytest.approx(S * math.exp((r - q) * T), rel=1e-10)


def test_parity_forward_refuses_when_unidentified():
    assert parity_forward([100.0, 110.0], [5.0, 2.0], [3.0, 6.0], 105) is None          # < 3 pairs
    assert parity_forward([90, 100, 110.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], 100) is None  # DF = 0: implausible


def _svi_iv(k, T):
    return math.sqrt(svi_w(k, 0.02 * T, 0.12 * math.sqrt(T), -0.4, 0.01, 0.12) / T)


def _upstox_payload(S, r, q):
    rows = []
    for day in ("2026-12-29", "2027-01-26", "2027-02-23"):
        exp = pd.Timestamp(day + " 15:30", tz="Asia/Kolkata")
        T = (exp - NOW.floor("s")).total_seconds() / (365 * 86400)
        F = S * math.exp((r - q) * T)
        for K in np.arange(round(F * 0.85, -2), F * 1.15, 100.0):
            iv = _svi_iv(math.log(K / F), T)
            c = float(bsm_price(S, K, T, r, q, iv, True))
            p = float(bsm_price(S, K, T, r, q, iv, False))
            md = lambda v: {"ltp": v, "volume": 10, "oi": 100, "close_price": v, "bid_price": v, "bid_qty": 50,
                            "ask_price": v, "ask_qty": 50, "prev_oi": 90}
            rows.append({"expiry": day, "pcr": 1.0, "strike_price": float(K), "underlying_key": "NSE_INDEX|Nifty 50",
                         "underlying_spot_price": S,
                         "call_options": {"instrument_key": "NSE_FO|1", "market_data": md(c), "option_greeks": {}},
                         "put_options": {"instrument_key": "NSE_FO|2", "market_data": md(p), "option_greeks": {}}})
    return rows


def test_upstox_documented_shape_round_trips_through_full_pipeline():
    S, r, q = 24000.0, 0.065, 0.012
    quotes = brokers.upstox_rows(_upstox_payload(S, r, q))
    csv, notes = build_chain_csv(quotes, S, NOW, "NIFTY 50 (NSE)", "INR", "european")
    assert len(notes["forwards"]) == 3 and not notes["skipped_expiries"]
    for f in notes["forwards"].values():
        assert f["rate"] == pytest.approx(r, abs=1e-8) and f["div_yield"] == pytest.approx(q, abs=1e-8)
    res = run_pipeline(csv, "LIVE SNAPSHOT · test NSE", False)
    assert res.ok and len(res.validation.rejected) == 0
    ok = res.chain[res.chain.iv_mid_status == "OK"]
    true = np.array([_svi_iv(k, T) for k, T in zip(ok.k, ok["T"])])
    assert np.max(np.abs(ok.iv_mid.to_numpy() - true)) < 1e-6
    assert sum(f.usable for f in res.fits) == 3


def test_alpaca_occ_parse_and_american_requires_explicit_opt_in():
    root, exp, typ, K = brokers.parse_occ("AAPL240426C00162500")
    assert (root, typ, K) == ("AAPL", "C", 162.5)
    assert exp == pd.Timestamp("2024-04-26 16:00", tz="America/New_York")
    S, r, q = 200.0, 0.045, 0.005
    snaps = {}
    for ymd, day in (("261218", "2026-12-18"), ("270115", "2027-01-15"), ("270219", "2027-02-19")):
        exp = pd.Timestamp(day + " 16:00", tz="America/New_York")
        T = (exp - NOW.floor("s")).total_seconds() / (365 * 86400)
        F = S * math.exp((r - q) * T)
        for K in np.arange(170.0, 231.0, 5.0):
            iv = _svi_iv(math.log(K / F), T)
            for t in ("C", "P"):
                px = float(bsm_price(S, K, T, r, q, iv, t == "C"))
                snaps[f"AAPL{ymd}{t}{int(round(K * 1000)):08d}"] = {
                    "latestQuote": {"ap": px, "as": 1, "ax": "A", "bp": px, "bs": 1, "bx": "A", "c": "A", "t": "x"}}
    csv, notes = build_chain_csv(brokers.alpaca_rows(snaps), S, NOW, "AAPL (US)", "USD", "american")
    assert len(notes["forwards"]) == 3
    default = run_pipeline(csv, "LIVE SNAPSHOT · test US", False)
    assert not default.ok and default.validation.rejected.reasons.str.contains("American").all()
    opted = run_pipeline(csv, "LIVE SNAPSHOT · test US", False,
                         PipelineSettings(validation=ValidationSettings(allow_american_as_european=True)))
    assert opted.ok and any("American" in w for w in opted.validation.warnings)


def test_missing_credentials_raise_clear_error(monkeypatch):
    monkeypatch.delenv("UPSTOX_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    with pytest.raises(brokers.MissingCredentials):
        brokers.fetch_upstox("NIFTY 50")
    with pytest.raises(brokers.MissingCredentials):
        brokers.fetch_alpaca("AAPL")
