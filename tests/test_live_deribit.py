"""Deribit adapter: offline. Rows are built by pricing known vols with BSM in Deribit's convention
(coin price = USD price / F), so the round trip through the full pipeline must recover them."""
import math
from datetime import datetime, timedelta, timezone

import numpy as np

from volscape.iv import implied_vol
from volscape.live.deribit import parse_expiry, to_chain
from volscape.pipeline import run_pipeline
from volscape.pricing import bsm_price
from volscape.surface import svi_w

NOW = datetime(2026, 11, 23, 5, 0, tzinfo=timezone.utc)
S = 60000.0


def _row(exp_code, K, typ, sigma, F, spread=0.0005, bid=True, ask=True):
    exp = parse_expiry(exp_code)
    T = (exp - NOW).total_seconds() / (365 * 86400)
    q = -math.log(F / S) / T
    usd = float(bsm_price(S, K, T, 0.0, q, sigma, typ == "C"))
    coin = usd / F
    return {"instrument_name": f"BTC-{exp_code}-{int(K)}-{typ}", "bid_price": coin - spread if bid else None,
            "ask_price": coin + spread if ask else None, "underlying_price": F, "estimated_delivery_price": S,
            "volume": 1.0, "open_interest": 10.0, "_true": sigma, "_T": T, "_q": q}


def test_parse_expiry_is_0800_utc():
    assert parse_expiry("27NOV26") == datetime(2026, 11, 27, 8, 0, tzinfo=timezone.utc)
    assert parse_expiry("4DEC26").day == 4


def test_conversion_recovers_vol_and_forward():
    r = _row("25DEC26", 64000, "C", 0.55, 60500.0, spread=0.0)
    csv, notes = to_chain([r], NOW, "BTC")
    import pandas as pd, io
    df = pd.read_csv(io.BytesIO(csv))
    row = df.iloc[0]
    assert row.rate == 0.0 and math.isclose(row.spot * math.exp(-row.dividend_yield * r["_T"]), 60500.0, rel_tol=1e-12)
    iv, st = implied_vol(row.ask, row.spot, row.strike, r["_T"], row.rate, row.dividend_yield, True)
    assert st == "OK" and abs(iv - 0.55) < 1e-6
    assert notes["n_written"] == 1 and notes["index_price"] == S


def test_missing_ask_skipped_missing_bid_zero_and_counted():
    rows = [_row("25DEC26", 60000, "P", 0.5, 60500.0, ask=False), _row("25DEC26", 70000, "C", 0.5, 60500.0, bid=False)]
    csv, notes = to_chain(rows, NOW, "BTC")
    assert notes["n_no_ask"] == 1 and notes["n_written"] == 1
    import pandas as pd, io
    assert pd.read_csv(io.BytesIO(csv)).bid.iloc[0] == 0.0


def test_full_pipeline_on_converted_snapshot_recovers_smiles():
    rows = []
    for code, F in (("4DEC26", 60150.0), ("25DEC26", 60500.0), ("29JAN27", 61100.0)):
        T = (parse_expiry(code) - NOW).total_seconds() / (365 * 86400)
        for K in np.linspace(F * math.exp(-0.35 * math.sqrt(T) * 2), F * math.exp(0.35 * math.sqrt(T) * 2), 15):
            K = round(K / 500) * 500
            k = math.log(K / F)
            sigma = math.sqrt(svi_w(k, 0.25 * T, 0.5 * T ** 0.5 / 4, -0.3, 0.02, 0.15) / T)   # SVI-shaped smile
            for typ in ("C", "P"):
                rows.append(_row(code, K, typ, sigma, F, spread=0.0))
    seen, uniq = set(), []
    for r in rows:
        if r["instrument_name"] not in seen:
            seen.add(r["instrument_name"]); uniq.append(r)
    csv, _ = to_chain(uniq, NOW, "BTC")
    res = run_pipeline(csv, "LIVE SNAPSHOT · test", False)
    assert res.ok and len(res.validation.rejected) == 0
    ok = res.chain[res.chain.iv_mid_status == "OK"]
    k = ok.k.to_numpy()
    T = ok["T"].to_numpy()
    true = np.sqrt(svi_w(k, 0.25 * T, 0.5 * T ** 0.5 / 4, -0.3, 0.02, 0.15) / T)
    assert np.max(np.abs(ok.iv_mid.to_numpy() - true)) < 1e-5
    assert sum(f.usable for f in res.fits) == 3
