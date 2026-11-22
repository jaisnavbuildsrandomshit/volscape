from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from volscape import diagnostics as dg
from volscape.pipeline import run_pipeline
from volscape.surface import (SVI_INSUFFICIENT, SVI_OK, BaselineSlice, FitSettings, SliceFit,
                              SurfaceModel, fit_svi_slice, gatheral_g, svi_w)
from volscape.synthetic import SyntheticSpec, generate_chain, ssvi_to_raw, ssvi_w

ROOT = Path(__file__).resolve().parents[1]
SPEC = SyntheticSpec()
TZ = "UTC"


@pytest.fixture(scope="module")
def clean_result():
    return run_pipeline((ROOT / "tests/fixtures/clean_chain.csv").read_bytes(), "clean", True)


# ---------------------------------------------------------------- synthetic determinism
def test_generator_is_seed_deterministic():
    a = generate_chain(SyntheticSpec(seed=11, iv_noise_volpts=0.3))
    b = generate_chain(SyntheticSpec(seed=11, iv_noise_volpts=0.3))
    c = generate_chain(SyntheticSpec(seed=12, iv_noise_volpts=0.3))
    pd.testing.assert_frame_equal(a, b)
    assert not a[["bid", "ask"]].equals(c[["bid", "ask"]])


# ---------------------------------------------------------------- SVI recovery
@pytest.mark.parametrize("days", [14, 91, 365, 730])
def test_svi_recovers_ssvi_slice_parameters(days):
    T = days / 365
    sd = np.sqrt(SPEC.v0 ** 2 * T)
    k = np.linspace(-3 * sd, 2.5 * sd, 25)
    iv = np.sqrt(ssvi_w(k, T, SPEC) / T)
    fit = fit_svi_slice(k, iv, T)
    assert fit.status == SVI_OK
    true = ssvi_to_raw(T, SPEC)
    for p in ("a", "b", "rho", "m", "sigma"):
        assert fit.params[p] == pytest.approx(true[p], rel=2e-3, abs=2e-5), p
    assert fit.max_abs_err_volpts < 1e-4


def test_svi_fit_is_deterministic(clean_result):
    again = run_pipeline((ROOT / "tests/fixtures/clean_chain.csv").read_bytes(), "clean", True)
    for f1, f2 in zip(clean_result.fits, again.fits):
        assert f1.params == f2.params and f1.best_start == f2.best_start


def test_svi_refuses_underidentified_slice():
    k = np.linspace(-0.1, 0.1, 5)
    fit = fit_svi_slice(k, 0.2 + 0 * k, 0.5, settings=FitSettings(min_points=7))
    assert fit.status == SVI_INSUFFICIENT and fit.params is None


def test_clean_pipeline_quality(clean_result):
    r = clean_result
    assert all(f.status == SVI_OK for f in r.fits) and len(r.fits) == len(SPEC.expiry_days)
    d = r.diagnostics("svi")
    # residuals come only from tick-rounded bid/ask midpoints of an exact SSVI model
    assert d["aggregate"]["iv_rmse_volpts"] < 0.25
    assert d["aggregate"]["pct_in_band"] > 95
    assert (d["calendar"].n_violations == 0).all()
    assert (d["butterfly"].min_g > 0).all()
    assert (d["call_checks"].convexity_violations == 0).all()


# ---------------------------------------------------------------- total-variance interpolation
def _flat_slices():
    e = [pd.Timestamp("2026-07-01", tz=TZ), pd.Timestamp("2026-12-01", tz=TZ)]
    k = np.linspace(-0.3, 0.3, 7)
    return [BaselineSlice(e[0], 0.25, k, 0.04 * 0.25 + 0.01 * k),
            BaselineSlice(e[1], 1.0, k, 0.03 * 1.0 + 0.02 * k)]


def test_total_variance_linear_in_T_and_masks():
    m = SurfaceModel("baseline", _flat_slices())
    w0, w1 = 0.01 + 0.01 * 0.1, 0.03 + 0.02 * 0.1
    assert m.total_variance(0.1, 0.625) == pytest.approx(0.5 * (w0 + w1), rel=1e-12)
    assert m.total_variance(0.1, 0.25) == pytest.approx(w0, rel=1e-12)
    assert np.isnan(m.total_variance(0.1, 1.5))           # no extrapolation in T
    assert np.isnan(m.total_variance(0.5, 0.5))           # outside observed k
    m2 = SurfaceModel("baseline", _flat_slices(), extrapolate_k=True)
    assert m2.total_variance(0.5, 0.25) == pytest.approx(0.01 + 0.01 * 0.3)   # flat w beyond last point
    # increasing total variance between slices -> interpolated surface is monotone in T
    T = np.linspace(0.25, 1.0, 50)
    assert np.all(np.diff(m.total_variance(0.0, T)) >= 0)


# ---------------------------------------------------------------- known violations
def test_calendar_violation_detected_at_crossing():
    e = [pd.Timestamp("2026-07-01", tz=TZ), pd.Timestamp("2026-12-01", tz=TZ)]
    k = np.linspace(-0.3, 0.3, 13)
    near = BaselineSlice(e[0], 0.25, k, 0.02 + 0.0 * k)
    far = BaselineSlice(e[1], 1.0, k, 0.03 - 0.05 * k)    # far < near for k > 0.2
    cal = dg.calendar_check(SurfaceModel("baseline", [near, far]))
    row = cal.iloc[0]
    assert row.n_violations > 0 and row.worst_dw == pytest.approx(0.03 - 0.015 - 0.02)
    assert row.k_at_worst == pytest.approx(0.3)


def _vogt_model():
    # Axel Vogt's raw-SVI slice (cited by Gatheral & Jacquier 2014): satisfies basic parameter
    # bounds but admits butterfly arbitrage.
    p = dict(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)
    e = pd.Timestamp("2027-06-15", tz=TZ)
    f = SliceFit(e, 0.99, SVI_OK, "", 30, -1.5, 1.5, params=p)
    chain = pd.DataFrame(dict(expiry=[e], spot=[1.0], rate=[0.0], dividend_yield=[0.0], T=[0.99]))
    return f, SurfaceModel("svi", [f]), chain


def test_butterfly_and_density_violation_detected():
    f, model, chain = _vogt_model()
    bf = dg.butterfly_svi([f])
    assert bf.min_g.iloc[0] < 0 and bf.n_negative.iloc[0] > 0
    summ, dens = dg.call_price_checks(model, chain)
    assert summ.convexity_violations.iloc[0] > 0
    assert summ.negative_mass.iloc[0] < 0               # negative density reported, not clipped
    d = dens[f.expiry]
    assert (d.density < 0).any()
    # density negative exactly where g < 0 (same k region)
    g = gatheral_g(d.k.to_numpy(), f.params)
    assert np.all(np.sign(g[np.abs(g) > 1e-3]) == np.sign(d.density.to_numpy()[np.abs(g) > 1e-3]))


def test_observed_call_checks_flag_nonconvex_quotes():
    e = pd.Timestamp("2026-12-01", tz=TZ)
    K = np.array([90, 95, 100, 105, 110.0])
    mid = np.array([12.0, 8.0, 6.5, 2.0, 1.0])          # slope -0.8, -0.3, -0.9 -> convexity break
    ch = pd.DataFrame(dict(expiry=e, option_type="C", strike=K, mid=mid, bid=mid - .05, ask=mid + .05))
    out = dg.observed_call_checks(ch)
    assert out.convex_mid.iloc[0] == 1 and out.mono_mid.iloc[0] == 0


def test_parity_check_band(clean_result):
    par = dg.parity_check(clean_result.chain)
    assert len(par) > 100 and par.within_band.mean() > 0.95
    ch = clean_result.chain.copy()
    i = ch.index[(ch.option_type == "C")][40]
    ch.loc[i, ["bid", "ask", "mid"]] += 1.0                # break parity on one pair
    p2 = dg.parity_check(ch)
    row = p2[(p2.expiry == ch.loc[i, "expiry"]) & (p2.strike == ch.loc[i, "strike"])]
    assert not row.within_band.iloc[0]


def test_display_grid_core_contains_no_extrapolation(clean_result):
    # every core cell is computed with extrapolation disabled and each grid row lies inside the
    # observed k ranges of both slices bracketing its maturity; k-extrapolated bands lie outside
    for model in (clean_result.svi, clean_result.baseline):
        KK, TT, IV = model.grid()["core"]
        assert np.isfinite(IV).all()
        sl = model.slices
        for r in range(KK.shape[0]):
            t, kmin, kmax = TT[r, 0], KK[r].min(), KK[r].max()
            pairs = [(a, b) for a, b in zip(sl[:-1], sl[1:]) if a.T - 1e-12 <= t <= b.T + 1e-12]
            assert any(kmin >= max(model.k_range(a)[0], model.k_range(b)[0]) - 1e-12 and
                       kmax <= min(model.k_range(a)[1], model.k_range(b)[1]) + 1e-12 for a, b in pairs), r
        g = SurfaceModel(model.method, model.slices, extrapolate_k=True).grid()
        assert set(g) == {"core", "left", "right"}
        assert np.all(g["left"][0].max(axis=1) <= g["core"][0].min(axis=1) + 1e-12)
        assert np.all(g["right"][0].min(axis=1) >= g["core"][0].max(axis=1) - 1e-12)
