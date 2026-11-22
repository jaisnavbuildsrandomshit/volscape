"""Calibration and static-arbitrage diagnostics.

All arbitrage checks here are evaluated on FINITE GRIDS. They detect violations at the sampled
points; they do not prove the global absence of arbitrage. Nothing is clipped or smoothed: a
negative density or a calendar crossing is reported with its location and magnitude.

Density convention (Breeden-Litzenberger): with C(K) the DISCOUNTED European call price,
    q(K) = e^{rT} * d2C/dK2,
estimated by the central second difference on a uniform STRIKE grid (step h):
    q_i = e^{rT} (C_{i+1} - 2 C_i + C_{i-1}) / h^2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .pricing import bsm_price, parity_residual
from .surface import SurfaceModel, gatheral_g

FINITE_GRID_NOTE = ("Finite-grid checks detect sampled violations only; they do not prove the "
                    "global absence of arbitrage between or beyond grid points.")


def _slice_for(model: SurfaceModel, expiry):
    for s in model.slices:
        if s.expiry == expiry:
            return s
    return None


def reconstruction(chain: pd.DataFrame, model: SurfaceModel, used_keys: set) -> pd.DataFrame:
    """Per-quote model IV/price vs observed mid, and whether the model price lies in [bid, ask].
    Model value for a quote uses its own expiry slice (no T interpolation)."""
    rows = chain[chain["iv_mid_status"] == "OK"].copy()
    iv_model = np.full(len(rows), np.nan)
    for exp, idx in rows.groupby("expiry").groups.items():
        s = _slice_for(model, exp)
        if s is None:
            continue
        pos = rows.index.get_indexer(idx)
        w = model.slice_w(s, rows.loc[idx, "k"].to_numpy(float), extrapolate=False)
        with np.errstate(invalid="ignore"):
            iv_model[pos] = np.sqrt(np.maximum(w, 0) / rows.loc[idx, "T"].to_numpy(float))
            iv_model[pos] = np.where(np.isfinite(w), iv_model[pos], np.nan)
    rows["iv_model"] = iv_model
    rows["model_price"] = np.where(
        np.isfinite(iv_model),
        bsm_price(rows.spot, rows.strike, rows["T"], rows.rate, rows.dividend_yield,
                  np.nan_to_num(iv_model, nan=0.0), rows.option_type == "C"),
        np.nan)
    rows["iv_err_volpts"] = (rows.iv_model - rows.iv_mid) * 100
    rows["price_err"] = rows.model_price - rows.mid
    rows["in_band"] = np.where(np.isfinite(rows.model_price),
                               (rows.model_price >= rows.bid - 1e-12) & (rows.model_price <= rows.ask + 1e-12), np.nan)
    rows["used_in_fit"] = [(e, k, t) in used_keys for e, k, t in zip(rows.expiry, rows.strike, rows.option_type)]
    return rows


def error_summary(recon: pd.DataFrame, chain: pd.DataFrame, subset: str = "used") -> tuple[pd.DataFrame, dict]:
    r = recon[recon.used_in_fit] if subset == "used" else recon
    out = []
    for exp, g in chain.groupby("expiry"):
        rg = r[(r.expiry == exp) & np.isfinite(r.iv_model)]
        n_all = len(g)
        ie, pe = rg.iv_err_volpts.to_numpy(float), rg.price_err.to_numpy(float)
        out.append(dict(
            expiry=exp, T=float(g["T"].iloc[0]), n_quotes=n_all, n_evaluated=len(rg),
            n_excluded=n_all - len(rg),
            iv_rmse_volpts=float(np.sqrt(np.mean(ie ** 2))) if len(rg) else np.nan,
            iv_max_abs_volpts=float(np.max(np.abs(ie))) if len(rg) else np.nan,
            price_rmse=float(np.sqrt(np.mean(pe ** 2))) if len(rg) else np.nan,
            price_max_abs=float(np.max(np.abs(pe))) if len(rg) else np.nan,
            pct_in_band=float(np.mean(rg.in_band.astype(float))) * 100 if len(rg) else np.nan,
        ))
    df = pd.DataFrame(out)
    rr = r[np.isfinite(r.iv_model)]
    agg = dict(
        n_evaluated=int(len(rr)),
        iv_rmse_volpts=float(np.sqrt(np.mean(rr.iv_err_volpts ** 2))) if len(rr) else np.nan,
        iv_max_abs_volpts=float(np.max(np.abs(rr.iv_err_volpts))) if len(rr) else np.nan,
        price_rmse=float(np.sqrt(np.mean(rr.price_err ** 2))) if len(rr) else np.nan,
        price_max_abs=float(np.max(np.abs(rr.price_err))) if len(rr) else np.nan,
        pct_in_band=float(np.mean(rr.in_band.astype(float))) * 100 if len(rr) else np.nan,
    )
    return df, agg


def calendar_check(model: SurfaceModel, n_k: int = 101, tol: float = 1e-10) -> pd.DataFrame:
    """Total variance must be non-decreasing in T at fixed k. Checked for each consecutive slice
    pair on a shared k grid covering the INTERSECTION of their observed k ranges."""
    out = []
    for s0, s1 in zip(model.slices[:-1], model.slices[1:]):
        lo = max(model.k_range(s0)[0], model.k_range(s1)[0])
        hi = min(model.k_range(s0)[1], model.k_range(s1)[1])
        if hi <= lo:
            out.append(dict(expiry_near=s0.expiry, expiry_far=s1.expiry, n_grid=0, n_violations=np.nan,
                            worst_dw=np.nan, k_at_worst=np.nan, note="no overlapping k range"))
            continue
        kg = np.linspace(lo, hi, n_k)
        dw = model.slice_w(s1, kg, True) - model.slice_w(s0, kg, True)
        viol = dw < -tol
        j = int(np.argmin(dw))
        out.append(dict(expiry_near=s0.expiry, expiry_far=s1.expiry, n_grid=n_k, n_violations=int(viol.sum()),
                        worst_dw=float(dw[j]), k_at_worst=float(kg[j]),
                        note="violation" if viol.any() else "ok on grid"))
    return pd.DataFrame(out)


def calendar_grid(model: SurfaceModel, n_k: int = 81):
    """dw/dT proxy matrix (w_{i+1}-w_i on common k grid; NaN where either slice is unobserved)."""
    if len(model.slices) < 2:
        return None
    lo = min(model.k_range(s)[0] for s in model.slices)
    hi = max(model.k_range(s)[1] for s in model.slices)
    kg = np.linspace(lo, hi, n_k)
    W = np.array([model.slice_w(s, kg, False) for s in model.slices])
    return kg, W[1:] - W[:-1]


def butterfly_svi(fits, n_k: int = 201) -> pd.DataFrame:
    out = []
    for f in fits:
        if not f.usable:
            continue
        kg = np.linspace(f.k_min, f.k_max, n_k)
        g = gatheral_g(kg, f.params)
        j = int(np.nanargmin(g))
        out.append(dict(expiry=f.expiry, T=f.T, min_g=float(g[j]), k_at_min=float(kg[j]),
                        n_negative=int((g < 0).sum()), n_grid=n_k, lee_wing_ok=f.lee_wing_ok))
    return pd.DataFrame(out)


def call_price_checks(model: SurfaceModel, chain: pd.DataFrame, n_strikes: int = 161,
                      tol_rel: float = 1e-10):
    """Model call prices on a uniform strike grid within each slice's observed k range:
    monotonicity (C non-increasing in K, slope >= -e^{-rT}), convexity, and discrete
    risk-neutral density. Returns (summary DataFrame, {expiry: density DataFrame})."""
    summary, dens = [], {}
    for s in model.slices:
        g = chain[chain.expiry == s.expiry]
        if g.empty:
            continue
        S, r, q, T = (float(g[c].iloc[0]) for c in ("spot", "rate", "dividend_yield", "T"))
        F = S * np.exp((r - q) * T)
        lo, hi = model.k_range(s)
        K = np.linspace(F * np.exp(lo), F * np.exp(hi), n_strikes)
        h = K[1] - K[0]
        kk = np.log(K / F)
        iv = np.sqrt(np.maximum(model.slice_w(s, kk, True), 0) / T)
        C = bsm_price(S, K, T, r, q, iv, True)
        tol = tol_rel * S
        dC = np.diff(C) / h
        d2 = (C[2:] - 2 * C[1:-1] + C[:-2]) / h ** 2
        dens_v = np.exp(r * T) * d2
        mono_viol = (np.diff(C) > tol) | (dC < -np.exp(-r * T) - 1e-9)
        conv_viol = d2 * h * h < -tol
        mass = float(np.sum(dens_v) * h)
        neg_mass = float(np.sum(np.minimum(dens_v, 0)) * h)
        summary.append(dict(expiry=s.expiry, T=T, n_strikes=n_strikes, strike_step=h,
                            monotonic_violations=int(mono_viol.sum()), convexity_violations=int(conv_viol.sum()),
                            min_density=float(dens_v.min()), negative_mass=neg_mass,
                            mass_in_range=mass))
        dens[s.expiry] = pd.DataFrame({"strike": K[1:-1], "k": kk[1:-1], "density": dens_v})
    return pd.DataFrame(summary), dens


def observed_call_checks(chain: pd.DataFrame, tol: float = 1e-9) -> pd.DataFrame:
    """Raw quote checks on call mids (sorted by strike, non-uniform spacing):
    monotonic: C(K_{i+1}) <= C(K_i); strict-quote monotonic: bid(K_{i+1}) <= ask(K_i);
    convexity: slope_{i+1} >= slope_i using mids."""
    out = []
    for exp, g in chain[chain.option_type == "C"].groupby("expiry"):
        g = g.sort_values("strike")
        K, mid, bid, ask = (g[c].to_numpy(float) for c in ("strike", "mid", "bid", "ask"))
        if len(K) < 3:
            out.append(dict(expiry=exp, n_calls=len(K), mono_mid=np.nan, mono_quotes=np.nan, convex_mid=np.nan))
            continue
        slope = np.diff(mid) / np.diff(K)
        out.append(dict(expiry=exp, n_calls=len(K),
                        mono_mid=int((np.diff(mid) > tol).sum()),
                        mono_quotes=int((bid[1:] > ask[:-1] + tol).sum()),
                        convex_mid=int((np.diff(slope) < -tol).sum())))
    return pd.DataFrame(out)


def parity_check(chain: pd.DataFrame) -> pd.DataFrame:
    """Put-call parity for each (expiry, strike) with both a call and a put accepted.
    Consistent if  C_bid - P_ask <= S e^{-qT} - K e^{-rT} <= C_ask - P_bid."""
    c = chain[chain.option_type == "C"].set_index(["expiry", "strike"])
    p = chain[chain.option_type == "P"].set_index(["expiry", "strike"])
    j = c.join(p[["bid", "ask", "mid"]], rsuffix="_p", how="inner")
    if j.empty:
        return pd.DataFrame()
    j = j.reset_index()
    theo = j.spot * np.exp(-j.dividend_yield * j["T"]) - j.strike * np.exp(-j.rate * j["T"])
    j["parity_residual_mid"] = parity_residual(j.mid, j.mid_p, j.spot, j.strike, j["T"], j.rate, j.dividend_yield)
    j["lower"] = j.bid - j.ask_p
    j["upper"] = j.ask - j.bid_p
    j["theoretical"] = theo
    j["within_band"] = (j.lower <= theo + 1e-12) & (theo <= j.upper + 1e-12)
    return j[["expiry", "strike", "k", "mid", "mid_p", "theoretical", "parity_residual_mid",
              "lower", "upper", "within_band"]].rename(columns={"mid": "call_mid", "mid_p": "put_mid"})


def coverage(chain: pd.DataFrame, obs: pd.DataFrame, fits) -> pd.DataFrame:
    fit_by = {f.expiry: f for f in fits}
    out = []
    for exp, g in chain.groupby("expiry"):
        o = obs[obs.expiry == exp]
        f = fit_by.get(exp)
        sd = np.sqrt(np.interp(0.0, o.k, o.w_obs)) if len(o) and o.k.min() <= 0 <= o.k.max() else np.nan
        out.append(dict(
            expiry=exp, days=float(g["T"].iloc[0] * 365), n_accepted=len(g), n_iv_ok=int((g.iv_mid_status == "OK").sum()),
            n_low_vega=int(g.low_vega.sum()), n_fit=len(o),
            k_min=float(o.k.min()) if len(o) else np.nan, k_max=float(o.k.max()) if len(o) else np.nan,
            sd_min=float(o.k.min() / sd) if np.isfinite(sd) else np.nan,
            sd_max=float(o.k.max() / sd) if np.isfinite(sd) else np.nan,
            atm_covered=bool(len(o) and o.k.min() <= 0 <= o.k.max()),
            svi_status=f.status if f else "NO_DATA"))
    return pd.DataFrame(out)


def headline_metrics(model: SurfaceModel, agg: dict, n_valid: int, n_total: int) -> dict:
    """30D ATM IV, 30D skew sigma(k=-0.1)-sigma(k=+0.1), term slope ATM(T_max_1y)-ATM(30D).
    Values are NaN when the requested point is outside the surface (no extrapolation)."""
    t30 = 30 / 365
    atm = float(model.iv(0.0, t30, extrapolate=False)) if model.available else np.nan
    skew = (float(model.iv(-0.1, t30, False)) - float(model.iv(0.1, t30, False))) * 100 if model.available else np.nan
    t_long = min(1.0, float(model.T[-1])) if model.available else np.nan
    slope = (float(model.iv(0.0, t_long, False)) - atm) * 100 if model.available else np.nan
    return dict(atm_iv_30d=atm, skew_30d_volpts=skew, term_slope_volpts=slope, term_slope_T=t_long,
                calib_rmse_volpts=agg.get("iv_rmse_volpts", np.nan), n_valid=n_valid, n_total=n_total)
