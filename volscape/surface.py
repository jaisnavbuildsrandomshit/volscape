"""Implied-volatility surface construction in (k, T) coordinates.

k = ln(K/F), F = S exp((r-q)T), total variance w = IV^2 T.

Two methods:
1. Baseline: per-expiry linear interpolation of total variance in k over the observed range
   (no extrapolation unless explicitly requested: flat total variance beyond the last
   observation), and linear interpolation of total variance in T between expiries at fixed k.
2. Raw SVI per expiry: w(k) = a + b[rho(k-m) + sqrt((k-m)^2 + sigma^2)], fitted by bounded
   least squares (scipy trf) from a deterministic multistart grid.

Neither method extrapolates in T outside [T_min, T_max] of usable slices. Interpolation in T is
linear in total variance, which preserves calendar monotonicity when the slices satisfy it.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

SVI_OK = "OK"
SVI_INSUFFICIENT = "INSUFFICIENT_STRIKES"
SVI_FAILED = "OPTIMIZER_FAILED"
SVI_DEGENERATE = "DEGENERATE"
SVI_POOR = "POOR_FIT"


@dataclass(frozen=True)
class FitSettings:
    quote_selection: str = "otm"       # "otm" (puts K<F, calls K>=F), "calls", "puts"
    min_points: int = 7                # > 5 parameters, required distinct strikes per slice
    min_k_span_sd: float = 0.75        # required k span in ATM standard deviations
    exclude_low_vega: bool = True
    iv_weight_floor: float = 0.0025    # vol decimal floor on IV uncertainty in weights
    weighting: str = "spread"          # "spread" (1/IV half-width) or "uniform"
    sigma_min: float = 1e-3
    sigma_max: float = 2.0
    rho_max: float = 0.995
    max_rmse_volpts: float = 2.5       # slice rejected as POOR_FIT above this weighted-agnostic RMSE
    n_multistart: int = 12
    max_nfev: int = 4000

    def to_dict(self):
        return asdict(self)


# ----------------------------------------------------------------------------- SVI math
def svi_w(k, a, b, rho, m, sigma):
    x = np.asarray(k, float) - m
    return a + b * (rho * x + np.sqrt(x * x + sigma * sigma))


def svi_derivs(k, a, b, rho, m, sigma):
    x = np.asarray(k, float) - m
    s = np.sqrt(x * x + sigma * sigma)
    w = a + b * (rho * x + s)
    w1 = b * (rho + x / s)
    w2 = b * sigma * sigma / s ** 3
    return w, w1, w2


def gatheral_g(k, params):
    """Butterfly density factor g(k); risk-neutral density ∝ g(k) * n(d2)/sqrt(w). g<0 => sampled
    butterfly arbitrage (Gatheral & Jacquier 2014)."""
    w, w1, w2 = svi_derivs(k, **params)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (1 - k * w1 / (2 * w)) ** 2 - (w1 ** 2 / 4) * (1 / w + 0.25) + w2 / 2


@dataclass
class SliceFit:
    expiry: pd.Timestamp
    T: float
    status: str
    message: str
    n_points: int
    k_min: float
    k_max: float
    params: dict | None = None
    rmse_volpts: float = np.nan
    max_abs_err_volpts: float = np.nan
    cost: float = np.nan
    nfev: int = 0
    best_start: int = -1
    n_starts: int = 0
    lee_wing_ok: bool | None = None
    min_w_ok: bool | None = None
    at_bounds: list = field(default_factory=list)

    @property
    def usable(self):
        return self.status == SVI_OK

    def to_row(self):
        d = dict(expiry=self.expiry.isoformat(), T=self.T, status=self.status, message=self.message,
                 n_points=self.n_points, k_min=self.k_min, k_max=self.k_max,
                 rmse_volpts=self.rmse_volpts, max_abs_err_volpts=self.max_abs_err_volpts,
                 cost=self.cost, nfev=self.nfev, best_start=self.best_start, n_starts=self.n_starts,
                 lee_wing_ok=self.lee_wing_ok, min_w_ok=self.min_w_ok, at_bounds=",".join(self.at_bounds))
        for p in ("a", "b", "rho", "m", "sigma"):
            d[p] = np.nan if self.params is None else self.params[p]
        return d


# ----------------------------------------------------------------------------- observations
def select_observations(chain: pd.DataFrame, settings: FitSettings) -> pd.DataFrame:
    """Rows used for fitting: mid IV solved OK, chosen side, optionally excluding low vega."""
    c = chain[chain["iv_mid_status"] == "OK"].copy()
    if settings.quote_selection == "otm":
        c = c[((c.option_type == "P") & (c.k < 0)) | ((c.option_type == "C") & (c.k >= 0))]
    elif settings.quote_selection == "calls":
        c = c[c.option_type == "C"]
    elif settings.quote_selection == "puts":
        c = c[c.option_type == "P"]
    if settings.exclude_low_vega:
        c = c[~c.low_vega]
    c = c.copy()
    c["w_obs"] = c.iv_mid ** 2 * c["T"]
    return c.sort_values(["expiry", "k"]).reset_index(drop=True)


def _weights(sl: pd.DataFrame, settings: FitSettings):
    if settings.weighting == "uniform":
        return np.ones(len(sl))
    u = np.maximum(sl["iv_halfwidth"].fillna(settings.iv_weight_floor).to_numpy(float), settings.iv_weight_floor)
    w = 1.0 / u
    return w / w.mean()


def fit_svi_slice(k, iv, T, weights=None, settings: FitSettings = FitSettings(), expiry=None) -> SliceFit:
    k = np.asarray(k, float)
    iv = np.asarray(iv, float)
    order = np.argsort(k)
    k, iv = k[order], iv[order]
    wts = np.ones_like(k) if weights is None else np.asarray(weights, float)[order]
    n = len(np.unique(np.round(k, 10)))
    expiry = expiry if expiry is not None else pd.Timestamp("1970-01-01", tz="UTC")
    base = SliceFit(expiry, T, SVI_INSUFFICIENT, "", n, float(k.min()) if len(k) else np.nan,
                    float(k.max()) if len(k) else np.nan)
    if n < settings.min_points:
        base.message = f"{n} distinct strikes < required {settings.min_points}"
        return base
    w_obs = iv ** 2 * T
    atm_sd = float(np.sqrt(np.interp(0.0, k, w_obs))) if k.min() <= 0 <= k.max() else float(np.sqrt(np.median(w_obs)))
    span = (k.max() - k.min()) / max(atm_sd, 1e-6)
    if span < settings.min_k_span_sd:
        base.message = f"k span {span:.2f} ATM std-devs < required {settings.min_k_span_sd}"
        return base

    # Box-bounded parametrisation: theta = (wmin, b, rho, m, sigma); a = wmin - b*sigma*sqrt(1-rho^2)
    # where wmin = min_k w(k) >= 0 guarantees nonnegative total variance everywhere.
    kspan = k.max() - k.min()
    wmax = float(w_obs.max())
    lo = np.array([0.0, 0.0, -settings.rho_max, k.min() - kspan, settings.sigma_min])
    # b <= 5 is deliberately looser than Lee's wing bound b(1+|rho|) <= 2 so violations are
    # reported by the diagnostic rather than hidden by the optimizer box.
    hi = np.array([2 * wmax, 5.0, settings.rho_max, k.max() + kspan, settings.sigma_max])

    def unpack(th):
        wmin, b, rho, m, sig = th
        return dict(a=wmin - b * sig * np.sqrt(1 - rho * rho), b=b, rho=rho, m=m, sigma=sig)

    def resid(th):
        w = np.maximum(svi_w(k, **unpack(th)), 1e-12)
        return wts * (np.sqrt(w / T) - iv)

    def jac(th):
        wmin, b, rho, m, sig = th
        x = k - m
        s = np.sqrt(x * x + sig * sig)
        c = np.sqrt(1 - rho * rho)
        w = np.maximum(wmin - b * sig * c + b * (rho * x + s), 1e-12)
        dw = np.column_stack([np.ones_like(k), -sig * c + rho * x + s, b * (sig * rho / c + x),
                              b * (-rho - x / s), b * (-c + sig / s)])
        return (wts / (2.0 * np.sqrt(w * T)))[:, None] * dw

    # deterministic multistart grid
    starts = []
    wmin0 = max(float(w_obs.min()) * 0.9, 1e-8)
    slope = (w_obs[-1] - w_obs[0]) / max(kspan, 1e-6)
    for rho0 in (-0.7, -0.3, 0.0, 0.4):
        for sig_mult in (0.3, 1.0, 3.0):
            sig0 = float(np.clip(sig_mult * atm_sd, settings.sigma_min * 2, settings.sigma_max / 2))
            b0 = float(np.clip(max(abs(slope), wmax / max(kspan, 1e-3)) / 1.5, 1e-6, hi[1] * 0.5))
            m0 = float(k[np.argmin(w_obs)]) * 0.5
            starts.append(np.clip([wmin0, b0, rho0, m0, sig0], lo + 1e-9, hi - 1e-9))
    starts = starts[: settings.n_multistart]

    best, best_i, total_nfev = None, -1, 0
    for i, s in enumerate(starts):
        try:
            r = least_squares(resid, s, jac=jac, bounds=(lo, hi), method="trf", x_scale="jac",
                              max_nfev=settings.max_nfev, xtol=1e-10, ftol=1e-11, gtol=1e-10)
        except (ValueError, FloatingPointError):
            continue
        total_nfev += r.nfev
        if r.status > 0 and (best is None or r.cost < best.cost - 1e-15):
            best, best_i = r, i
    base.n_starts = len(starts)
    base.nfev = total_nfev
    if best is None:
        base.status, base.message = SVI_FAILED, "no multistart run converged"
        return base

    params = {k_: float(v) for k_, v in unpack(best.x).items()}
    w_fit = svi_w(k, **params)
    err = np.sqrt(np.maximum(w_fit, 0) / T) - iv
    base.params = params
    base.cost = float(best.cost)
    base.best_start = best_i
    base.rmse_volpts = float(np.sqrt(np.mean(err ** 2)) * 100)
    base.max_abs_err_volpts = float(np.max(np.abs(err)) * 100)
    base.lee_wing_ok = bool(params["b"] * (1 + abs(params["rho"])) <= 2.0 + 1e-12)
    base.min_w_ok = bool(params["a"] + params["b"] * params["sigma"] * np.sqrt(1 - params["rho"] ** 2) >= -1e-14)
    names = ["wmin", "b", "rho", "m", "sigma"]
    rel = 1e-6 * (hi - lo)
    base.at_bounds = [names[j] for j in range(5) if j > 0 and (best.x[j] - lo[j] < rel[j] or hi[j] - best.x[j] < rel[j])]
    degenerate = [p for p in base.at_bounds if p in ("rho", "m", "sigma")]
    if params["b"] < 1e-8:
        degenerate.append("b≈0 (flat slice)")
    if degenerate:
        base.status = SVI_DEGENERATE
        base.message = "parameters at bound: " + ", ".join(degenerate) + "; slice not identified"
    elif base.rmse_volpts > settings.max_rmse_volpts:
        base.status = SVI_POOR
        base.message = f"IV RMSE {base.rmse_volpts:.2f} vol pts > {settings.max_rmse_volpts}"
    else:
        base.status = SVI_OK
        base.message = f"converged ({best.message.split('.')[0]})"
    return base


def fit_svi_surface(obs: pd.DataFrame, settings: FitSettings = FitSettings()) -> list[SliceFit]:
    fits = []
    for exp, sl in obs.groupby("expiry", sort=True):
        fits.append(fit_svi_slice(sl.k, sl.iv_mid, float(sl["T"].iloc[0]), _weights(sl, settings), settings, exp))
    return fits


# ----------------------------------------------------------------------------- baseline
@dataclass
class BaselineSlice:
    expiry: pd.Timestamp
    T: float
    k: np.ndarray
    w: np.ndarray

    def total_variance(self, kq, extrapolate=False):
        kq = np.asarray(kq, float)
        out = np.interp(kq, self.k, self.w)
        if not extrapolate:
            out = np.where((kq < self.k[0]) | (kq > self.k[-1]), np.nan, out)
        return out


def build_baseline(obs: pd.DataFrame, min_points: int = 3) -> list[BaselineSlice]:
    sl = []
    for exp, g in obs.groupby("expiry", sort=True):
        g = g.groupby("k", as_index=False)["w_obs"].mean().sort_values("k")
        if len(g) >= min_points:
            sl.append(BaselineSlice(exp, float(obs.loc[obs.expiry == exp, "T"].iloc[0]),
                                    g.k.to_numpy(float), g.w_obs.to_numpy(float)))
    return sl


# ----------------------------------------------------------------------------- surface model
class SurfaceModel:
    """Uniform interface over slices: total_variance(k, T) with extrapolation masks."""

    def __init__(self, method: str, slices: list, extrapolate_k: bool = False):
        self.method = method
        self.extrapolate_k = extrapolate_k
        if method == "svi":
            self.slices = [s for s in slices if s.usable]
        else:
            self.slices = list(slices)
        self.slices.sort(key=lambda s: s.T)
        self.T = np.array([s.T for s in self.slices])

    @property
    def available(self):
        return len(self.slices) >= 2

    def k_range(self, s):
        return (s.k_min, s.k_max) if self.method == "svi" else (s.k[0], s.k[-1])

    def slice_w(self, s, k, extrapolate=None):
        extrapolate = self.extrapolate_k if extrapolate is None else extrapolate
        k = np.asarray(k, float)
        if self.method == "svi":
            w = svi_w(k, **s.params)
            if not extrapolate:
                lo, hi = self.k_range(s)
                w = np.where((k < lo) | (k > hi), np.nan, w)
            return w
        return s.total_variance(k, extrapolate)

    def total_variance(self, k, T, extrapolate=None):
        """Linear-in-T interpolation of total variance at fixed k; NaN outside [T_min, T_max]."""
        k = np.asarray(k, float)
        T = np.asarray(T, float)
        k, T = np.broadcast_arrays(k, T)
        out = np.full(k.shape, np.nan)
        if not self.slices:
            return out
        if len(self.slices) == 1:
            m = np.isclose(T, self.T[0])
            out[m] = self.slice_w(self.slices[0], k[m], extrapolate)
            return out
        idx = np.clip(np.searchsorted(self.T, T, side="right") - 1, 0, len(self.T) - 2)
        inside = (T >= self.T[0] - 1e-12) & (T <= self.T[-1] + 1e-12)
        for j in np.unique(idx[inside]):
            m = inside & (idx == j)
            s0, s1 = self.slices[j], self.slices[j + 1]
            lam = (T[m] - s0.T) / (s1.T - s0.T)
            out[m] = (1 - lam) * self.slice_w(s0, k[m], extrapolate) + lam * self.slice_w(s1, k[m], extrapolate)
        return out

    def iv(self, k, T, extrapolate=None):
        w = self.total_variance(k, T, extrapolate)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where((w >= 0) & (np.asarray(T) > 0), np.sqrt(np.maximum(w, 0) / np.asarray(T)), np.nan)

    def grid(self, n_k=61, n_t=40, pad_frac=0.25):
        """Maturity-dependent (k, T) grid for display/export.

        Each row in maturity interval [T_j, T_j+1] spans the INTERSECTION of those two slices'
        observed k ranges, so the core sheet contains interpolated values only (no k-extrapolation). Returns dict with KK, TT, IV for the core and,
        when k-extrapolation is enabled, left/right extrapolated bands of width pad_frac*(k_hi-k_lo).
        """
        # One rectangular patch per maturity interval [T_j, T_j+1]; node rows are duplicated so the
        # coverage boundary steps cleanly at each expiry instead of being joined by spurious facets.
        lo_s = np.array([self.k_range(s)[0] for s in self.slices])
        hi_s = np.array([self.k_range(s)[1] for s in self.slices])
        span = self.T[-1] - self.T[0]
        rows_t, rows_j = [], []
        for j in range(len(self.T) - 1):
            m = max(2, int(round(n_t * (self.T[j + 1] - self.T[j]) / span)) + 1)
            rows_t.extend(np.linspace(self.T[j], self.T[j + 1], m)); rows_j.extend([j] * m)
        tv, jv = np.array(rows_t), np.array(rows_j)
        lo = np.maximum(lo_s[jv], lo_s[jv + 1])
        hi = np.minimum(hi_s[jv], hi_s[jv + 1])
        u = np.linspace(0.0, 1.0, n_k)
        KK = lo[:, None] + u[None, :] * (hi - lo)[:, None]
        KK = np.clip(KK, lo[:, None], hi[:, None])   # guard float overshoot at row ends
        TT = np.repeat(tv[:, None], n_k, axis=1)

        def patch_iv(K, extrapolate):
            iv = np.empty_like(K)
            for r in range(len(tv)):
                j, t = jv[r], tv[r]
                s0, s1 = self.slices[j], self.slices[j + 1]
                lam = (t - s0.T) / (s1.T - s0.T)
                if lam <= 1e-12:
                    w = self.slice_w(s0, K[r], extrapolate)
                elif lam >= 1 - 1e-12:
                    w = self.slice_w(s1, K[r], extrapolate)
                else:
                    w = (1 - lam) * self.slice_w(s0, K[r], extrapolate) + lam * self.slice_w(s1, K[r], extrapolate)
                with np.errstate(invalid="ignore"):
                    iv[r] = np.where(w >= 0, np.sqrt(np.maximum(w, 0) / t), np.nan)
            return iv

        out = {"core": (KK, TT, patch_iv(KK, False))}
        if self.extrapolate_k:
            w = (hi - lo)[:, None] * pad_frac
            nb = max(8, n_k // 4)
            ub = np.linspace(0.0, 1.0, nb)[None, :]
            KL = (lo[:, None] - w) + ub * w
            KR = hi[:, None] + ub * w
            TB = np.repeat(tv[:, None], nb, axis=1)
            out["left"] = (KL, TB, patch_iv(KL, True))
            out["right"] = (KR, TB, patch_iv(KR, True))
        return out
