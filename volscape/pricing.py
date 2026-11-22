"""European Black-Scholes-Merton pricing, bounds and Greeks (vectorised).

Inputs: spot S, strike K, time T (years, ACT/365), continuous rate r, continuous dividend
yield q, volatility sigma (decimal). ``is_call`` is a boolean (array) flag.

Limits handled deliberately (no NaN leakage from 0/0):
* T <= 0  -> value at expiry = intrinsic max(S-K, 0) / max(K-S, 0) (no discounting).
* sigma = 0 (T > 0) -> discounted forward intrinsic DF*max(F-K, 0) (calls),
  DF*max(K-F, 0) (puts): the deterministic-forward limit of BSM.

Greek units (all per ONE unit of underlying, before any contract multiplier):
* delta : dV/dS                (currency per currency of spot)
* gamma : d2V/dS2              (per currency of spot, per currency of spot)
* vega  : dV/dsigma            (currency per 1.00 of vol; /100 = per vol point)
* theta : dV/dt = -dV/dT       (currency per year of calendar time; /365 = per day)
* rho   : dV/dr, q held fixed  (currency per 1.00 of rate; /100 = per 1 % rate)
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def _npdf(x):
    return np.exp(-0.5 * x * x) / _SQRT_2PI


def forward(S, T, r, q):
    return np.asarray(S, float) * np.exp((np.asarray(r, float) - np.asarray(q, float)) * np.asarray(T, float))


def _d1d2(F, K, T, sigma):
    vs = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * vs * vs) / vs
    return d1, d1 - vs


def bsm_price(S, K, T, r, q, sigma, is_call):
    """Vectorised BSM price with explicit expiry and zero-vol limits."""
    S, K, T, r, q, sigma = np.broadcast_arrays(*(np.asarray(x, float) for x in (S, K, T, r, q, sigma)))
    is_call = np.broadcast_to(np.asarray(is_call, bool), S.shape)
    out = np.full(S.shape, np.nan)

    expired = T <= 0
    out[expired] = np.where(is_call[expired], np.maximum(S[expired] - K[expired], 0.0),
                            np.maximum(K[expired] - S[expired], 0.0))
    live = ~expired
    Tl, Kl, sig, c = T[live], K[live], sigma[live], is_call[live]
    F = S[live] * np.exp((r[live] - q[live]) * Tl)
    df = np.exp(-r[live] * Tl)
    val = np.empty(Tl.shape)
    zero = sig <= 0
    val[zero] = df[zero] * np.where(c[zero], np.maximum(F[zero] - Kl[zero], 0.0),
                                    np.maximum(Kl[zero] - F[zero], 0.0))
    pos = ~zero
    if np.any(pos):
        d1, d2 = _d1d2(F[pos], Kl[pos], Tl[pos], sig[pos])
        call = df[pos] * (F[pos] * ndtr(d1) - Kl[pos] * ndtr(d2))
        put = df[pos] * (Kl[pos] * ndtr(-d2) - F[pos] * ndtr(-d1))
        val[pos] = np.where(c[pos], call, put)
    out[live] = val
    return out if out.ndim else float(out)


def price_bounds(S, K, T, r, q, is_call):
    """Model-free no-arbitrage bounds for European options (continuous r, q).

    call: max(S e^{-qT} - K e^{-rT}, 0) <= C <= S e^{-qT}
    put : max(K e^{-rT} - S e^{-qT}, 0) <= P <= K e^{-rT}
    """
    S, K, T, r, q = (np.asarray(x, float) for x in (S, K, T, r, q))
    is_call = np.asarray(is_call, bool)
    Tc = np.maximum(T, 0.0)
    s_disc = S * np.exp(-q * Tc)
    k_disc = K * np.exp(-r * Tc)
    lower = np.where(is_call, np.maximum(s_disc - k_disc, 0.0), np.maximum(k_disc - s_disc, 0.0))
    upper = np.where(is_call, s_disc, k_disc)
    return lower, upper


def greeks(S, K, T, r, q, sigma, is_call):
    """Analytic BSM Greeks (units in module docstring). Defined for T > 0 and sigma > 0;
    other entries are NaN (degenerate there, never silently zero)."""
    S, K, T, r, q, sigma = np.broadcast_arrays(*(np.asarray(x, float) for x in (S, K, T, r, q, sigma)))
    is_call = np.broadcast_to(np.asarray(is_call, bool), S.shape)
    ok = (T > 0) & (sigma > 0)
    res = {k: np.full(S.shape, np.nan) for k in ("delta", "gamma", "vega", "theta", "rho")}
    if not np.any(ok):
        return res
    S_, K_, T_, r_, q_, s_, c = S[ok], K[ok], T[ok], r[ok], q[ok], sigma[ok], is_call[ok]
    sqrtT = np.sqrt(T_)
    F = S_ * np.exp((r_ - q_) * T_)
    d1, d2 = _d1d2(F, K_, T_, s_)
    dq, dr, pdf = np.exp(-q_ * T_), np.exp(-r_ * T_), _npdf(d1)
    common = -S_ * dq * pdf * s_ / (2.0 * sqrtT)
    vals = {
        "delta": np.where(c, dq * ndtr(d1), -dq * ndtr(-d1)),
        "gamma": dq * pdf / (S_ * s_ * sqrtT),
        "vega": S_ * dq * pdf * sqrtT,
        "theta": np.where(c, common - r_ * K_ * dr * ndtr(d2) + q_ * S_ * dq * ndtr(d1),
                          common + r_ * K_ * dr * ndtr(-d2) - q_ * S_ * dq * ndtr(-d1)),
        "rho": np.where(c, K_ * T_ * dr * ndtr(d2), -K_ * T_ * dr * ndtr(-d2)),
    }
    for name, v in vals.items():
        res[name][ok] = v
    return res


def parity_residual(call, put, S, K, T, r, q):
    """C - P - (S e^{-qT} - K e^{-rT}); zero under European put-call parity."""
    return call - put - (S * np.exp(-q * T) - K * np.exp(-r * T))
