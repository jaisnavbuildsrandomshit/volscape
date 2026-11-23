# VOLSCAPE methodology

## 1. Conventions
- Time: `T = (expiry − valuation) / (365 · 86400 s)`, ACT/365 fixed. Both timestamps must be
  timezone-aware and are converted to UTC. Expired contracts (`T ≤ 0`) are rejected at validation.
- Rates: `r` and `q` are continuously compounded annual decimals. The spot and valuation time must be
  unique within a snapshot. `r` and `q` must be single-valued within each expiry, so a term structure
  across expiries is allowed.
- Forward and coordinates: `F = S·e^{(r−q)T}`, log-moneyness `k = ln(K/F)`, total variance `w = σ²T`.
- Prices are per unit of underlying. Contract multipliers appear only in the portfolio layer.

## 2. Pricing (European BSM)
`d₁,₂ = [ln(F/K) ± ½σ²T] / (σ√T)`, `C = e^{−rT}[F·N(d₁) − K·N(d₂)]`, `P = e^{−rT}[K·N(−d₂) − F·N(−d₁)]`.

- **Limits.** At `T ≤ 0` the value is intrinsic `max(S−K,0)` or `max(K−S,0)`, undiscounted. At `σ = 0`
  with `T > 0` it is `e^{−rT}·max(F−K,0)` or `e^{−rT}·max(K−F,0)`. Both are evaluated without 0/0.
- **Model-free bounds.** `max(Se^{−qT} − Ke^{−rT}, 0) ≤ C ≤ Se^{−qT}` and
  `max(Ke^{−rT} − Se^{−qT}, 0) ≤ P ≤ Ke^{−rT}`.
- **Greeks** (analytic, per unit of underlying):
  - delta `∂V/∂S`;
  - gamma `∂²V/∂S²`;
  - vega `∂V/∂σ`, per 1.00 vol, shown ÷100 as per vol point;
  - theta `−∂V/∂T`, per year, shown ÷365 as per calendar day;
  - rho `∂V/∂r` with `q` fixed, per 1.00, shown ÷100 as per 1%.

  Greeks are NaN, not silently zero, at `T ≤ 0` or `σ ≤ 0`. Tests check them against central finite
  differences with scale-aware tolerances.
- **Put–call parity.** `C − P = Se^{−qT} − Ke^{−rT}`. For every paired C/P at the same expiry and strike,
  the check reports the mid residual and whether `C_bid − P_ask ≤ Se^{−qT} − Ke^{−rT} ≤ C_ask − P_bid`.

## 3. Implied volatility
Brent root-finding (`scipy.optimize.brentq`) on `[σ_lo, σ_hi]`, default `[1e−4, 5]`, with `xtol = 1e−12`,
after the bound checks. The statuses are never collapsed:

| status | meaning |
|---|---|
| `OK` | converged; the price residual is within `1e−9·max(price, 1e−8·S)` |
| `INVALID_INPUT` | nonfinite inputs, or `S` or `K` not positive |
| `EXPIRED` | `T ≤ 0` |
| `INVALID_PRICE` | the target lies outside the model-free bounds (for example a deep-ITM bid below intrinsic) |
| `NO_BRACKET` | inside the bounds but not attainable within `[σ_lo, σ_hi]` |
| `NOT_CONVERGED` | Brent failed, or the residual exceeds the tolerance |

**Uncertainty and stability.** The first-order IV half-width is `½(ask − bid) / vega`. When vega per vol
point falls below a threshold (default 0.005 currency units), the quote is flagged `LOW_VEGA`: one tick
of price there moves IV by many vol points. Such quotes are excluded from fitting by default, and fit
residuals are weighted by `1 / max(IV half-width, 0.25 vol pt)`.

## 4. Surface methods
**Observations.** These are the out-of-the-money quotes (puts for `k < 0`, calls for `k ≥ 0`) whose mid IV
solved `OK`. Calls-only and puts-only are also available.

**Baseline.** Per expiry, total variance is interpolated linearly in `k` over the observed range, with no
extrapolation unless enabled; enabling it holds `w` flat beyond the last point. Across expiries, `w` is
interpolated linearly in `T` at fixed `k`. When adjacent slices are ordered in `w`, this preserves
calendar monotonicity.

**Raw SVI.** Each expiry is fitted with `w(k) = a + b[ρ(k−m) + √((k−m)² + σ²)]`:
- Parametrisation: the optimiser works on box-bounded variables `(w_min, b, ρ, m, σ)` with
  `a = w_min − bσ√(1−ρ²)` and `w_min ≥ 0`, which guarantees `w(k) ≥ 0`.
- Bounds: `0 ≤ b ≤ 5`, `|ρ| ≤ 0.995`, `m` within the observed k range ± its span, and
  `σ ∈ [1e−3, 2]`. The bound on `b` is deliberately looser than Lee's wing condition `b(1+|ρ|) ≤ 2`, so
  that wing violations are reported rather than hidden by the optimiser.
- Objective: weighted IV residuals, solved with `scipy.optimize.least_squares` (method `trf`), an
  analytic Jacobian, and a deterministic multistart grid over ρ and σ. The lowest-cost converged run
  is kept.
- A slice is rejected with an explicit status when:
  - it has fewer than 7 distinct strikes, or its k span is below 0.75 ATM standard deviations
    (`INSUFFICIENT_STRIKES`);
  - no run converges (`OPTIMIZER_FAILED`);
  - ρ, m or σ is pinned at a bound, or `b ≈ 0` (`DEGENERATE`);
  - IV RMSE exceeds 2.5 vol pts (`POOR_FIT`).
- Rejected slices are shown in the parameter table and on the term-structure chart, and are excluded
  from the surface. At least two usable maturities are required for a surface.
- Passing the parameter bounds does **not** make a fit arbitrage-free. The calendar and butterfly
  checks are reported separately.

**Display grid.** Each row in maturity interval `[T_j, T_{j+1}]` spans the **intersection** of those two
slices' observed k ranges and is evaluated with extrapolation disabled. Node rows are duplicated, so
coverage steps cleanly at each expiry. Each slice's own observed range is drawn as a separate line.
Extrapolated bands, when enabled, are separate translucent sheets and are flagged `extrapolated=true`
in exports. Maturity is never extrapolated.

## 5. Diagnostics
- **Reconstruction.** Each quote's model IV comes from its own expiry slice. The model price is
  `BSM(model IV)`. Errors are reported in vol points and currency, per expiry and in aggregate, both for
  the fit set and for all IV-solvable quotes, together with the share of model prices inside `[bid, ask]`.
- **Calendar.** `w(k, T_{j+1}) ≥ w(k, T_j)` is checked on a 101-point grid over the intersection of each
  consecutive pair's observed k ranges. The count, the worst `Δw` and its location are reported.
- **Butterfly (model).** Discounted model call prices are computed on a uniform 161-point strike grid
  inside each slice's observed range. The checks are monotonicity (`C` non-increasing and slope
  `≥ −e^{−rT}`) and convexity (second difference `≥ 0`).
- **Risk-neutral density** (Breeden–Litzenberger): `q(K) = e^{rT} · ∂²C/∂K²`, where `C` is the
  discounted call price and the derivative is a central second difference in strike with step `h`.
  It is never clipped; negative mass is reported. The reported mass covers only the observed strike
  range, so values below 1 are expected.
- **SVI butterfly.** Gatheral's `g(k) = (1 − k w′/2w)² − (w′²/4)(1/w + 1/4) + w″/2` is evaluated on
  201 points per slice, together with Lee's wing check `b(1+|ρ|) ≤ 2`.
- **Observed quotes.** Call mids are checked for monotonicity and convexity in strike, plus a strict
  tradable test `bid(K_{i+1}) ≤ ask(K_i)`.
- **Coverage.** For each expiry: quote counts, the k range in absolute terms and in ATM standard
  deviations, whether ATM is covered, and the fit status.

All of these are finite-grid checks. They detect sampled violations only and do not prove the absence
of arbitrage between or beyond the grid points.

## 6. Scenarios
- **Legs.** Each leg has a type, an expiry from the snapshot, a strike, a signed quantity (contracts), a
  premium per unit, and a multiplier (default 100).
- **Accounting.** Initial cost = `Σ qty·premium·mult` (positive means a net debit). Scenario value =
  `Σ qty·V·mult`. P&L = value − initial cost, so it **includes** premium and multipliers. It
  **excludes** financing or carry on premium, fees and margin.
- **Shocks.** Spot moves to `S(1+Δs)`, every leg's IV gets a parallel shift `Δσ` in vol points, and
  calendar time advances by `h` days. Every leg is fully repriced with BSM at `T − h/365`. A leg that
  expires within the horizon settles at intrinsic on the scenario spot. `r` and `q` stay at the
  snapshot values for the leg's expiry.
- **Dynamics.**
  - *Sticky-strike*: `σ_leg = σ_surface(ln(K/F₀(T)), T)`. A fixed strike's vol does not move with spot.
  - *Sticky-log-moneyness*: `σ_leg = σ_surface(ln(K/F₁(T)), T)`, where `F₁` uses the shocked spot. The
    smile translates with the forward.

  In both cases vol is read at the leg's original maturity, with no roll-down.
- **N/A.** A cell shows N/A when the required surface point is outside coverage (with extrapolation
  off) or when the shocked vol is `≤ 0`. Net Greeks use the units in §2, scaled by `qty·mult`.

## 7. Live Deribit option snapshots
`volscape/live/deribit.py` reads `get_book_summary_by_currency?kind=option`. For each expiry, the
forward `F_T` is the median instrument `underlying_price`. The spot `S` is the index price. Deribit quotes
are in coin per unit of underlying, so `USD = coin quote × F_T`. VOLSCAPE then sets `r = 0` and
`q_T = −ln(F_T/S)/T`, so that `S·e^{(r−q)T} = F_T` exactly. Expiry is 08:00 UTC on the coded date, and `T`
is measured from the fetch time on ACT/365.

To verify the convention, IVs were re-solved from Deribit mark prices and compared with their published
`mark_iv`. On a live BTC snapshot the median gap was ≤ 0.2 vol pt per expiry for OTM options (≤ 0.03 for
tenors beyond a week). The remaining difference reflects seconds of timestamp skew. Rows without an ask
are skipped and counted. A missing bid is written as 0 and flagged `ZERO_BID`. Everything else goes
through the same validation, IV, SVI and diagnostic code as any CSV. Deribit options are European and
cash-settled, which is consistent with the BSM convention used here.

## 8. Live NSE / US snapshots (Upstox, Alpaca)
Neither API supplies a forward or a rate. For each expiry VOLSCAPE fits a least-squares line through
`(K, C_mid − P_mid)` over the 10 strikes nearest the spot. For European options `C − P = DF·F − DF·K`,
so `slope = −DF` and `intercept = DF·F`. VOLSCAPE then sets `r_T = −ln(DF)/T` and
`q_T = r_T − ln(F/S)/T`, so `F = S·e^{(r−q)T}`.

An expiry is skipped and reported when it has fewer than 3 two-sided pairs or an implausible
`DF ∉ (0.80, 1.02]`; the forward is never guessed. Expiry times are 15:30 IST for NSE and 16:00 New York
for US.

US equity and ETF options are American. Parity then holds only approximately, and BSM is only an
approximation, so those rows are marked American. Validation rejects them unless the user explicitly
enables the European approximation, which then shows a warning everywhere.

## 9. Synthetic data
The generator is SSVI (Gatheral & Jacquier, 2014):
- `w(k,T) = θ/2·(1 + ρφk + √((φk+ρ)² + 1 − ρ²))`;
- `θ(T) = σ_ATM(T)²T` with `σ_ATM(T) = v_∞ + (v₀ − v_∞)e^{−T/τ}`;
- power-law `φ(θ) = η / (θ^γ (1+θ)^{1−γ})`;
- defaults `ρ = −0.55`, `η = 1.05`, `γ = 0.42`, `v₀ = 0.24`, `v_∞ = 0.19`, `τ = 0.35`.

Because `η(1+|ρ|) ≤ 2` and `γ ≤ 0.5`, the generating surface is free of static arbitrage, and each slice
maps exactly to raw SVI. That mapping makes clean data an identified SVI recovery target.

Quotes are `BSM(IV + seeded Gaussian noise)`. The half-spread is `max(1 tick, 0.35·vega per vol pt + 2% of
price)`, rounded outward to a 0.01 tick, with bids floored at 0. The widths are stylised and make no
claim of market realism.
