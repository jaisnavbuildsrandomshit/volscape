# VOLSCAPE

A local, offline options-volatility research workstation: validate an option chain, solve bid/mid/ask
implied vols, fit an inspectable surface (total-variance interpolation baseline and raw SVI), audit it
with static-arbitrage diagnostics, and stress a small European portfolio. Built with Python, NumPy,
SciPy, pandas, Plotly and Streamlit. Everything runs on your machine, with no accounts, APIs or network
services at runtime.

> The bundled datasets are **SYNTHETIC**, generated from a disclosed SSVI model with a fixed seed.
> They are not market data, and the interface labels them that way everywhere.

![3D surface workstation](docs/screenshots/workstation_surface.jpg)

| Smile · term structure · calibration | Arbitrage diagnostics | Risk playground |
|---|---|---|
| ![](docs/screenshots/smile_term_calibration.jpg) | ![](docs/screenshots/arbitrage_diagnostics.jpg) | ![](docs/screenshots/risk_playground.jpg) |

*Screenshots were captured from the running app (bundled synthetic demo, 1600×1000 viewport).*

## Quick start

Requires Python 3.11+ (3.12 recommended). The first run needs internet once to install pinned wheels.
After that the app works fully offline: Streamlit and Plotly serve their bundled assets locally.

**Windows** (PowerShell, or double-click `start.cmd`)
```powershell
.\start.cmd
```
In `cmd.exe`, type `.\start.cmd` with the path: a bare `start.cmd` is parsed as cmd's built-in `start`.
**macOS / Linux**
```bash
chmod +x start.sh && ./start.sh
```
Then open http://127.0.0.1:8501. Both scripts locate their own folder, pick a suitable Python, create
`.venv`, install `requirements.txt` only when it changes, and print actionable errors. They never change
global settings. Run the test suite with `.\start.cmd test` or `./start.sh test`.

Manual equivalent:
```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     # macOS/Linux: .venv/bin/python
.venv/Scripts/python -m streamlit run app.py
```

## Workflow
1. **Load**: bundled synthetic demo, clean fixture, deliberately dirty fixture, a freshly seeded
   synthetic chain, or **import a CSV**. A failed import is reported with row-level reasons. The demo is
   never substituted silently.
2. **Validate**: required columns and types, timezone-aware timestamps, positive spot and strike,
   `ask ≥ bid`, one snapshot spot and valuation time, rate and dividend consistent within each expiry,
   duplicates (every copy rejected), expired contracts, and model-free price bounds. Wide spreads, zero
   bids and mid-below-bound quotes are flagged rather than rejected. American-style rows are rejected
   unless you explicitly opt into a European approximation, which then shows a warning everywhere.
3. **Solve IV**: Brent inversion for bid, mid and ask, with distinct statuses (`OK`, `INVALID_INPUT`,
   `EXPIRED`, `INVALID_PRICE`, `NO_BRACKET`, `NOT_CONVERGED`). IV uncertainty is the half-spread divided
   by vega, and low-vega quotes are flagged.
4. **Fit**: baseline linear-in-total-variance interpolation, or raw SVI per expiry. SVI uses bounded
   trust-region least squares from a deterministic multistart grid. Settings, status, residuals and
   parameters are recorded, and unsupported or degenerate slices are rejected.
5. **Explore**: the 3D surface (rotate, zoom, contours, wireframe, observed points, colorscales,
   camera reset), a linked smile and ATM term structure, and four tabs: Calibration, Quote Quality,
   Arbitrage Diagnostics and Risk Playground.
6. **Export**: surface grid, SVI parameters, accepted and rejected quotes, error and arbitrage
   diagnostics as CSV, standalone interactive HTML charts with plotly.js embedded, and a
   machine-readable experiment record (input SHA-256, source label, full settings, seed, dependency
   versions, results).

### CSV schema
Required: `valuation_time, expiry, strike, option_type, bid, ask, spot, rate, dividend_yield`.
Optional: `underlying, currency, volume, open_interest, exercise_style`.
Timestamps must carry an offset, for example `2026-06-15T20:00:00+00:00`. A template and an example
chain can be downloaded inside the app (Import CSV), and `data/example_chain.csv` is in the repo.

## Numerical conventions (summary)
- European Black–Scholes–Merton with continuous `r` and `q`, and ACT/365 fixed year fractions from
  timezone-aware timestamps.
- `F = S·e^{(r−q)T}`, `k = ln(K/F)`, total variance `w = σ²T`.
- At `T ≤ 0` the value is intrinsic. At `σ = 0` it is the discounted forward intrinsic.
- Greeks are per unit of underlying: delta, gamma, vega per 1.00 vol (shown per vol point), theta
  per year (shown per calendar day), and rho per 1.00 rate (shown per 1%).
- No extrapolation in maturity. Extrapolation in k is off by default, and when enabled it is drawn
  as a translucent grey sheet and dashed smile segments.

Full derivations are in [METHODOLOGY.md](METHODOLOGY.md). Results from the bundled experiment are in
[docs/RESULTS_SYNTHETIC.md](docs/RESULTS_SYNTHETIC.md), regenerated by `python scripts/run_experiment.py`.

## Reading the 3D surface
The green sheet covers only the k range observed in **both** adjacent expiries, interpolated linearly
in total variance across T. Thin lines show each expiry's fit over its own observed range, so the
"fins" at the wing edges show coverage that neighbouring expiries lack. Gaps are deliberate: sparse
coverage is shown, not hidden.

Controls: drag to orbit, right-drag to pan, and zoom with the toolbar's zoom mode. The mouse wheel
scrolls the page rather than the camera. The camera persists across control changes until you press
**Reset camera**. The toolbar camera icon exports a 3× PNG.

## Architecture
```
volscape/
  conventions.py   units, ACT/365, model convention string
  pricing.py       BSM price, bounds, Greeks, parity residual (vectorised, explicit limits)
  iv.py            Brent IV with distinct statuses; chain-level bid/mid/ask solve, vega, IV half-width
  data.py          CSV loading, row-level validation, snapshot metadata, template
  synthetic.py     seeded SSVI chain generator + dirty-fixture injector
  surface.py       baseline interpolation, raw SVI fitting, SurfaceModel (k,T) interface, display grid
  diagnostics.py   reconstruction errors, calendar/butterfly/convexity/density, parity, coverage
  scenarios.py     portfolio legs, net Greeks, spot×vol×time P&L, sticky-strike vs sticky-log-moneyness
  pipeline.py      CSV bytes → validated chain → IV → fits → diagnostics
  charts.py        themed Plotly figures and standalone HTML export
  export.py        experiment record (JSON) and grid exports
app.py             Streamlit workstation
scripts/           generate_data.py, run_experiment.py, smoke_app.py
tests/             pytest suite (pricing, IV, data, surface, diagnostics, scenarios)
```

## Verification
- `pytest`: 267 tests. They cover known-answer prices and Greeks, parity, IV round trips, limits,
  finite-difference Greeks, determinism, SVI recovery on SSVI slices, interpolation, known
  calendar/butterfly violations, CSV rejections and duplicates, portfolio accounting, and the
  display-grid no-extrapolation invariant.
- `python scripts/smoke_app.py`: a headless Streamlit run through every main interaction.
- Manual inspection in a Chromium-based browser on Windows, including camera persistence and the
  screenshots above. `start.sh` has not been executed on macOS or Linux.

## Limitations
- European exercise only. American contracts need an explicit approximation opt-in, which ignores the
  early-exercise premium.
- Each slice is fitted independently with raw SVI. There is no joint arbitrage-free surface
  parametrisation such as SSVI. Diagnostics report violations; they do not repair them.
- Finite-grid arbitrage checks detect sampled violations only. They do not prove the global absence of
  arbitrage.
- Scenario vols are read at each leg's original maturity, with no roll-down along the term structure.
  P&L excludes financing, fees and margin.
- The synthetic bid/ask widths are stylised and make no claim of realism.
- SVI calibration on the bundled chain takes about 5 seconds and is cached per dataset and settings.
