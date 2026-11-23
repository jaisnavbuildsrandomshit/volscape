# HANDOFF

## Status
The full application works end to end: data, IV, surfaces, diagnostics, scenarios, exports and the UI.
The UI uses a minimal "phosphor" terminal theme with hairline separators, monospace type and a green
colorscale.

## Commands (last run results)
- `pytest`: see the verification section of README.md for the latest count.
- `python scripts/smoke_app.py`: SMOKE OK. This is a headless Streamlit run through every main interaction.
- `python scripts/run_experiment.py`: regenerates `exports/synthetic_demo/` and `docs/RESULTS_SYNTHETIC.md`.
- `.\start.cmd test` (Windows): 267 passed, exit 0. `start.sh` has LF endings but has not been executed on macOS/Linux.

## Decisions worth knowing
- Streamlit remounts Plotly charts on every rerun, so `uirevision` cannot keep the 3D camera. A small
  script injected once into the page (`_CAMERA_KEEPER` in `app.py`) saves the camera to sessionStorage,
  keyed by chart id and revision, and re-applies it after each remount.
- Wheel zoom is disabled on 3D charts so the page scrolls. Zoom is available from the toolbar.
- The display grid core covers only the intersection of adjacent slices' observed k ranges (see
  METHODOLOGY §4), and the "fins" are real coverage differences.
- `.streamlit/config.toml` sets `fileWatcherType = "none"`: restart the server after editing modules.

## Known issues / next steps
- No joint SSVI/eSSVI calibration across expiries yet; slices are fitted independently.
- All SURFACE tabs and the LIVE workspace were viewed in the browser at pane width (~660–1024 px) and at
  1600x1000. The dirty fixture and the import-empty state are covered by the headless smoke run only.
- LIVE: the watchlist `MarketBoard` mutates quotes on the feed thread without the hub lock (simple
  attribute writes; snapshot copies them). Good enough for display; revisit if more logic is added.
- A live options source (e.g. a public exchange options API) would be needed for a live IV surface.
