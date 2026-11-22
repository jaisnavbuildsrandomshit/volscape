# HANDOFF

## Status
The full application works end to end: data, IV, surfaces, diagnostics, scenarios, exports and the UI.
The UI uses a minimal "phosphor" terminal theme with hairline separators, monospace type and a green
colorscale.

## Commands (last run results)
- `pytest`: 267 passed.
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
- The Quote Quality and Export tabs, the dirty fixture and the import-empty state were checked through
  the headless smoke run plus a brief manual look, not in a full visual review.
