"""VOLSCAPE - Streamlit workstation. Run with:  streamlit run app.py  (or start.cmd / start.sh)."""
from __future__ import annotations

import html
import io
import json
import zipfile
from datetime import datetime, timezone
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1  # noqa: F401  (camera keeper)

from volscape import charts as ch
from volscape import diagnostics as dg
from volscape.data import TEMPLATE_CSV, ValidationSettings
from volscape.export import experiment_record, frame_records, surface_grid_frame, to_json
from volscape.iv import IVSettings
from volscape.pipeline import PipelineSettings, run_pipeline
from volscape.pricing import bsm_price
from volscape.scenarios import (STICKY_MONEYNESS, STICKY_STRIKE, initial_cost, legs_from_frame,
                                portfolio_greeks, scenario_pnl)
from volscape.surface import FitSettings, SurfaceModel
from volscape.synthetic import SyntheticSpec, generate_chain
from volscape.live_ui import get_hub, render_live_workspace
from volscape.live.deribit import fetch_book, to_chain

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="VOLSCAPE", layout="wide", initial_sidebar_state="collapsed")

# ------------------------------------------------------------------------------------------ style
st.markdown("""
<style>
:root { --bg:#020604; --panel:#040b07; --line:#10301e; --line2:#1b4a30; --text:#c8efd5; --dim:#7ea58d; --faint:#5f8a72;
        --g:#39ef8b; --g2:#22b46b; --cyan:#3bd4cf; --amber:#e9b54a; --red:#ff5b60;
        --mono: 'Source Code Pro', 'JetBrains Mono', 'Cascadia Mono', 'Cascadia Code', 'SF Mono', SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace; }
[class^='vs-'], [class*=' vs-'] { font-family: var(--mono); }
.stApp { background: var(--bg); }
.stApp::before { content:""; position:fixed; inset:0; pointer-events:none; z-index:0;
  background: repeating-linear-gradient(0deg, rgba(57,239,139,0.018) 0px, rgba(57,239,139,0.018) 1px, transparent 1px, transparent 3px); }
.block-container { padding-top: 0.9rem; padding-bottom: 2rem; max-width: 100%; }
[data-testid="stSidebar"] { background: #010402; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] .block-container { padding-top: 0.8rem; }
header[data-testid="stHeader"] { background: transparent; height: 2rem; }
p, label, span, div { letter-spacing: 0.01em; }
.vs-head { display:flex; align-items:baseline; gap:20px; flex-wrap:wrap; padding: 2px 0 10px 0;
           border-bottom:1px solid var(--line); margin-bottom: 8px; }
.vs-logo { font-weight:600; font-size:1.2rem; letter-spacing:0.24em; color:var(--text); }
.vs-logo span { color: var(--g); }
.vs-logo i { color: var(--faint); font-style:normal; letter-spacing:0; margin-left:6px; }
.vs-badge { font-size:0.70rem; letter-spacing:0.14em; text-transform:uppercase; }
.vs-syn { color: var(--amber); } .vs-imp { color: var(--cyan); }
.vs-meta { color: var(--faint); font-size:0.74rem; display:flex; gap:18px; flex-wrap:wrap; }
.vs-meta b { color: var(--dim); font-weight:500; }
.vs-conv { color: var(--faint); font-size:0.68rem; margin: 0 0 12px 0; }
.vs-cards { display:grid; grid-template-columns: repeat(auto-fit, minmax(118px, 1fr)); gap:0; margin-bottom: 10px;
            border-top:1px solid var(--line); border-bottom:1px solid var(--line); }
.vs-card { padding:8px 14px 9px 12px; border-left:1px solid var(--line); background: transparent; }
.vs-card:first-child { border-left: none; padding-left: 2px; }
.vs-card .l { color: var(--faint); font-size:0.64rem; letter-spacing:0.16em; text-transform:uppercase; }
.vs-card .v { white-space: nowrap; color: var(--g); font-size:1.08rem; font-weight:500; font-variant-numeric: tabular-nums; margin-top:3px; }
.vs-card .s { color: var(--faint); font-size:0.64rem; margin-top:2px; }
.vs-card .v.warn { color: var(--amber); }
/* segmented controls & pills: square hairline chips */
[data-testid="stButtonGroup"] button { border-radius: 0 !important; border: 1px solid var(--line2) !important;
  background: transparent !important; color: var(--dim) !important; font-size: 0.72rem !important;
  letter-spacing: 0.10em; text-transform: uppercase; min-height: 1.9rem; }
[data-testid="stButtonGroup"] button[kind$="Active"], [data-testid="stButtonGroup"] button[aria-checked="true"],
[data-testid="stButtonGroup"] button[aria-pressed="true"] { border-color: var(--g) !important; color: var(--g) !important;
  background: rgba(57,239,139,0.06) !important; }
[data-testid="stRadio"] label p { font-size: 0.78rem; }
/* workspace switch: radio rendered as two hairline tabs */
.st-key-ws [role="radiogroup"] { gap: 0 !important; justify-content: flex-end; flex-wrap: nowrap !important; }
.st-key-ws [role="radiogroup"] > label { border: 1px solid var(--line2); padding: 4px 14px; margin: 0 !important; }
.st-key-ws [role="radiogroup"] > label:has(input:checked) { border-color: var(--g); background: rgba(57,239,139,0.06); }
.st-key-ws [role="radiogroup"] > label:has(input:checked) p { color: var(--g); }
.st-key-ws [role="radiogroup"] > label > div:first-child { display: none; }
.st-key-ws p { letter-spacing: 0.16em; font-size: 0.72rem !important; }
.vs-panel-title { color: var(--dim); font-size:0.66rem; letter-spacing:0.18em; text-transform:uppercase; margin: 2px 0 6px 0; }
.vs-panel-title::before { content: "// "; color: var(--g2); }
.vs-note { color: var(--dim); font-size:0.72rem; border-left: 1px solid var(--amber); padding: 3px 10px; margin: 6px 0 10px 0; }
.vs-ok { color: var(--g); } .vs-bad { color: var(--red); } .vs-warn { color: var(--amber); }
[data-testid="stWidgetLabel"] p { color: var(--dim) !important; font-size: 0.74rem !important; letter-spacing: 0.06em; }
[data-testid="stTabs"] [role="tablist"] { border-bottom: 1px solid var(--line); gap: 0; }
[data-testid="stTabs"] button[role="tab"] { padding: 6px 16px !important; border-right: 1px solid var(--line) !important; }
[data-testid="stTabs"] button[role="tab"]:first-child { padding-left: 2px !important; }
[data-testid="stTabs"] button[role="tab"] p { font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase; color: var(--dim); }
[data-testid="stTabs"] button[aria-selected="true"] p { color: var(--g); }
[data-baseweb="tab-highlight"] { background-color: var(--g) !important; height: 1px !important; }
[data-baseweb="tab-border"] { display: none; }
div[data-testid="stExpander"] details { border: 1px solid var(--line); border-radius: 0; background: transparent; }
div[data-testid="stExpander"] summary p { font-size: 0.74rem; letter-spacing: 0.08em; color: var(--dim); }
.stDownloadButton button, .stButton button { border:1px solid var(--line2); border-radius: 0; background: transparent; white-space: nowrap;
  color: var(--text); font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase; }
.stDownloadButton button:hover, .stButton button:hover { border-color: var(--g); color: var(--g); background: rgba(57,239,139,0.04); }
[data-baseweb="select"] > div, [data-baseweb="input"], .stNumberInput input { border-radius: 0 !important; border-color: var(--line2) !important;
  background: var(--panel) !important; }
[data-testid="stDataFrame"], [data-testid="stDataEditor"] { border: 1px solid var(--line); }
[data-testid="stAlert"] { border-radius: 0; }
hr { border-color: var(--line); }
/* HUD framing: thin corner brackets around every chart panel (no boxes) */
[data-testid="stPlotlyChart"] { position: relative; }
[data-testid="stPlotlyChart"]::before, [data-testid="stPlotlyChart"]::after {
  content: ""; position: absolute; width: 14px; height: 14px; pointer-events: none; z-index: 2; }
[data-testid="stPlotlyChart"]::before { top: 0; left: 0; border-top: 1px solid var(--g2); border-left: 1px solid var(--g2); }
[data-testid="stPlotlyChart"]::after { bottom: 0; right: 0; border-bottom: 1px solid var(--g2); border-right: 1px solid var(--g2); }
/* phosphor glow on live values and the logo accent, nothing else */
.vs-card .v { text-shadow: 0 0 14px rgba(57,239,139,0.28); }
.vs-card .v.warn { text-shadow: 0 0 14px rgba(233,181,74,0.25); }
.vs-logo span { text-shadow: 0 0 18px rgba(57,239,139,0.45); }
.vs-logo i { animation: none; }
.stApp { background: radial-gradient(1400px 700px at 60% -20%, rgba(57,239,139,0.05), transparent 60%), var(--bg) !important; }
</style>
""", unsafe_allow_html=True)


# Streamlit remounts Plotly charts on every rerun, so plotly's uirevision alone cannot keep the
# 3D camera. This injects (once, into the page itself) a tiny script that remembers each 3D
# chart's camera after user rotation/zoom - keyed by chart id and camera revision - and
# re-applies it after a remount. "Reset camera" bumps the revision, which forgets it.
_CAMERA_KEEPER = """
<script>
(function () {
  const P = window.parent, doc = P.document;
  if (doc.getElementById('vs-camera-keeper')) return;
  const s = doc.createElement('script');
  s.id = 'vs-camera-keeper';
  s.textContent = `(function(){
    const store = window.sessionStorage;
    function hook(gd){
      if (gd.__vsHooked || !gd.layout || !String(gd.layout.meta||'').startsWith('vs3d:') || !gd.on) return;
      gd.__vsHooked = true;
      const key = 'vscam:' + gd.layout.meta + ':' + gd.layout.uirevision;
      const saved = store.getItem(key);
      if (saved && window.Plotly) { try { window.Plotly.relayout(gd, {'scene.camera': JSON.parse(saved)}); } catch (e) {} }
      gd.on('plotly_relayout', function(ev){
        if (!Object.keys(ev||{}).some(function(k){ return k.indexOf('scene.camera') === 0; })) return;
        const c = gd._fullLayout && gd._fullLayout.scene && gd._fullLayout.scene.camera;
        if (c) store.setItem(key, JSON.stringify({eye: c.eye, center: c.center, up: c.up}));
      });
    }
    function scan(){ document.querySelectorAll('.js-plotly-plot').forEach(hook); }
    new MutationObserver(scan).observe(document.body, {childList: true, subtree: true});
    setInterval(scan, 600);
  })();`;
  doc.head.appendChild(s);
})();
</script>
"""
st.components.v1.html(_CAMERA_KEEPER, height=0)

PLOT_CFG = {"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 3, "filename": "volscape_chart"},
            "modeBarButtonsToRemove": ["lasso2d", "select2d"]}
# 3D charts: wheel scrolls the page (not the camera); zoom via toolbar zoom mode or the camera reset.
PLOT_CFG_3D = {**PLOT_CFG, "scrollZoom": False}

BUNDLED = {
    "synthetic · demo": ("data/synthetic_demo.csv",
                                       "SYNTHETIC · bundled demo chain (SSVI model, seed 7, 0.15 vol-pt IV noise)", 7),
    "synthetic · clean": ("tests/fixtures/clean_chain.csv", "SYNTHETIC · clean test fixture (SSVI, seed 7, no noise)", 7),
    "synthetic · dirty": ("data/synthetic_dirty.csv",
                                                  "SYNTHETIC · deliberately dirty validation fixture (injected defects)", 7),
}


@st.cache_resource(max_entries=12, show_spinner=False)
def cached_pipeline(content: bytes, label: str, synthetic: bool, vkey: tuple, ivkey: tuple, fkey: tuple, min_vega: float):
    settings = PipelineSettings(
        validation=ValidationSettings(*vkey), iv=IVSettings(sigma_lo=ivkey[0], sigma_hi=ivkey[1]),
        fit=FitSettings(**dict(fkey)), min_vega_volpt=min_vega)
    return run_pipeline(content, label, synthetic, settings)


@st.cache_data(show_spinner=False, max_entries=8)
def deribit_snapshot(currency: str, nonce: int):
    """One Deribit option-book snapshot per (currency, refresh press). Network only when called."""
    rows = fetch_book(currency)
    return to_chain(rows, datetime.now(timezone.utc), currency)


@st.cache_data(show_spinner=False)
def generated_csv(seed: int, noise: float) -> bytes:
    return generate_chain(SyntheticSpec(seed=seed, iv_noise_volpts=noise)).to_csv(index=False).encode()


def fmt(x, spec="{:.2f}", na="N/A"):
    return na if x is None or not np.isfinite(x) else spec.format(x)


# ------------------------------------------------------------------------------------------ sidebar (advanced)
with st.sidebar:
    st.markdown('<div class="vs-panel-title">Advanced settings</div>', unsafe_allow_html=True)
    with st.expander("Validation rules"):
        max_spread = st.slider("Wide-spread flag: (ask−bid)/mid >", 0.1, 2.0, 0.60, 0.05)
        reject_wide = st.checkbox("Reject wide spreads (instead of flag)", False)
        allow_am = st.checkbox("Price American-style rows as European (explicit approximation)", False,
                               help="Off by default: American rows are rejected. When on, they are kept, "
                                    "flagged, and a warning is shown everywhere.")
    with st.expander("IV solver (Brent)"):
        s_lo = st.number_input("σ lower bracket", 1e-5, 0.5, 1e-4, format="%.5f")
        s_hi = st.number_input("σ upper bracket", 0.5, 20.0, 5.0, 0.5)
        min_vega = st.number_input("Low-vega threshold (vega per vol pt, currency)", 0.0, 1.0, 0.005, 0.001, format="%.3f")
    with st.expander("SVI calibration"):
        qsel = st.selectbox("Quotes used", ["otm", "calls", "puts"], format_func={
            "otm": "OTM (puts K<F, calls K≥F)", "calls": "calls only", "puts": "puts only"}.get)
        weighting = st.selectbox("Residual weights", ["spread", "uniform"],
                                 format_func={"spread": "1 / IV half-spread (vega-aware)", "uniform": "uniform"}.get)
        excl_lv = st.checkbox("Exclude low-vega quotes", True)
        min_pts = int(st.number_input("Min distinct strikes per slice", 6, 40, 7))
        max_rmse = st.number_input("Reject slice if IV RMSE > (vol pts)", 0.1, 20.0, 2.5, 0.1)
        n_ms = int(st.slider("Deterministic multistarts", 1, 12, 12))
    st.caption("Network use is opt-in. SURFACE fetches a Deribit public option-book snapshot only when a "
               "'live · deribit' dataset is selected (or refreshed); all other datasets are offline. LIVE opens "
               "Binance (real time) and Yahoo (delayed) connections only after Connect; they stay open in every "
               "workspace until Disconnect.")

if "cam_rev" not in st.session_state:
    st.session_state.cam_rev = 0

# ------------------------------------------------------------------------------------------ header row
h_left, h_ds, h_ws = st.columns([2.5, 1.5, 1.3], gap="small", vertical_alignment="center")
with h_ws:
    ws = st.radio("Workspace", ["SURFACE", "LIVE"], horizontal=True, key="ws", label_visibility="collapsed")
head_ph = h_left.empty()

if ws == "LIVE":
    head_ph.markdown('<div class="vs-head"><div class="vs-logo">VOL<span>SCAPE</span><i>_</i></div>'
                     '<span class="vs-badge vs-imp">[ live · market data ]</span>'
                     '</div>', unsafe_allow_html=True)
    render_live_workspace()
    st.stop()

with h_ds:
    choice = st.selectbox("Dataset", list(BUNDLED) + ["live · deribit BTC", "live · deribit ETH", "synthetic · seed", "import csv"], index=0,
                          label_visibility="collapsed", key="dataset")
seed, spec_dict = None, None
content, label, synthetic = None, "", True
if choice in BUNDLED:
    path, label, seed = BUNDLED[choice]
    content = (ROOT / path).read_bytes()
    spec_dict = SyntheticSpec(seed=7, iv_noise_volpts=0.15 if "demo" in path else 0.0).to_dict() if "dirty" not in path else None
elif choice.startswith("live · deribit"):
    synthetic = False
    cur = choice.split()[-1]
    nonce_key = f"deribit_nonce_{cur}"
    st.session_state.setdefault(nonce_key, 0)
    r1, r2 = st.columns([5, 1], vertical_alignment="center")
    if r2.button("Refresh snapshot", width="stretch"):
        st.session_state[nonce_key] += 1
    try:
        with st.spinner(f"Fetching Deribit {cur} option book (public API)…"):
            content, dnotes = deribit_snapshot(cur, st.session_state[nonce_key])
        label = f"LIVE SNAPSHOT · Deribit {cur} options · public API · fetched {dnotes['fetched_at'][11:19]} UTC"
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(dnotes["fetched_at"])).total_seconds()
        r1.markdown(f'<div class="vs-conv">Deribit public option book · {dnotes["n_raw"]} instruments · '
                    f'{dnotes["n_no_ask"]} without an ask skipped · index {dnotes["index_price"]:,.2f} USD · '
                    f'snapshot age {age:,.0f}s (not streaming; press refresh) · quotes converted coin→USD at each '
                    f'expiry&#39;s forward, r = 0, basis as q (see METHODOLOGY)</div>', unsafe_allow_html=True)
    except Exception as exc:  # noqa: BLE001 - network/HTTP problems are shown, never replaced with other data
        st.error(f"Deribit snapshot unavailable: {type(exc).__name__}: {exc}. No other dataset is substituted.")
        st.stop()
elif choice == "synthetic · seed":
    g1, g2, _ = st.columns([1, 2, 3])
    seed = int(g1.number_input("Seed", 0, 10_000_000, 7, 1))
    noise = g2.slider("IV noise (vol points, Gaussian)", 0.0, 2.0, 0.15, 0.05)
    content = generated_csv(seed, noise)
    label = f"SYNTHETIC · generated in-app (SSVI, seed {seed}, {noise:.2f} vol-pt noise)"
    spec_dict = SyntheticSpec(seed=seed, iv_noise_volpts=noise).to_dict()
else:
    synthetic = False
    u1, u2, u3 = st.columns([3, 1, 1], vertical_alignment="bottom")
    up = u1.file_uploader("Option chain CSV", type=["csv"])
    u2.download_button("CSV template", TEMPLATE_CSV, "volscape_template.csv", "text/csv", width="stretch")
    u3.download_button("Example chain", (ROOT / "data/example_chain.csv").read_bytes(),
                       "volscape_example_chain.csv", "text/csv", width="stretch")
    if up is not None:
        content = up.getvalue()
        label = f"IMPORTED · {up.name}"

# ------------------------------------------------------------------------------------------ pipeline
if content is None:
    head_ph.markdown('<div class="vs-head"><div class="vs-logo">VOL<span>SCAPE</span><i>_</i></div>'
                     '<span class="vs-badge vs-imp">[ import · awaiting csv ]</span></div>', unsafe_allow_html=True)
    st.info("Upload a CSV with columns: valuation_time, expiry, strike, option_type, bid, ask, spot, rate, "
            "dividend_yield (optional: underlying, currency, volume, open_interest, exercise_style). "
            "Timestamps must include a timezone offset, e.g. 2026-06-15T20:00:00+00:00. "
            "Nothing is shown until a file is imported — the demo is never substituted silently.")
    st.stop()

fkey = tuple(sorted(dict(quote_selection=qsel, weighting=weighting, exclude_low_vega=excl_lv, min_points=min_pts,
                         max_rmse_volpts=float(max_rmse), n_multistart=n_ms).items()))
with st.spinner("Validating quotes, solving implied vols, calibrating SVI slices…"):
    res = cached_pipeline(content, label, synthetic, (float(max_spread), reject_wide, allow_am), (float(s_lo), float(s_hi)),
                          fkey, float(min_vega))
val = res.validation
meta = val.meta
badge = '<span class="vs-badge vs-syn">[ synthetic · not market data ]</span>' if meta.is_synthetic else     '<span class="vs-badge vs-imp">[ live snapshot · deribit · market data ]</span>' if label.startswith("LIVE SNAPSHOT") else \
    '<span class="vs-badge vs-imp">[ imported observations ]</span>'
vt = meta.valuation_time.strftime("%Y-%m-%d %H:%M UTC") if meta.valuation_time is not None else "—"
live_on = get_hub().connected
live_mark = ('<span class="vs-badge vs-imp">◉ live feed connected · disconnect in LIVE</span>' if live_on else "")
head_ph.markdown(f'<div class="vs-head"><div class="vs-logo">VOL<span>SCAPE</span><i>_</i></div>{badge}{live_mark}</div>',
                 unsafe_allow_html=True)
st.markdown(f"""<div class="vs-meta"><span>src <b>{html.escape(meta.source_label)}</b></span>
<span>und <b>{html.escape(meta.underlying)}</b></span><span>ccy <b>{html.escape(meta.currency)}</b></span>
<span>t₀ <b>{vt}</b></span><span>S <b>{fmt(meta.spot, '{:,.4f}', '—')}</b></span>
<span>sha256 <b>{meta.input_sha256[:12]}…</b></span></div>
<div class="vs-conv">vp = vol points · model: {html.escape(meta.model_convention)} · {html.escape(meta.exercise_note)}</div>
""", unsafe_allow_html=True)

if not res.ok:
    st.error("This dataset could not be used. " + " ".join(val.fatal))
    st.caption("The previous or demo dataset is NOT substituted. Fix the file and re-import.")
    if len(val.rejected):
        st.dataframe(val.rejected, width="stretch", hide_index=True)
        st.download_button("Download row-level rejection reasons", val.rejected.to_csv(index=False), "rejected_rows.csv")
    st.stop()

for w in val.warnings:
    st.markdown(f'<div class="vs-note">{html.escape(w)}</div>', unsafe_allow_html=True)
for m in res.messages:
    st.markdown(f'<div class="vs-note">{html.escape(m)}</div>', unsafe_allow_html=True)

chain = res.chain
expiries = sorted(chain.expiry.unique())

# ------------------------------------------------------------------------------------------ control strip
methods = [m for m, mod in (("svi", res.svi), ("baseline", res.baseline)) if mod is not None and mod.available]
if not methods:
    st.error("No surface can be built: fewer than two usable maturities.")
s1, s2, s3, s4, s5 = st.columns([1.15, 1.0, 1.25, 1.0, 1.35], gap="small", vertical_alignment="bottom")
method = s1.radio("Method", methods or ["svi"], horizontal=True, format_func={"svi": "SVI", "baseline": "linear w"}.get)
z_mode = s2.radio("Z axis", ["iv", "w"], horizontal=True, format_func={"iv": "IV %", "w": "w"}.get)
x_mode = s3.selectbox("X axis", ["k", "moneyness", "strike"],
                      format_func={"k": "k = ln(K/F)", "moneyness": "K/F", "strike": "strike K"}.get)
cscale = s4.selectbox("Colorscale", list(ch.COLORSCALES))
default_exp = int(np.argmin([abs((e - meta.valuation_time).days - 30) for e in expiries]))
sel = s5.selectbox("Inspect expiry", expiries, index=default_exp,
                   format_func=lambda e: f"{e:%Y-%m-%d} · {(e - meta.valuation_time).total_seconds() / 86400:.0f}d")
l1, l2 = st.columns([4.2, 1.2], gap="small", vertical_alignment="center")
layers = l1.pills("Layers", ["points", "slices", "contours", "wireframe", "extrapolate k"], selection_mode="multi",
                  default=["points", "slices", "contours"], label_visibility="collapsed", key="layers") or []
show_pts, show_slices, contours = "points" in layers, "slices" in layers, "contours" in layers
wire, extrap = "wireframe" in layers, "extrapolate k" in layers
if l2.button("Reset camera", width="stretch", help="Restore the default 3D view"):
    st.session_state.cam_rev += 1

base_model = res.model(method)
model = SurfaceModel(method, base_model.slices, extrapolate_k=extrap) if base_model is not None else None
diag = res.diagnostics(method)
hd = diag["headline"]

cards = [
    ("ATM IV · 30D", fmt(hd["atm_iv_30d"] * 100, "{:.2f}%"), "fwd ATM · k=0", False),
    ("Skew · 30D", fmt(hd["skew_30d_volpts"], "{:+.2f}"), "σ(−.10)−σ(+.10) · vp", False),
    ("Term slope", fmt(hd["term_slope_volpts"], "{:+.2f}"), f"{hd['term_slope_T'] * 365:.0f}d−30d ATM · vp"
     if np.isfinite(hd["term_slope_T"]) else "unavailable", False),
    ("Calibration RMSE", fmt(diag["aggregate"]["iv_rmse_volpts"], "{:.3f}"), f"vp · {method} · fit set",
     diag["aggregate"]["iv_rmse_volpts"] > 1 if np.isfinite(diag["aggregate"]["iv_rmse_volpts"]) else True),
    ("Valid quotes", f"{len(chain)} / {val.n_input}", f"{len(val.rejected)} rej · {int((chain.iv_mid_status == 'OK').sum())} iv ok",
     len(val.rejected) > 0),
    ("SVI slices", f"{sum(f.usable for f in res.fits)} / {len(res.fits)}", "usable / tried",
     any(not f.usable for f in res.fits)),
]
if model is not None and model.available:
    fig3d = ch.surface_figure(model, chain, res.obs, x_mode=x_mode, z_mode=z_mode, colorscale=cscale,
                              show_points=show_pts, wireframe=wire, contours=contours, highlight_expiry=sel,
                              show_slices=show_slices, uirevision=f"cam{st.session_state.cam_rev}", height=600)
    st.plotly_chart(fig3d, key="surface3d", config=PLOT_CFG_3D, theme=None)
    st.markdown('<div class="vs-conv">sheet = k range observed in BOTH adjacent expiries (linear in w across T) · '
                'slices = each expiry&#39;s fit over its own observed range · drag = orbit · right-drag = pan · '
                'zoom = toolbar zoom mode · wheel scrolls the page · camera persists until reset</div>',
                unsafe_allow_html=True)
else:
    fig3d = None
    st.warning("Surface unavailable for this method — see messages above and the Calibration tab.")

st.markdown('<div class="vs-cards">' + "".join(
    f'<div class="vs-card"><div class="l">{l}</div><div class="v{" warn" if w else ""}">{v}</div><div class="s">{s}</div></div>'
    for l, v, s, w in cards) + "</div>", unsafe_allow_html=True)

c_smile, c_term = st.columns(2, gap="medium")
with c_smile:
    sm_x = st.radio("Smile axis", ["k", "moneyness", "strike"], horizontal=True,
                    format_func={"k": "k", "moneyness": "K/F", "strike": "K"}.get, key="smx")
    fig_smile = ch.smile_figure(res, sel, sm_x, show_baseline=True, extrapolate=extrap)
    st.plotly_chart(fig_smile, key="smile", config=PLOT_CFG, theme=None)
with c_term:
    st.markdown('<div style="height:2.35rem"></div>', unsafe_allow_html=True)
    fig_term = ch.term_figure(res, sel)
    st.plotly_chart(fig_term, key="term", config=PLOT_CFG, theme=None)

# ------------------------------------------------------------------------------------------ tabs
t_cal, t_q, t_arb, t_risk, t_exp = st.tabs(["Calibration", "Quote Quality", "Arbitrage Diagnostics", "Risk Playground", "Export"])

NUM = st.column_config.NumberColumn
DT = st.column_config.DatetimeColumn

with t_cal:
    a, b = st.columns([1.25, 1], gap="medium")
    with a:
        st.markdown('<div class="vs-panel-title">Raw SVI slices · w(k) = a + b[ρ(k−m) + √((k−m)² + σ²)]</div>', unsafe_allow_html=True)
        sv = pd.DataFrame([f.to_row() for f in res.fits])
        sv["expiry"] = pd.to_datetime(sv.expiry)
        sv["days"] = sv["T"] * 365
        st.dataframe(sv[["expiry", "days", "status", "n_points", "a", "b", "rho", "m", "sigma", "rmse_volpts",
                         "max_abs_err_volpts", "lee_wing_ok", "best_start", "n_starts", "nfev", "message"]],
                     hide_index=True, width="stretch", column_config={
                         "expiry": DT(format="YYYY-MM-DD"), "days": NUM(format="%.0f"),
                         **{p: NUM(format="%.6f") for p in ("a", "b", "rho", "m", "sigma")},
                         "rmse_volpts": NUM("RMSE (vol pts)", format="%.3f"),
                         "max_abs_err_volpts": NUM("max |err| (vol pts)", format="%.3f"),
                         "lee_wing_ok": st.column_config.CheckboxColumn("b(1+|ρ|) ≤ 2")})
        st.caption("Bounded trust-region least squares (scipy `least_squares`, method `trf`) on IV residuals, "
                   "parametrised so min_k w(k) ≥ 0; deterministic multistart grid over ρ and σ. Slices with too few "
                   "strikes, parameters pinned at bounds, or IV RMSE above the threshold are rejected and excluded "
                   "from the surface — never hidden. Basic parameter bounds do NOT imply absence of arbitrage; see "
                   "the Arbitrage Diagnostics tab.")
        with st.expander("Recorded optimizer settings"):
            st.json(res.settings.to_dict())
    with b:
        st.markdown('<div class="vs-panel-title">Method comparison (all IV-solvable quotes vs fit set)</div>', unsafe_allow_html=True)
        rows = []
        for mth in ("svi", "baseline"):
            mo = res.model(mth)
            if mo is None or not mo.slices:
                continue
            dd = res.diagnostics(mth)
            for sub, agg in (("fit set", dd["aggregate"]), ("all quotes", dd["aggregate_all"])):
                rows.append(dict(method=mth.upper(), subset=sub, **agg))
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
            "iv_rmse_volpts": NUM("IV RMSE (vp)", format="%.3f"), "iv_max_abs_volpts": NUM("IV max (vp)", format="%.3f"),
            "price_rmse": NUM("price RMSE", format="%.4f"), "price_max_abs": NUM("price max", format="%.4f"),
            "pct_in_band": NUM("in bid/ask %", format="%.1f")})
        st.caption("The baseline interpolates the fit set exactly, so its fit-set error is ~0 by construction; "
                   "compare methods on 'all quotes' (includes ITM quotes not used in fitting).")
        st.plotly_chart(ch.per_expiry_error_figure(diag["per_expiry"]), key="perexp", config=PLOT_CFG, theme=None)
    st.plotly_chart(ch.residual_stems(diag["recon"]), key="resid3d", config=PLOT_CFG_3D, theme=None)
    st.markdown('<div class="vs-panel-title">Per-expiry reconstruction errors</div>', unsafe_allow_html=True)
    st.dataframe(diag["per_expiry"], hide_index=True, width="stretch", column_config={
        "expiry": DT(format="YYYY-MM-DD"), "T": NUM(format="%.4f"), "iv_rmse_volpts": NUM(format="%.3f"),
        "iv_max_abs_volpts": NUM(format="%.3f"), "price_rmse": NUM(format="%.4f"), "price_max_abs": NUM(format="%.4f"),
        "pct_in_band": NUM("model in bid/ask %", format="%.1f")})

with t_q:
    n_flag = int((chain["flags"] != "").sum())
    st.markdown(f"""<div class="vs-cards" >
    <div class="vs-card"><div class="l">input rows</div><div class="v">{val.n_input}</div></div>
    <div class="vs-card"><div class="l">accepted</div><div class="v">{len(chain)}</div></div>
    <div class="vs-card"><div class="l">rejected</div><div class="v{' warn' if len(val.rejected) else ''}">{len(val.rejected)}</div></div>
    <div class="vs-card"><div class="l">accepted w/ flags</div><div class="v{' warn' if n_flag else ''}">{n_flag}</div></div>
    <div class="vs-card"><div class="l">low-vega quotes</div><div class="v">{int(chain.low_vega.sum())}</div>
    <div class="s">vega/vol-pt &lt; {min_vega:.3f}</div></div></div>""", unsafe_allow_html=True)
    a, b = st.columns([1.3, 1], gap="medium")
    with a:
        st.markdown('<div class="vs-panel-title">Rejected rows (row-level reasons)</div>', unsafe_allow_html=True)
        if len(val.rejected):
            st.dataframe(val.rejected, hide_index=True, width="stretch", height=260)
            st.download_button("Download rejected rows (CSV)", val.rejected.to_csv(index=False), "rejected_rows.csv", "text/csv")
        else:
            st.caption("No rows rejected.")
        st.markdown('<div class="vs-panel-title">IV solver status by quote side</div>', unsafe_allow_html=True)
        stat = pd.DataFrame({s: chain[f"iv_{s}_status"].value_counts() for s in ("bid", "mid", "ask")}).fillna(0).astype(int)
        st.dataframe(stat, width="stretch")
        st.caption("INVALID_PRICE on bids is expected for deep-ITM quotes whose bid lies below the model-free lower bound "
                   "(including zero bids). NO_BRACKET = inside bounds but outside the σ bracket. These are reported, not repaired.")
    with b:
        if len(val.rejected):
            st.plotly_chart(ch.rejection_figure(val.rejected), key="rej", config=PLOT_CFG, theme=None)
        fl = chain["flags"][chain["flags"] != ""].str.split(";").explode().value_counts()
        if len(fl):
            st.markdown('<div class="vs-panel-title">Accepted-with-flag counts</div>', unsafe_allow_html=True)
            st.dataframe(fl.rename("rows"), width="stretch")
    st.plotly_chart(ch.spread_figure(chain), key="spread", config=PLOT_CFG, theme=None)
    a, b = st.columns(2, gap="medium")
    with a:
        par = dg.parity_check(chain)
        st.markdown('<div class="vs-panel-title">Put–call parity (paired C/P, same expiry & strike)</div>', unsafe_allow_html=True)
        if len(par):
            st.caption(f"{int(par.within_band.sum())} / {len(par)} pairs consistent: C_bid − P_ask ≤ S·e^(−qT) − K·e^(−rT) ≤ C_ask − P_bid. "
                       f"Median |mid residual| = {par.parity_residual_mid.abs().median():.4f} {meta.currency}.")
            st.dataframe(par.sort_values("within_band"), hide_index=True, width="stretch", height=260, column_config={
                "expiry": DT(format="YYYY-MM-DD"), "k": NUM(format="%.4f"),
                **{c: NUM(format="%.4f") for c in ("call_mid", "put_mid", "theoretical", "parity_residual_mid", "lower", "upper")}})
        else:
            st.caption("No compatible call/put pairs.")
    with b:
        st.markdown('<div class="vs-panel-title">Coverage by expiry</div>', unsafe_allow_html=True)
        cov = dg.coverage(chain, res.obs, res.fits)
        st.dataframe(cov, hide_index=True, width="stretch", height=260, column_config={
            "expiry": DT(format="YYYY-MM-DD"), "days": NUM(format="%.0f"), "k_min": NUM(format="%.3f"), "k_max": NUM(format="%.3f"),
            "sd_min": NUM("k_min (ATM sd)", format="%.2f"), "sd_max": NUM("k_max (ATM sd)", format="%.2f")})
    with st.expander("Accepted quotes with solved IVs"):
        show = chain[["csv_line", "expiry", "strike", "option_type", "bid", "ask", "mid", "k", "iv_bid", "iv_mid", "iv_ask",
                      "iv_mid_status", "iv_halfwidth", "vega_volpt", "low_vega", "flags"]]
        st.dataframe(show, hide_index=True, width="stretch", height=320, column_config={
            "expiry": DT(format="YYYY-MM-DD"), "k": NUM(format="%.4f"),
            **{c: NUM(format="%.4f") for c in ("iv_bid", "iv_mid", "iv_ask", "iv_halfwidth", "vega_volpt")}})

with t_arb:
    st.markdown(f'<div class="vs-note">{dg.FINITE_GRID_NOTE} Violations are reported as computed — densities are not clipped and '
                'surfaces are not smoothed. A raw-SVI fit is not called arbitrage-free because its parameters satisfy bounds.</div>',
                unsafe_allow_html=True)
    if model is None or not model.slices:
        st.warning("No slices available for diagnostics.")
    else:
        a, b = st.columns([1, 1.25], gap="medium")
        with a:
            st.markdown('<div class="vs-panel-title">Calendar: w(k, T₂) ≥ w(k, T₁) on shared k grid</div>', unsafe_allow_html=True)
            cal = diag["calendar"]
            if len(cal):
                st.dataframe(cal, hide_index=True, width="stretch", column_config={
                    "expiry_near": DT(format="YYYY-MM-DD"), "expiry_far": DT(format="YYYY-MM-DD"),
                    "worst_dw": NUM("worst Δw", format="%.2e"), "k_at_worst": NUM(format="%.3f")})
                st.caption("Grid: 101 points over the intersection of the two slices' observed k ranges.")
        with b:
            fig_cal = ch.calendar_surface(model, dg.calendar_grid(model))
            if fig_cal is not None:
                st.plotly_chart(fig_cal, key="cal3d", config=PLOT_CFG_3D, theme=None)
        a, b = st.columns([1, 1.25], gap="medium")
        with a:
            st.markdown('<div class="vs-panel-title">Butterfly: call-price monotonicity, convexity, density</div>', unsafe_allow_html=True)
            st.dataframe(diag["call_checks"], hide_index=True, width="stretch", column_config={
                "expiry": DT(format="YYYY-MM-DD"), "T": NUM(format="%.4f"), "strike_step": NUM(format="%.4f"),
                "min_density": NUM(format="%.3e"), "negative_mass": NUM(format="%.2e"), "mass_in_range": NUM(format="%.4f")})
            st.caption("Model DISCOUNTED call prices on a uniform 161-point strike grid within each slice's observed range. "
                       "Density q(K) = e^{rT}·∂²C/∂K² (central second difference in strike). mass_in_range < 1 is expected: "
                       "probability outside the observed strikes is not estimated.")
            if method == "svi" and len(diag["butterfly"]):
                st.markdown('<div class="vs-panel-title">SVI Gatheral g(k) ≥ 0 (201-pt grid, observed range)</div>', unsafe_allow_html=True)
                st.dataframe(diag["butterfly"], hide_index=True, width="stretch", column_config={
                    "expiry": DT(format="YYYY-MM-DD"), "T": NUM(format="%.4f"), "min_g": NUM(format="%.4f"),
                    "k_at_min": NUM(format="%.3f"), "lee_wing_ok": st.column_config.CheckboxColumn("Lee wing b(1+|ρ|) ≤ 2")})
            st.markdown('<div class="vs-panel-title">Observed call mids (raw quotes, no model)</div>', unsafe_allow_html=True)
            st.dataframe(dg.observed_call_checks(chain), hide_index=True, width="stretch", column_config={
                "expiry": DT(format="YYYY-MM-DD"), "mono_mid": "monotonic viol. (mid)",
                "mono_quotes": "monotonic viol. (bid>prev ask)", "convex_mid": "convexity viol. (mid)"})
            st.caption("Mid-based violations on raw quotes can be within bid/ask noise; the strict quote test uses tradable prices.")
        with b:
            if diag["density"]:
                st.plotly_chart(ch.density_waterfall(diag["density"], chain, sel), key="dens3d",
                                config=PLOT_CFG_3D, theme=None)
                if sel not in diag["density"]:
                    st.caption("Selected expiry has no density (slice unavailable for this method).")
            else:
                st.caption("No density available for this method.")

with t_risk:
    st.markdown('<div class="vs-note">European legs repriced with BSM under each scenario. P&L = scenario value − initial cost; '
                'includes premium paid/received and the contract multiplier; excludes financing/carry of premium, fees and margin. '
                'Legs expiring before the horizon settle at intrinsic value on the scenario spot. Vols come from the selected '
                f'surface method ({method.upper()}) at each leg\'s original maturity (no roll-down).</div>', unsafe_allow_html=True)
    if model is None or not model.available:
        st.warning("Risk playground needs an available surface.")
    else:
        exp_labels = [f"{e:%Y-%m-%d}" for e in expiries]
        S0 = float(meta.spot)
        key = meta.input_sha256[:10] + method
        if st.session_state.get("legs_key") != key:
            e60 = expiries[int(np.argmin([abs((e - meta.valuation_time).days - 60) for e in expiries]))]
            strikes = np.sort(chain.loc[chain.expiry == e60, "strike"].unique())
            F60 = float(chain.loc[chain.expiry == e60, "forward"].iloc[0])
            near = lambda x: float(strikes[np.argmin(np.abs(strikes - x))])
            legs0 = pd.DataFrame(dict(type=["Call", "Call", "Put"], expiry=[f"{e60:%Y-%m-%d}"] * 3,
                                      strike=[near(F60), near(F60 * 1.05), near(F60 * 0.95)],
                                      quantity=[1.0, -1.0, 1.0], premium=[np.nan] * 3,
                                      multiplier=[1.0 if label.startswith("LIVE SNAPSHOT") else 100.0] * 3))
            st.session_state.legs_df = legs0
            st.session_state.legs_key = key
            st.session_state.legs_ver = st.session_state.get("legs_ver", 0) + 1
            st.session_state.fill_premiums = True

        def model_premiums(df):
            out = df.copy()
            legs, _ = legs_from_frame(out.assign(premium=out.premium.fillna(0.0)), chain)
            vals = []
            for _, r in out.iterrows():
                try:
                    e = expiries[exp_labels.index(str(r["expiry"]))]
                    g = chain[chain.expiry == e].iloc[0]
                    kq = np.log(float(r["strike"]) / float(g.forward))
                    sig = float(model.iv(kq, float(g["T"]), extrapolate=extrap))
                    vals.append(round(float(bsm_price(S0, float(r["strike"]), float(g["T"]), float(g.rate), float(g.dividend_yield),
                                                      sig, str(r["type"]).startswith("C"))), 4) if np.isfinite(sig) else np.nan)
                except (ValueError, KeyError, TypeError, IndexError):
                    vals.append(np.nan)
            out["premium"] = vals
            return out

        if st.session_state.get("fill_premiums"):
            st.session_state.legs_df = model_premiums(st.session_state.legs_df)
            st.session_state.fill_premiums = False

        a, b = st.columns([1.5, 1], gap="medium")
        with a:
            st.markdown('<div class="vs-panel-title">Legs (premium per unit of underlying; edit to your execution price)</div>', unsafe_allow_html=True)
            edited = st.data_editor(
                st.session_state.legs_df, num_rows="dynamic", hide_index=True, width="stretch",
                key=f"legs_editor_{st.session_state.legs_ver}",
                column_config={
                    "type": st.column_config.SelectboxColumn(options=["Call", "Put"], required=True),
                    "expiry": st.column_config.SelectboxColumn(options=exp_labels, required=True),
                    "strike": NUM(min_value=0.0001, format="%.2f", required=True),
                    "quantity": NUM("quantity (contracts, ±)", format="%.2f", required=True),
                    "premium": NUM("premium / unit", min_value=0.0, format="%.4f"),
                    "multiplier": NUM(min_value=0.0001, format="%.0f", required=True)})
            if st.button("Set premiums to current model value"):
                st.session_state.legs_df = model_premiums(edited)
                st.session_state.legs_ver += 1
                st.rerun()
        legs, errs = legs_from_frame(edited.dropna(subset=["type", "expiry", "strike", "quantity", "multiplier"]), chain)
        for e in errs:
            st.warning(e)
        if edited.premium.isna().any():
            st.warning("Some legs have no premium (model value unavailable or blank) and are excluded until a premium is entered.")
        with b:
            st.markdown('<div class="vs-panel-title">Scenario settings</div>', unsafe_allow_html=True)
            dyn = st.radio("Surface dynamics", [STICKY_STRIKE, STICKY_MONEYNESS, "compare"], horizontal=True,
                           help="sticky-strike: σ(K) fixed as spot moves (k computed with the ORIGINAL forward). "
                                "sticky-log-moneyness: σ is a function of k = ln(K/F) with F from the SHOCKED spot, "
                                "so the smile translates with the forward.")
            s_rng = st.slider("Spot shock range (±%)", 5, 50, 20, 5)
            v_rng = st.slider("IV shock range (± vol pts)", 2, 30, 10, 1)
            max_days = int(max(l.T for l in legs) * 365) if legs else 0
            hz = st.slider("Time forward (calendar days)", 0, max(max_days + 5, 1), 0, 1,
                           help="Legs reaching expiry within the horizon settle at intrinsic on the scenario spot.")
            show3d = st.toggle("3D P&L surface", True)
        if legs:
            g = portfolio_greeks(legs, model, S0, extrapolate=extrap)
            cost = initial_cost(legs)
            exp_legs = [l.label for l in legs if l.T * 365 <= hz]
            cur = meta.currency
            st.markdown(f"""<div class="vs-cards" >
            <div class="vs-card"><div class="l">initial cost</div><div class="v">{cost:,.2f}</div><div class="s">{cur}, + = net debit</div></div>
            <div class="vs-card"><div class="l">model value</div><div class="v">{fmt(g['model_value'], '{:,.2f}')}</div><div class="s">{cur} at surface vols</div></div>
            <div class="vs-card"><div class="l">Δ delta</div><div class="v">{fmt(g['delta'], '{:,.2f}')}</div><div class="s">underlying units</div></div>
            <div class="vs-card"><div class="l">Γ gamma</div><div class="v">{fmt(g['gamma'], '{:,.4f}')}</div><div class="s">Δ per 1 {cur} spot</div></div>
            <div class="vs-card"><div class="l">vega</div><div class="v">{fmt(g['vega_volpt'], '{:,.2f}')}</div><div class="s">{cur} per vol pt</div></div>
            <div class="vs-card"><div class="l">Θ theta</div><div class="v">{fmt(g['theta_day'], '{:,.2f}')}</div><div class="s">{cur} per calendar day</div></div>
            <div class="vs-card"><div class="l">ρ rho</div><div class="v">{fmt(g['rho_pct'], '{:,.2f}')}</div><div class="s">{cur} per 1% rate</div></div>
            </div>""", unsafe_allow_html=True)
            if exp_legs:
                st.caption("Expired at horizon (intrinsic settlement): " + ", ".join(exp_legs))
            ss = np.linspace(-s_rng / 100, s_rng / 100, 25)
            vs = np.linspace(-v_rng, v_rng, 21)
            dyns = [STICKY_STRIKE, STICKY_MONEYNESS] if dyn == "compare" else [dyn]
            pnls = {d: scenario_pnl(legs, model, S0, ss, vs, hz, d, extrap) for d in dyns}
            side_3d = show3d and dyn != "compare"
            cols = st.columns(len(dyns) + (1 if dyn == "compare" else 0) + (1 if side_3d else 0), gap="small")
            for c, d in zip(cols, dyns):
                with c:
                    st.plotly_chart(ch.pnl_heatmap(pnls[d], ss, vs, f"P&L · {d} · +{hz}d", cur), key=f"pnl_{d}", config=PLOT_CFG, theme=None)
            if dyn == "compare":
                with cols[-1]:
                    st.plotly_chart(ch.pnl_heatmap(pnls[STICKY_MONEYNESS] - pnls[STICKY_STRIKE], ss, vs,
                                                   "Difference: sticky-log-moneyness − sticky-strike", cur), key="pnl_diff",
                                    config=PLOT_CFG, theme=None)
            n_na = int(np.isnan(pnls[dyns[0]]).sum())
            if n_na:
                st.caption(f"{n_na} scenario cells are N/A: the required surface point is outside observed coverage "
                           "(enable k-extrapolation to evaluate, explicitly marked) or the shocked IV is ≤ 0.")
            if side_3d:
                with cols[-1]:
                    st.plotly_chart(ch.pnl_surface(pnls[dyns[0]], ss, vs, cur, height=420), key="pnl3d",
                                    config=PLOT_CFG_3D, theme=None)
            elif show3d:
                st.plotly_chart(ch.pnl_surface(pnls[dyns[0]], ss, vs, cur), key="pnl3d", config=PLOT_CFG_3D, theme=None)
        else:
            st.info("Add at least one valid leg.")

with t_exp:
    st.markdown('<div class="vs-panel-title">Reproducible exports</div>', unsafe_allow_html=True)
    rec = experiment_record(res, method, seed=seed if synthetic else None, synthetic_spec=spec_dict)
    exp_json = to_json(rec)
    grid_csv = surface_grid_frame(model).to_csv(index=False) if model is not None and model.available else ""
    svi_csv = pd.DataFrame([f.to_row() for f in res.fits]).to_csv(index=False)
    acc_csv = chain.to_csv(index=False)
    rej_csv = val.rejected.to_csv(index=False)
    diag_csv = diag["per_expiry"].assign(method=method).to_csv(index=False)
    arb_csv = pd.concat([diag["calendar"].assign(check="calendar"), diag["call_checks"].assign(check="call_price"),
                         diag["butterfly"].assign(check="svi_g")], ignore_index=True).to_csv(index=False)
    htmls = {"surface_3d.html": (fig3d, "VOLSCAPE · IV surface"), "smile.html": (fig_smile, "VOLSCAPE · smile"),
             "term_structure.html": (fig_term, "VOLSCAPE · ATM term structure")}
    a, b, c = st.columns(3, gap="medium")
    with a:
        st.download_button("Experiment record (JSON)", exp_json, "volscape_experiment.json", "application/json", width="stretch")
        st.download_button("Surface grid (CSV)", grid_csv, f"surface_grid_{method}.csv", "text/csv", width="stretch", disabled=not grid_csv)
        st.download_button("SVI parameters & optimizer status (CSV)", svi_csv, "svi_parameters.csv", "text/csv", width="stretch")
    with b:
        st.download_button("Accepted quotes + IVs (CSV)", acc_csv, "accepted_quotes.csv", "text/csv", width="stretch")
        st.download_button("Rejected rows + reasons (CSV)", rej_csv, "rejected_rows.csv", "text/csv", width="stretch")
        st.download_button("Error diagnostics (CSV)", diag_csv, f"errors_{method}.csv", "text/csv", width="stretch")
        st.download_button("Arbitrage diagnostics (CSV)", arb_csv, f"arbitrage_{method}.csv", "text/csv", width="stretch")
    with c:
        for fn, (fig, title) in htmls.items():
            if fig is not None:
                st.download_button(f"Interactive chart: {fn}", ch.standalone_html(fig, title), fn, "text/html", width="stretch")
        st.caption("HTML charts embed plotly.js from the installed package and work offline. Use the chart toolbar camera "
                   "icon for 3× PNG export.")
    if st.button("Write full bundle to exports/ folder"):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = ROOT / "exports" / f"{stamp}_{method}"
        out.mkdir(parents=True, exist_ok=True)
        files = {"experiment.json": exp_json, "surface_grid.csv": grid_csv, "svi_parameters.csv": svi_csv,
                 "accepted_quotes.csv": acc_csv, "rejected_rows.csv": rej_csv, "errors.csv": diag_csv, "arbitrage.csv": arb_csv}
        for fn, (fig, title) in htmls.items():
            if fig is not None:
                files[fn] = ch.standalone_html(fig, title)
        for fn, txt in files.items():
            (out / fn).write_text(txt, encoding="utf-8")
        st.success(f"Wrote {len(files)} files to {out.relative_to(ROOT)}")
    with st.expander("Experiment record preview"):
        st.code(exp_json[:6000] + ("\n…" if len(exp_json) > 6000 else ""), language="json")

with st.expander("Methodology & assumptions"):
    st.markdown(r"""
**Pricing.** European Black–Scholes–Merton with continuous rate $r$ and dividend yield $q$;
$F = S e^{(r-q)T}$, $C = e^{-rT}[F N(d_1) - K N(d_2)]$, $d_{1,2} = \frac{\ln(F/K) \pm \tfrac12\sigma^2T}{\sigma\sqrt T}$.
$T$ is ACT/365 fixed from timezone-aware timestamps. At $T\le0$ the value is intrinsic; at $\sigma=0$ it is the discounted forward intrinsic.

**Implied vol.** Brent root-finding on $[\sigma_{lo}, \sigma_{hi}]$ after model-free bound checks
$\max(Se^{-qT}-Ke^{-rT},0) \le C \le Se^{-qT}$. Statuses: OK, INVALID_INPUT, EXPIRED, INVALID_PRICE, NO_BRACKET, NOT_CONVERGED.
IV uncertainty ≈ half-spread / vega; quotes with vega per vol point below the threshold are flagged LOW-VEGA and excluded from fitting by default.

**Surfaces.** Coordinates $k=\ln(K/F)$, $w=\sigma^2T$. Baseline: linear in $w$ across $k$ per expiry and linear in $w$ across $T$.
Raw SVI: $w(k)=a+b[\rho(k-m)+\sqrt{(k-m)^2+\sigma^2}]$ per expiry, bounded least squares on IV residuals.
No extrapolation in $T$; $k$-extrapolation is off by default and visibly marked when on.

**Diagnostics.** Calendar: $w(k,T_2)\ge w(k,T_1)$ on a shared grid. Butterfly: discounted call prices monotone/convex in $K$;
density $q(K)=e^{rT}\partial^2C/\partial K^2$; SVI $g(k)\ge0$. Finite-grid checks detect sampled violations and do not prove global absence of arbitrage.

**Scenarios.** Full BSM repricing under spot × vol × time shocks; sticky-strike vs sticky-log-moneyness defined in the Risk tab.
Full derivations: `METHODOLOGY.md`.
""")
