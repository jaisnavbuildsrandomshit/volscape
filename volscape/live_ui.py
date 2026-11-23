"""LIVE workspace (Streamlit). Market data from the vendored orderflow-terminal reader.

Nothing connects until the user presses Connect. The panel refreshes in a fragment once per
second so the rest of the page (and the SURFACE workspace) never reruns on a tick.
"""
from __future__ import annotations

import html
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

from . import charts as ch
from .live.hub import LiveHub
from .live.state import SYMBOLS

PLOT_CFG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


@st.cache_resource(show_spinner=False)
def get_hub() -> LiveHub:
    # one hub per server process: a local single-user workstation shares one connection
    return LiveHub()


def _n(x, spec="{:,.2f}", na="N/A"):
    return na if x is None or (isinstance(x, float) and not np.isfinite(x)) else spec.format(x)


def _dp(sym: str) -> int:
    s = SYMBOLS.get(sym)
    return s.price_dp if s else 2


def _status_class(status: str) -> str:
    return {"LIVE": "vs-ok", "DELAYED": "vs-warn"}.get(status, "vs-bad")


def render_live_workspace() -> None:
    hub = get_hub()
    syms = list(SYMBOLS)
    c1, c2, c3 = st.columns([1.3, 1, 3.7], gap="small", vertical_alignment="bottom")
    sym = c1.selectbox("Symbol (Binance Spot)", syms, index=syms.index(hub.symbol), key="live_sym")
    if hub.connected:
        if c2.button("Disconnect", width="stretch"):
            hub.stop()
            st.rerun()
        hub.set_symbol(sym)
    else:
        if c2.button("Connect", width="stretch", type="primary"):
            hub.start(sym)
            st.rerun()
    c3.markdown('<div class="vs-conv">Binance Spot public market data (real time, no keys) · Yahoo Finance quotes '
                '(delayed, not exchange real-time) · reader vendored from orderflow-terminal (MIT). '
                'Market data is shown as received; it never feeds the synthetic option surface.</div>',
                unsafe_allow_html=True)
    if not hub.connected:
        st.markdown('<div class="vs-note">Not connected. Press Connect to open the public feeds. No network '
                    'connection is made before that.</div>', unsafe_allow_html=True)
        return
    _landscape_panel(hub)
    _live_panel(hub)


@st.fragment(run_every=5.0)
def _landscape_panel(hub: LiveHub) -> None:
    # Own fragment at a slower cadence: Streamlit rebuilds the WebGL scene on every update, so updating
    # it every second flickers. Fixed axis ranges keep each rebuild visually stable.
    c1, c2 = st.columns([6, 1], vertical_alignment="center")
    paused = c2.toggle("Pause 3D", False, key="land_pause")
    if paused and "land_fig" in st.session_state:
        fig = st.session_state.land_fig
    else:
        fig = ch.book_landscape(hub.snapshot().book_history)
        st.session_state.land_fig = fig
    c1.markdown('<div class="vs-conv">order-book landscape · refreshes every 5 s · pause to inspect</div>',
                unsafe_allow_html=True)
    if fig is not None:
        st.plotly_chart(fig, key="live_land", config={**PLOT_CFG, "scrollZoom": False}, theme=None)
        st.markdown('<div class="vs-conv">terrain = displayed top-20 resting quantity per bin around the then-current '
                    'mid (window adapts in coarse steps), one sample per second (bounded to 180 s) · empty rows = '
                    'seconds without a book update · not total supply/demand</div>', unsafe_allow_html=True)
    else:
        st.caption("Order-book landscape needs at least 2 seconds of book samples.")


@st.fragment(run_every=1.0)
def _live_panel(hub: LiveHub) -> None:
    s = hub.snapshot()
    dp = _dp(s.symbol)
    pfmt = "{:,.%df}" % dp
    utc = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    age = _n(s.receive_age_s, "{:.1f}s")
    stale = s.receive_age_s is None or s.receive_age_s > 5
    st.markdown(
        f'<div class="vs-meta"><span>feed <b class="{_status_class(s.status)}">{html.escape(s.status)}</b></span>'
        f'<span>sym <b>{s.symbol}</b></span><span>endpoint <b>{html.escape(s.detail or "—")}</b></span>'
        f'<span>recv age <b class="{"vs-warn" if stale else ""}">{age}</b></span>'
        f'<span>msgs <b>{s.msg_rate:.0f}/s</b></span><span>trades <b>{_n(s.trade_rate, "{:.1f}")}/s</b></span><span>gaps <b>{s.gaps}</b></span>'
        f'<span>dup <b>{s.duplicates}</b></span><span>missed <b>{s.missed}</b></span>'
        f'<span>watchlist <b class="{_status_class(s.crypto_status)}">crypto {s.crypto_status}</b> · '
        f'<b class="{_status_class(s.yahoo_status)}">equities {s.yahoo_status}</b></span>'
        f'<span>clock <b>{utc}</b></span></div>', unsafe_allow_html=True)
    if stale and s.status == "LIVE":
        st.markdown('<div class="vs-note">No message for more than 5 s: values below may be stale.</div>',
                    unsafe_allow_html=True)
    warm = s.window_s < 60
    cards = [
        ("Last", _n(s.last, pfmt), "trade price", False),
        ("Mid", _n(s.mid, pfmt), "(bid+ask)/2", False),
        ("Spread", _n(s.spread_bps, "{:.3g} bp"), _n(s.spread, pfmt) + " abs", False),
        ("Imbalance", _n(s.imbalance, "{:+.3f}"), "top-20 qty", False),
        ("VWAP", _n(s.vwap, pfmt), "since connect", False),
        ("Flow · 60s", _n(s.signed_flow_60s, "{:+,.0f}"), "warming up" if warm else "buy − sell notional",
         warm),
    ]
    st.markdown('<div class="vs-cards">' + "".join(
        f'<div class="vs-card"><div class="l">{l}</div><div class="v{" warn" if w else ""}">{v}</div>'
        f'<div class="s">{sub}</div></div>' for l, v, sub, w in cards) + "</div>", unsafe_allow_html=True)

    a, b = st.columns([1, 1.25], gap="medium")
    with a:
        st.plotly_chart(ch.live_depth_figure(s.bids, s.asks), key="live_depth", config=PLOT_CFG, theme=None)
    with b:
        st.plotly_chart(ch.live_price_figure(s.price_series, s.vwap), key="live_px", config=PLOT_CFG, theme=None)
    a, b = st.columns([1, 1.25], gap="medium")
    with a:
        st.markdown('<div class="vs-panel-title">Trade tape · newest first</div>', unsafe_allow_html=True)
        if s.trades:
            tape = pd.DataFrame(s.trades, columns=["time_ms", "price", "qty", "side", "notional"])
            tape["time"] = pd.to_datetime(tape.time_ms, unit="ms", utc=True).dt.strftime("%H:%M:%S.%f").str[:-3]
            st.dataframe(tape[["time", "side", "price", "qty", "notional"]], hide_index=True, width="stretch",
                         height=330, column_config={
                             "price": st.column_config.NumberColumn(format=f"%.{dp}f"),
                             "qty": st.column_config.NumberColumn(format="%.5f"),
                             "notional": st.column_config.NumberColumn(format="%.2f")})
        else:
            st.caption("No trades received yet.")
    with b:
        if s.flow_series:
            st.plotly_chart(ch.live_flow_figure(s.flow_series), key="live_flow", config=PLOT_CFG, theme=None)
        else:
            st.caption("Flow histogram warming up.")
        if s.events:
            st.markdown('<div class="vs-panel-title">Feed events (deduplicated)</div>', unsafe_allow_html=True)
            st.markdown("<div class='vs-conv'>" + "<br>".join(
                f"{datetime.fromtimestamp(t, timezone.utc):%H:%M:%S} {html.escape(m)}" for t, m in s.events[:6])
                + "</div>", unsafe_allow_html=True)

    st.markdown('<div class="vs-panel-title">Watchlist · crypto real time (Binance) · equities delayed (Yahoo)</div>',
                unsafe_allow_html=True)
    if s.yahoo_error:
        st.markdown(f'<div class="vs-note">Yahoo: {html.escape(s.yahoo_error)} — equity rows show N/A.</div>',
                    unsafe_allow_html=True)
    rows = []
    for q in s.quotes:
        rng = None
        if q["price"] is not None and q["high"] is not None and q["low"] is not None and q["high"] > q["low"]:
            rng = (q["price"] - q["low"]) / (q["high"] - q["low"])
        rows.append(dict(group=q["group"], name=q["name"], symbol=q["symbol"],
                         feed="LIVE" if q["source"] == "BINANCE" else "DELAYED",
                         price=q["price"], chg_pct=q["change_pct"], day_range=rng,
                         trend=q["series"][-48:] if q["series"] else None))
    if rows:
        wl = pd.DataFrame(rows)
        for c in ("price", "chg_pct", "day_range"):
            wl[c] = pd.to_numeric(wl[c], errors="coerce")   # missing -> NaN (blank), never 0
        st.dataframe(wl, hide_index=True, width="stretch", height=420, column_config={
            "price": st.column_config.NumberColumn(format="%,.4g"),
            "chg_pct": st.column_config.NumberColumn("chg %", format="%+.2f"),
            "day_range": st.column_config.ProgressColumn("position in range", min_value=0.0, max_value=1.0,
                                                         format="%.2f"),
            "trend": st.column_config.LineChartColumn("trend")})
        st.caption("Empty cells = no price received yet or source unavailable; missing values are never filled in. "
                   "Crypto change is vs 24h open; "
                   "equity change is vs previous close.")
