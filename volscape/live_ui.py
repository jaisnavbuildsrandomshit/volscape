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
from .live.markets import EQUITIES, fetch_equity_chart

import asyncio
import shutil
from pathlib import Path

import plotly
import plotly.graph_objects as go
import streamlit.components.v1 as components

_ROOT = Path(__file__).resolve().parents[1]
_STATIC = _ROOT / "static" / "plotly.min.js"        # served by Streamlit at /app/static/ (enableStaticServing)
if not _STATIC.exists():
    _STATIC.parent.mkdir(exist_ok=True)
    shutil.copy(Path(plotly.__file__).parent / "package_data" / "plotly.min.js", _STATIC)
_PANEL = (Path(__file__).parent / "live_panel.html").read_text(encoding="utf-8")

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
    market = st.radio("Market", ["crypto · real time", "stocks & indices · delayed"], horizontal=True,
                      key="live_market", label_visibility="collapsed")
    if market.startswith("stocks"):
        _render_equities()
        return
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
    # Charts live in one browser-side panel that keeps its own Binance stream and updates in place
    # (Plotly.react): Streamlit never rebuilds them, so there is no per-tick flicker or lag.
    components.html(_PANEL.replace("__SYMBOL__", sym), height=1460, scrolling=False)
    _live_panel(hub)


@st.cache_data(ttl=60, show_spinner=False)
def _equity_day(symbol: str):
    return asyncio.run(fetch_equity_chart(symbol))


def _render_equities() -> None:
    names = {f"{e.name} · {e.group}": e.symbol for e in EQUITIES}
    c1, c2 = st.columns([1.6, 4], vertical_alignment="bottom")
    pick = c1.selectbox("Stock / index", list(names), key="eq_pick")
    c2.markdown('<div class="vs-conv">Yahoo Finance 1-minute bars for the current/last session · DELAYED (not exchange '
                'real-time) · refreshed every 60 s · no public order book exists for these, so no depth landscape · '
                'option surfaces for NSE / US underlyings: SURFACE → live · NSE (Upstox) / live · US (Alpaca)</div>',
                unsafe_allow_html=True)
    if not st.session_state.get("eq_ok"):
        if st.button("Load delayed day chart (connects to Yahoo Finance)", type="primary"):
            st.session_state.eq_ok = True
            st.rerun()
        return
    _equity_panel(names[pick])


@st.fragment(run_every=60.0)
def _equity_panel(symbol: str) -> None:
    try:
        chart = _equity_day(symbol)
    except Exception as exc:  # noqa: BLE001 - shown, never replaced
        st.error(f"Day chart unavailable for {symbol}: {exc}")
        return
    if not chart.bars:
        st.warning("No bars returned for this session (market may not have opened yet).")
        return
    x = pd.to_datetime([b.t for b in chart.bars], unit="s", utc=True).tz_convert(chart.tz_name or "UTC")
    o, h, l, c = ([getattr(b, f) for b in chart.bars] for f in "ohlc")
    v = [b.v for b in chart.bars]
    ref = chart.prev_close
    chg = None if not ref else (chart.price / ref - 1) * 100 if chart.price else None
    is_open = chart.market_open(datetime.now(timezone.utc).timestamp())
    cards = [("Last", _n(chart.price, "{:,.2f}"), f"{chart.currency} · delayed", False),
             ("Change", _n(chg, "{:+.2f}%"), "vs previous close", False),
             ("Day high", _n(chart.day_high, "{:,.2f}"), "session", False),
             ("Day low", _n(chart.day_low, "{:,.2f}"), "session", False),
             ("Prev close", _n(ref, "{:,.2f}"), chart.exchange, False),
             ("Market", "OPEN" if is_open else "CLOSED", chart.tz_name, not is_open)]
    st.markdown('<div class="vs-cards">' + "".join(
        f'<div class="vs-card"><div class="l">{a}</div><div class="v{" warn" if w else ""}">{b}</div>'
        f'<div class="s">{d}</div></div>' for a, b, d, w in cards) + "</div>", unsafe_allow_html=True)
    fig = go.Figure([
        go.Candlestick(x=x, open=o, high=h, low=l, close=c, name=chart.name,
                       increasing=dict(line=dict(color=ch.TEAL, width=1), fillcolor=ch.TEAL),
                       decreasing=dict(line=dict(color=ch.RED, width=1), fillcolor=ch.RED)),
        go.Bar(x=x, y=v, yaxis="y2", marker_color=["rgba(57,239,139,0.25)" if cc >= oo else "rgba(255,91,96,0.25)"
                                                    for cc, oo in zip(c, o)], hovertemplate="vol %{y:,.0f}<extra></extra>")])
    if ref:
        fig.add_hline(y=ref, line=dict(color=ch.AMBER, width=1, dash="dot"), annotation_text="prev close",
                      annotation_font=dict(color=ch.AMBER, size=10))
    ch.base_layout(fig, height=520, title=f"{chart.name} · {chart.exchange} · 1-minute bars · session of "
                                         f"{x[-1]:%Y-%m-%d} ({chart.tz_name}) · delayed")
    fig.update_layout(showlegend=False, uirevision="eq", xaxis=dict(rangeslider=dict(visible=False)),
                      yaxis=dict(domain=[0.22, 1]), yaxis2=dict(domain=[0, 0.18]))
    fig.update_xaxes(**ch._axis(f"time ({chart.tz_name})"))
    fig.update_yaxes(**ch._axis(""))
    st.plotly_chart(fig, key="eq_day", config=PLOT_CFG, theme=None)


@st.fragment(run_every=2.0)
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
