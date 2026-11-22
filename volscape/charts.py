"""Plotly figures with the VOLSCAPE workstation theme. All values plotted are computed values."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .surface import SurfaceModel

BG = "#020604"
PANEL = "#030a06"
GRID = "#0c2617"
AXIS = "#1b4a30"
TEXT = "#c8efd5"
MUTED = "#6e977f"
TEAL = "#39ef8b"      # primary phosphor accent (observed / fit set)
CYAN = "#3bd4cf"      # model curves
AMBER = "#e9b54a"     # diagnostics / warnings
RED = "#ff5b60"
VIOLET = "#9fb8a8"    # baseline (neutral, dotted)
FONT = "'Source Code Pro', 'JetBrains Mono', 'Cascadia Mono', 'SF Mono', SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"

COLORSCALES = {
    "Phosphor": [[0, "#010805"], [0.2, "#03301a"], [0.5, "#0d7a42"], [0.8, "#2fd97c"], [1, "#8dffc0"]],
    "Abyss": [[0, "#081a2e"], [0.25, "#0b3b5c"], [0.5, "#0f766e"], [0.75, "#2dd4bf"], [1, "#f0f9c8"]],
    "Viridis": "Viridis", "Cividis": "Cividis", "Ice": "ice", "Plasma": "Plasma", "Thermal": "thermal",
}
X_LABEL = {"k": "log-moneyness k = ln(K/F)", "moneyness": "moneyness K/F", "strike": "strike K"}


def _axis(title, **kw):
    return dict(title=dict(text=title, font=dict(color=MUTED, size=12)), gridcolor=GRID, zerolinecolor=AXIS,
                linecolor=AXIS, tickfont=dict(color=MUTED, size=11), **kw)


def base_layout(fig, height=360, title=None, **kw):
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=TEXT, size=11), margin=dict(l=56, r=18, t=70 if title else 30, b=48),
        # title pinned to the top of the figure; legend sits on its own row between title and plot
        title=dict(text=title, font=dict(size=12, color=TEXT), x=0.0, xanchor="left", xref="paper",
                   y=0.985, yref="container", yanchor="top") if title else None,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED, size=10), orientation="h",
                    yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        hoverlabel=dict(bgcolor="#03100a", bordercolor=AXIS, font=dict(family=FONT, color=TEXT, size=12)),
        **kw)
    return fig


def _rq_interp(chain: pd.DataFrame):
    g = chain.groupby("expiry").agg(T=("T", "first"), r=("rate", "first"), q=("dividend_yield", "first")).sort_values("T")
    S = float(chain.spot.iloc[0])
    return lambda T: S * np.exp((np.interp(T, g["T"], g.r) - np.interp(T, g["T"], g.q)) * T)


def x_transform(k, fwd, mode):
    if mode == "k":
        return k
    if mode == "moneyness":
        return np.exp(k)
    return fwd * np.exp(k)


def surface_figure(model: SurfaceModel, chain: pd.DataFrame, obs: pd.DataFrame, *, x_mode="k", z_mode="iv",
                   colorscale="Phosphor", show_points=True, wireframe=False, contours=True,
                   highlight_expiry=None, uirevision="cam", height=640):
    grids = model.grid(n_k=71, n_t=48)
    fwd_of = _rq_interp(chain)
    conv = (lambda iv, T: iv * 100) if z_mode == "iv" else (lambda iv, T: iv ** 2 * T)
    sheets = {name: (x_transform(KK, fwd_of(TT), x_mode), TT * 365, conv(IV, TT)) for name, (KK, TT, IV) in grids.items()}
    X, D, Zc = sheets["core"]
    ztitle = "implied vol (%)" if z_mode == "iv" else "total variance w = IV²·T"
    zfmt = ".2f" if z_mode == "iv" else ".5f"
    finite = Zc[np.isfinite(Zc)]
    cmin, cmax = (float(np.nanpercentile(finite, 1)), float(np.nanpercentile(finite, 99))) if finite.size else (0, 1)
    cs = COLORSCALES.get(colorscale, colorscale)
    fig = go.Figure()
    ctr = dict(z=dict(show=contours, usecolormap=True, highlightcolor=TEXT, project=dict(z=False), width=1),
               x=dict(show=False, highlight=False), y=dict(show=False, highlight=False))
    hover = (f"x %{{x:.4f}}<br>days %{{y:.1f}}<br>{'IV %' if z_mode == 'iv' else 'w'} %{{z:{zfmt}}}<extra>%{{fullData.name}}</extra>")
    for side in ("left", "right"):
        if side in sheets:
            Xe, De, Ze = sheets[side]
            fig.add_trace(go.Surface(x=Xe, y=De, z=Ze, name="extrapolated in k", showscale=False, opacity=0.4,
                                     showlegend=side == "left", colorscale=[[0, "#3b4558"], [1, "#9aa6ba"]],
                                     hovertemplate=hover.replace("<extra>", "<br>EXTRAPOLATED<extra>"),
                                     contours=dict(z=dict(show=False))))
    fig.add_trace(go.Surface(
        x=X, y=D, z=Zc, name=f"{model.method.upper()} surface", colorscale=cs, cmin=cmin, cmax=cmax,
        contours=ctr, opacity=1.0 if not wireframe else 0.82, hovertemplate=hover,
        lighting=dict(ambient=0.62, diffuse=0.55, specular=0.12, roughness=0.75, fresnel=0.05),
        lightposition=dict(x=1000, y=-2000, z=3000),
        colorbar=dict(title=dict(text=ztitle, side="right", font=dict(color=MUTED, size=11)), thickness=12,
                      len=0.62, x=1.0, tickfont=dict(color=MUTED, size=10), outlinewidth=0)))
    if wireframe:
        xs, ys, zs = [], [], []
        nt, nk = X.shape
        for i in list(range(0, nt, max(1, nt // 16))) + [nt - 1]:
            xs += list(X[i]) + [None]; ys += list(D[i]) + [None]; zs += list(Zc[i]) + [None]
        for j in list(range(0, nk, max(1, nk // 18))) + [nk - 1]:
            xs += list(X[:, j]) + [None]; ys += list(D[:, j]) + [None]; zs += list(Zc[:, j]) + [None]
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name="wireframe", hoverinfo="skip",
                                   line=dict(color="rgba(213,222,235,0.28)", width=1.5)))
    if show_points and obs is not None and len(obs):
        o = obs
        zo = o.iv_mid * 100 if z_mode == "iv" else o.w_obs
        xo = x_transform(o.k, o.forward, x_mode)
        fig.add_trace(go.Scatter3d(
            x=xo, y=o["T"] * 365, z=zo, mode="markers", name="observed mid IV (fit set)",
            marker=dict(size=2.2, color="#e4ffef", line=dict(width=0)),
            customdata=np.c_[o.strike, o.option_type, o.iv_bid * 100, o.iv_ask * 100],
            hovertemplate="K %{customdata[0]:.2f} %{customdata[1]}<br>days %{y:.1f}<br>mid %{z:" + zfmt +
                          "}<br>bid/ask IV %{customdata[2]:.2f} / %{customdata[3]:.2f}<extra>observed</extra>"))
    # each expiry's fitted smile over ITS OWN observed k range: shows where coverage extends beyond
    # the range shared with neighbouring expiries (the surface sheet only spans the shared range)
    xs, ys, zs = [], [], []
    for s in model.slices:
        lo, hi = model.k_range(s)
        kk = np.linspace(lo, hi, 60)
        iv = np.sqrt(np.maximum(model.slice_w(s, kk, False), 0) / s.T)
        xs += list(x_transform(kk, fwd_of(s.T), x_mode)) + [None]
        ys += [s.T * 365] * 60 + [None]
        zs += list(conv(iv, s.T)) + [None]
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name="fitted slice (observed range)",
                               line=dict(color="rgba(141,255,192,0.55)", width=2), hoverinfo="skip"))
    if highlight_expiry is not None:
        s = next((s for s in model.slices if s.expiry == highlight_expiry), None)
        if s is not None:
            lo, hi = model.k_range(s)
            kk = np.linspace(lo, hi, 80)
            iv = model.iv(kk, s.T, extrapolate=False)
            fig.add_trace(go.Scatter3d(x=x_transform(kk, fwd_of(s.T), x_mode), y=np.full(80, s.T * 365),
                                       z=conv(iv, s.T), mode="lines", name="selected expiry",
                                       line=dict(color=CYAN, width=4), hoverinfo="skip"))
    sc_axis = lambda t: dict(title=dict(text=t, font=dict(color=MUTED, size=11)), backgroundcolor=PANEL,
                             gridcolor=GRID, zerolinecolor=AXIS, showbackground=True,
                             tickfont=dict(color=MUTED, size=10), color=MUTED)
    base_layout(fig, height=height)
    fig.update_layout(
        meta="vs3d:surface", uirevision=uirevision, margin=dict(l=0, r=0, t=8, b=0),
        legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top", orientation="v", bgcolor="rgba(0,0,0,0)"),
        scene=dict(xaxis=sc_axis(X_LABEL[x_mode]), yaxis=sc_axis("days to expiry (ACT/365)"), zaxis=sc_axis(ztitle),
                   aspectmode="manual", aspectratio=dict(x=1.45, y=1.45, z=0.8), bgcolor=BG,
                   camera=dict(eye=dict(x=-1.3, y=-1.35, z=0.72), center=dict(x=0, y=0, z=-0.1)),
                   uirevision=uirevision))
    return fig


def smile_figure(res, expiry, x_mode="k", show_baseline=True, extrapolate=False, height=380):
    ch = res.chain[(res.chain.expiry == expiry)]
    used = set(zip(res.obs.expiry, res.obs.strike, res.obs.option_type))
    ch = ch.assign(used=[(e, k, t) in used for e, k, t in zip(ch.expiry, ch.strike, ch.option_type)])
    T = float(ch["T"].iloc[0])
    F = float(ch.forward.iloc[0])
    fig = go.Figure()
    ok = ch[ch.iv_mid_status == "OK"]
    for subset, name, color, sym in ((ok[ok.used], "mid IV (fit set)", TEAL, "circle"),
                                     (ok[~ok.used & ~ok.low_vega], "mid IV (not fitted)", MUTED, "circle-open"),
                                     (ok[ok.low_vega], "low-vega (excluded)", AMBER, "x-thin-open")):
        if subset.empty:
            continue
        up = (subset.iv_ask - subset.iv_mid) * 100
        dn = (subset.iv_mid - subset.iv_bid) * 100
        fig.add_trace(go.Scatter(
            x=x_transform(subset.k, F, x_mode), y=subset.iv_mid * 100, mode="markers", name=name,
            marker=dict(color=color, size=7, symbol=sym, line=dict(width=1.4, color=color)),
            error_y=dict(type="data", symmetric=False, array=up.fillna(0), arrayminus=dn.fillna(0),
                         color="rgba(139,153,176,0.55)", thickness=1, width=0),
            customdata=np.c_[subset.strike, subset.option_type, subset.bid, subset.ask,
                             subset.iv_bid * 100, subset.iv_ask * 100, subset.vega_volpt],
            hovertemplate=("K %{customdata[0]:.2f} %{customdata[1]}  bid/ask %{customdata[2]:.2f}/%{customdata[3]:.2f}"
                           "<br>IV mid %{y:.2f}%  bid %{customdata[4]:.2f}  ask %{customdata[5]:.2f}"
                           "<br>vega/vol-pt %{customdata[6]:.4f}<extra></extra>")))
    kmin, kmax = float(ch.k.min()), float(ch.k.max())
    pad = 0.25 * (kmax - kmin)
    for model, name, color, dash in ((res.svi, "SVI fit", CYAN, "solid"), (res.baseline, "baseline (linear w)", VIOLET, "dot")):
        if model is None or (name.startswith("baseline") and not show_baseline):
            continue
        s = next((s for s in model.slices if s.expiry == expiry), None)
        if s is None:
            continue
        lo, hi = model.k_range(s)
        kk = np.linspace(lo, hi, 200)
        fig.add_trace(go.Scatter(x=x_transform(kk, F, x_mode), y=np.sqrt(np.maximum(model.slice_w(s, kk, False), 0) / T) * 100,
                                 mode="lines", name=name, line=dict(color=color, width=1.4, dash=dash),
                                 hovertemplate="model IV %{y:.2f}%<extra>" + name + "</extra>"))
        if extrapolate:
            for a, b in ((lo - pad, lo), (hi, hi + pad)):
                ke = np.linspace(a, b, 40)
                fig.add_trace(go.Scatter(x=x_transform(ke, F, x_mode), y=np.sqrt(np.maximum(model.slice_w(s, ke, True), 0) / T) * 100,
                                         mode="lines", showlegend=False, line=dict(color=color, width=1.4, dash="dash"),
                                         opacity=0.6, hovertemplate="EXTRAPOLATED %{y:.2f}%<extra>" + name + "</extra>"))
    fit = next((f for f in res.fits if f.expiry == expiry), None)
    status = f"SVI {fit.status}" if fit else "no SVI"
    base_layout(fig, height=height, title=f"Smile · {expiry:%Y-%m-%d} · {T * 365:.0f}d · F={F:.3f} · {status}")
    fig.update_xaxes(**_axis(X_LABEL[x_mode]))
    fig.update_yaxes(**_axis("implied vol (%) · bars = bid/ask IV"))
    if extrapolate:
        for x0, x1 in ((kmin - pad, kmin), (kmax, kmax + pad)):
            fig.add_vrect(x0=x_transform(x0, F, x_mode), x1=x_transform(x1, F, x_mode), fillcolor="rgba(139,153,176,0.07)",
                          line_width=0, annotation_text="extrapolated", annotation_font=dict(color=MUTED, size=10))
    return fig


def term_figure(res, selected=None, height=380):
    fig = go.Figure()
    for model, name, color, dash in ((res.svi, "SVI ATM (k=0)", CYAN, "solid"), (res.baseline, "baseline ATM (k=0)", VIOLET, "dot")):
        if model is None or not model.slices:
            continue
        T = model.T
        atm = np.array([float(np.sqrt(max(model.slice_w(s, 0.0, False), 0) / s.T)) if np.isfinite(model.slice_w(s, 0.0, False)) else np.nan
                        for s in model.slices]) * 100
        fig.add_trace(go.Scatter(x=T * 365, y=atm, mode="lines+markers", name=name, line=dict(color=color, width=1.3, dash=dash),
                                 marker=dict(size=6), hovertemplate="%{x:.0f}d  ATM IV %{y:.2f}%<extra>" + name + "</extra>"))
    fails = [f for f in res.fits if not f.usable]
    if fails:
        fig.add_trace(go.Scatter(x=[f.T * 365 for f in fails], y=[None] * len(fails), mode="markers", name="failed SVI slice",
                                 marker=dict(color=RED, symbol="x", size=9)))
        for f in fails:
            fig.add_vline(x=f.T * 365, line=dict(color=RED, width=1, dash="dot"))
    if selected is not None:
        T = float(res.chain.loc[res.chain.expiry == selected, "T"].iloc[0])
        fig.add_vline(x=T * 365, line=dict(color=AMBER, width=1.2, dash="dash"),
                      annotation_text="selected", annotation_font=dict(color=AMBER, size=10))
    base_layout(fig, height=height, title="ATM term structure · forward ATM (k = 0)")
    fig.update_xaxes(**_axis("days to expiry (ACT/365)"))
    fig.update_yaxes(**_axis("ATM implied vol (%)"))
    return fig


def residual_figure(recon: pd.DataFrame, height=360):
    r = recon[np.isfinite(recon.iv_model)]
    fig = go.Figure()
    for used, name, op in ((True, "fit set", 1.0), (False, "not fitted (reconstruction only)", 0.45)):
        rr = r[r.used_in_fit == used]
        if rr.empty:
            continue
        fig.add_trace(go.Scatter(
            x=rr.k, y=rr.iv_err_volpts, mode="markers", name=name, opacity=op,
            marker=dict(size=6, color=rr["T"] * 365, colorscale=COLORSCALES["Phosphor"], showscale=used,
                        colorbar=dict(title=dict(text="days", font=dict(color=MUTED, size=10)), thickness=10,
                                      tickfont=dict(color=MUTED, size=10)) if used else None,
                        symbol="circle" if used else "circle-open"),
            error_y=dict(type="data", array=rr.iv_halfwidth * 100, color="rgba(139,153,176,0.35)", thickness=1, width=0),
            customdata=np.c_[rr["T"] * 365, rr.strike, rr.in_band.astype(float)],
            hovertemplate="%{customdata[0]:.0f}d K %{customdata[1]:.2f}<br>model − mid %{y:.3f} vol pts"
                          "<br>in bid/ask band %{customdata[2]:.0f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=AXIS, width=1))
    base_layout(fig, height=height, title="IV reconstruction error (model − mid) · bars = ± IV half-spread")
    fig.update_xaxes(**_axis("log-moneyness k"))
    fig.update_yaxes(**_axis("error (vol points)"))
    return fig


def per_expiry_error_figure(per_exp: pd.DataFrame, height=300):
    d = per_exp.dropna(subset=["iv_rmse_volpts"])
    fig = go.Figure()
    x = [f"{e:%Y-%m-%d}" for e in d.expiry]
    fig.add_trace(go.Bar(x=x, y=d.iv_rmse_volpts, name="IV RMSE", marker_color=TEAL))
    fig.add_trace(go.Bar(x=x, y=d.iv_max_abs_volpts, name="max |IV error|", marker_color=AMBER))
    base_layout(fig, height=height, title="Per-expiry IV error (vol points, fit set)", barmode="group")
    fig.update_xaxes(**_axis("expiry"))
    fig.update_yaxes(**_axis("vol points"))
    return fig


def spread_figure(chain: pd.DataFrame, height=340):
    c = chain[np.isfinite(chain.iv_bid) & np.isfinite(chain.iv_ask)]
    fig = go.Figure(go.Scatter(
        x=c.k, y=(c.iv_ask - c.iv_bid) * 100, mode="markers",
        marker=dict(size=6, color=c["T"] * 365, colorscale=COLORSCALES["Phosphor"], showscale=True,
                    colorbar=dict(title=dict(text="days", font=dict(color=MUTED, size=10)), thickness=10,
                                  tickfont=dict(color=MUTED, size=10))),
        customdata=np.c_[c.strike, c.option_type, c.rel_spread * 100],
        hovertemplate="K %{customdata[0]:.2f} %{customdata[1]}<br>IV spread %{y:.2f} vol pts<br>rel. price spread %{customdata[2]:.1f}%<extra></extra>"))
    base_layout(fig, height=height, title="Bid/ask spread in IV terms (quotes with both IVs solvable)")
    fig.update_xaxes(**_axis("log-moneyness k"))
    fig.update_yaxes(**_axis("IV ask − IV bid (vol points)"), type="log")
    return fig


def rejection_figure(rejected: pd.DataFrame, height=300):
    reasons = rejected.reasons.str.split("; ").explode().str.replace(r"[\d\.\-:+T']+", "", regex=True).str.split("(").str[0].str.strip()
    vc = reasons.value_counts().head(14)[::-1]
    fig = go.Figure(go.Bar(x=vc.values, y=vc.index, orientation="h", marker_color=AMBER))
    base_layout(fig, height=height, title="Rejection reasons (row may have several)")
    fig.update_xaxes(**_axis("rows"))
    fig.update_yaxes(**_axis(""), automargin=True)
    return fig


def calendar_figure(model: SurfaceModel, calendar_grid, height=360):
    if calendar_grid is None:
        return None
    kg, dW = calendar_grid
    labels = [f"{a.T * 365:.0f}d→{b.T * 365:.0f}d" for a, b in zip(model.slices[:-1], model.slices[1:])]
    m = np.nanmax(np.abs(dW)) if np.isfinite(dW).any() else 1
    fig = go.Figure(go.Heatmap(
        x=kg, y=labels, z=dW, zmid=0, zmin=-m, zmax=m,
        colorscale=[[0, RED], [0.5, "#020604"], [0.501, "#06301b"], [1, TEAL]],
        colorbar=dict(title=dict(text="Δw", font=dict(color=MUTED, size=10)), thickness=10, tickfont=dict(color=MUTED, size=10)),
        hovertemplate="k %{x:.3f}<br>%{y}<br>Δw %{z:.6f}<extra></extra>"))
    base_layout(fig, height=height, title="Calendar Δw = w(T₂) − w(T₁) · red < 0 = violation")
    fig.update_xaxes(**_axis("log-moneyness k · blank = unobserved in either slice"))
    fig.update_yaxes(**_axis(""), automargin=True)
    return fig


def density_figure(dens: pd.DataFrame, expiry, height=340):
    fig = go.Figure()
    neg = dens.density.where(dens.density < 0)
    fig.add_trace(go.Scatter(x=dens.strike, y=dens.density, mode="lines", name="q(K) = e^{rT} ∂²C/∂K²",
                             line=dict(color=TEAL, width=1.3), fill="tozeroy", fillcolor="rgba(57,239,139,0.08)"))
    if neg.notna().any():
        fig.add_trace(go.Scatter(x=dens.strike, y=neg, mode="lines", name="negative density (violation)",
                                 line=dict(color=RED, width=3)))
    base_layout(fig, height=height, title=f"Discrete risk-neutral density · {expiry:%Y-%m-%d} (model calls, observed k range)")
    fig.update_xaxes(**_axis("strike K"))
    fig.update_yaxes(**_axis("density per unit strike"))
    return fig


def pnl_heatmap(pnl, spot_shocks, vol_shocks, title, currency, height=420):
    m = np.nanmax(np.abs(pnl)) if np.isfinite(pnl).any() else 1
    text = np.where(np.isfinite(pnl), "", "N/A")
    fig = go.Figure(go.Heatmap(
        x=np.asarray(spot_shocks) * 100, y=vol_shocks, z=pnl, zmid=0, zmin=-m, zmax=m, text=text,
        texttemplate="%{text}", textfont=dict(color=MUTED, size=9),
        colorscale=[[0, "#ff5b60"], [0.35, "#4a1416"], [0.5, "#020604"], [0.65, "#0a3d22"], [1, "#39ef8b"]],
        colorbar=dict(title=dict(text=f"P&L ({currency})", font=dict(color=MUTED, size=10)), thickness=10,
                      tickfont=dict(color=MUTED, size=10)),
        hovertemplate="spot %{x:+.1f}%<br>vol %{y:+.1f} pts<br>P&L %{z:,.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", marker=dict(symbol="cross-thin-open", size=14, color=AMBER, line=dict(width=2)),
                             name="unshocked", hoverinfo="skip"))
    base_layout(fig, height=height, title=title)
    fig.update_xaxes(**_axis("spot shock (%)"))
    fig.update_yaxes(**_axis("parallel IV shock (vol points)"))
    return fig


def pnl_surface(pnl, spot_shocks, vol_shocks, currency, uirevision="pnl", height=460):
    fig = go.Figure(go.Surface(x=np.asarray(spot_shocks) * 100, y=vol_shocks, z=pnl, colorscale=COLORSCALES["Phosphor"],
                               contours=dict(z=dict(show=True, usecolormap=True, width=1)),
                               colorbar=dict(title=dict(text=f"P&L ({currency})", font=dict(color=MUTED, size=10)), thickness=10,
                                             tickfont=dict(color=MUTED, size=10)),
                               hovertemplate="spot %{x:+.1f}%<br>vol %{y:+.1f}<br>P&L %{z:,.2f}<extra></extra>"))
    ax = lambda t: dict(title=dict(text=t, font=dict(color=MUTED, size=11)), backgroundcolor=PANEL, gridcolor=GRID,
                        showbackground=True, tickfont=dict(color=MUTED, size=10))
    base_layout(fig, height=height)
    fig.update_layout(meta="vs3d:pnl", uirevision=uirevision, margin=dict(l=0, r=0, t=8, b=0),
                      scene=dict(xaxis=ax("spot shock (%)"), yaxis=ax("IV shock (vol pts)"), zaxis=ax(f"P&L ({currency})"),
                                 bgcolor=BG, aspectratio=dict(x=1.3, y=1.3, z=0.7)))
    return fig


def standalone_html(fig, title):
    """Self-contained HTML with the plotly.js bundled from the installed Python package (offline)."""
    f = go.Figure(fig)
    f.update_layout(paper_bgcolor=BG, title=dict(text=title, font=dict(color=TEXT, size=14), x=0.01))
    return f.to_html(include_plotlyjs=True, full_html=True, config={"displaylogo": False,
                                                                    "toImageButtonOptions": {"format": "png", "scale": 3}})
