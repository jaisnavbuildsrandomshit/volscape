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
# markers need a scale whose low end is still visible on the near-black background
DOTS = [[0, "#2a7a50"], [0.5, "#39ef8b"], [1, "#d9ffe9"]]

X_LABEL = {"k": "log-moneyness k = ln(K/F)", "moneyness": "moneyness K/F", "strike": "strike K"}


def _axis(title, **kw):
    return dict(title=dict(text=title, font=dict(color=MUTED, size=12)), gridcolor=GRID, zerolinecolor=AXIS,
                linecolor=AXIS, tickfont=dict(color=MUTED, size=11), **kw)


def base_layout(fig, height=360, title=None, **kw):
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=TEXT, size=11), margin=dict(l=56, r=18, t=44 if title else 18, b=96),
        # title pinned to the top of the figure; legend sits on its own row between title and plot
        title=dict(text=title, font=dict(size=12, color=TEXT), x=0.0, xanchor="left", xref="paper",
                   y=0.985, yref="container", yanchor="top") if title else None,
        # legend under the x-axis title: wraps freely at narrow widths without covering the title/plot
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED, size=10), orientation="h",
                    yanchor="top", y=-0.2, xanchor="left", x=0.0),
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
                   highlight_expiry=None, show_slices=True, uirevision="cam", height=640):
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
    if show_slices:
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
        legend=dict(x=0.0, y=0.0, xanchor="left", yanchor="bottom", orientation="h", bgcolor="rgba(0,0,0,0)"),
        scene=dict(xaxis=sc_axis(X_LABEL[x_mode]), yaxis=sc_axis("days to expiry (ACT/365)"), zaxis=sc_axis(ztitle),
                   aspectmode="manual", aspectratio=dict(x=1.45, y=1.45, z=0.8), bgcolor=BG,
                   camera=dict(eye=dict(x=-1.3, y=-1.35, z=0.72), center=dict(x=0, y=0, z=-0.1)),
                   uirevision=uirevision))
    return fig


def smile_figure(res, expiry, x_mode="k", show_baseline=True, extrapolate=False, height=430):
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


def term_figure(res, selected=None, height=430):
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
            marker=dict(size=6, color=rr["T"] * 365, colorscale=DOTS, showscale=used,
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
        marker=dict(size=6, color=c["T"] * 365, colorscale=DOTS, showscale=True,
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


def pnl_surface(pnl, spot_shocks, vol_shocks, currency, uirevision="pnl", height=500):
    m = np.nanmax(np.abs(pnl)) if np.isfinite(pnl).any() else 1
    fig = go.Figure(go.Surface(
        x=np.asarray(spot_shocks) * 100, y=vol_shocks, z=pnl, cmin=-m, cmax=m,
        colorscale=[[0, RED], [0.5, "#06180e"], [1, TEAL]],
        contours=dict(z=dict(show=True, usecolormap=True, width=1, project=dict(z=True)),
                      x=dict(show=False), y=dict(show=False)),
        lighting=dict(ambient=0.6, diffuse=0.6, specular=0.12, roughness=0.7),
        colorbar=dict(title=dict(text=f"P&L ({currency})", font=dict(color=MUTED, size=10)), thickness=10,
                      tickfont=dict(color=MUTED, size=10)),
        hovertemplate="spot %{x:+.1f}%<br>vol %{y:+.1f} pts<br>P&L %{z:,.2f}<extra></extra>"))
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[float(pnl[len(vol_shocks) // 2, len(spot_shocks) // 2])],
                               mode="markers", marker=dict(size=4, color=AMBER), name="unshocked",
                               hovertemplate="unshocked P&L %{z:,.2f}<extra></extra>"))
    fig.update_layout(scene=scene_axes("spot shock (%)", "IV shock (vol pts)", f"P&L ({currency})",
                                       aspectratio=dict(x=1.4, y=1.2, z=0.7)))
    return _layout3d(fig, "vs3d:pnl", "P&L surface · spot × parallel IV shock · NaN = N/A (not evaluable)", height,
                     uirevision=uirevision)


def standalone_html(fig, title):
    """Self-contained HTML with the plotly.js bundled from the installed Python package (offline)."""
    f = go.Figure(fig)
    f.update_layout(paper_bgcolor=BG, title=dict(text=title, font=dict(color=TEXT, size=14), x=0.01))
    return f.to_html(include_plotlyjs=True, full_html=True, config={"displaylogo": False,
                                                                    "toImageButtonOptions": {"format": "png", "scale": 3}})


# ----------------------------------------------------------------------------- live market charts
def live_depth_figure(bids, asks, height=300):
    """Cumulative DISPLAYED depth (top levels only) as step lines: not a full supply/demand curve."""
    fig = go.Figure()
    if bids:
        px_b = [p for p, _ in bids]
        cum_b = np.cumsum([q for _, q in bids])
        fig.add_trace(go.Scatter(x=px_b, y=cum_b, mode="lines", line=dict(color=TEAL, width=1.4, shape="hv"),
                                 fill="tozeroy", fillcolor="rgba(57,239,139,0.07)", name="bids (cum)",
                                 hovertemplate="%{x:,.6g}<br>cum qty %{y:,.6g}<extra>bids</extra>"))
    if asks:
        px_a = [p for p, _ in asks]
        cum_a = np.cumsum([q for _, q in asks])
        fig.add_trace(go.Scatter(x=px_a, y=cum_a, mode="lines", line=dict(color=RED, width=1.4, shape="hv"),
                                 fill="tozeroy", fillcolor="rgba(255,91,96,0.06)", name="asks (cum)",
                                 hovertemplate="%{x:,.6g}<br>cum qty %{y:,.6g}<extra>asks</extra>"))
    base_layout(fig, height=height, title="Displayed depth · cum qty")
    fig.update_xaxes(**_axis("price"))
    fig.update_yaxes(**_axis("cumulative base qty"))
    return fig


def live_price_figure(prices, vwap=None, height=300):
    fig = go.Figure()
    n = len(prices)
    x = np.arange(-n + 1, 1) if n else []
    fig.add_trace(go.Scatter(x=x, y=prices, mode="lines", line=dict(color=TEXT, width=1.2), name="last trade (1s close)",
                             hovertemplate="t%{x}s<br>%{y:,.6g}<extra></extra>"))
    if vwap is not None and n:
        fig.add_trace(go.Scatter(x=[x[0], x[-1]], y=[vwap, vwap], mode="lines", name="session VWAP",
                                 line=dict(color=AMBER, width=1, dash="dot"), hoverinfo="skip"))
    base_layout(fig, height=height, title=f"Price · last {n}s · 1s closes")
    fig.update_xaxes(**_axis("seconds from now"))
    fig.update_yaxes(**_axis("price"))
    return fig


def live_flow_figure(flows, height=240):
    n = len(flows)
    x = np.arange(-n + 1, 1)
    fig = go.Figure(go.Bar(x=x, y=flows, marker_color=[TEAL if v >= 0 else RED for v in flows], marker_line_width=0,
                           hovertemplate="t%{x}s<br>net %{y:,.0f}<extra></extra>"))
    base_layout(fig, height=height, title="Signed flow / s · buy − sell aggressor notional")
    fig.update_xaxes(**_axis("seconds from now"))
    fig.update_yaxes(**_axis("quote notional"))
    fig.update_layout(bargap=0.15)
    return fig


# ----------------------------------------------------------------------------- shared 3D scene style
def scene_axes(xt, yt, zt, **kw):
    def ax(t):
        return dict(title=dict(text=t, font=dict(color=MUTED, size=11)), backgroundcolor="#030a06",
                    gridcolor="#0f2c1c", zerolinecolor="#1f5a3a", showbackground=True, showspikes=False,
                    tickfont=dict(color=MUTED, size=10), color=MUTED)
    return dict(xaxis=ax(xt), yaxis=ax(yt), zaxis=ax(zt), bgcolor=BG, aspectmode="manual",
                aspectratio=kw.pop("aspectratio", dict(x=1.35, y=1.35, z=0.7)),
                camera=kw.pop("camera", dict(eye=dict(x=-1.45, y=-1.45, z=0.8), center=dict(x=0, y=0, z=-0.12))), **kw)


def _layout3d(fig, meta, title, height, uirevision="3d"):
    base_layout(fig, height=height, title=title)
    fig.update_layout(meta=meta, uirevision=uirevision, margin=dict(l=0, r=0, t=44 if title else 8, b=0),
                      legend=dict(x=0.0, y=0.0, xanchor="left", yanchor="bottom", orientation="h",
                                  bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED, size=10)))
    return fig


# ----------------------------------------------------------------------------- 3D analytics
def density_waterfall(dens_by_expiry: dict, chain: pd.DataFrame, selected=None, height=480):
    """Risk-neutral density per expiry as vertical curtains in (k, T) space. Each curtain is the discrete
    density over that slice's own observed strike range: nothing between expiries is invented."""
    fig = go.Figure()
    T_of = chain.groupby("expiry")["T"].first()
    for exp, d in sorted(dens_by_expiry.items(), key=lambda kv: T_of.get(kv[0], 0)):
        days = float(T_of.get(exp, np.nan)) * 365
        k = d.k.to_numpy()
        q_k = (d.density * d.strike).to_numpy()        # density per unit log-moneyness: q_K * K
        sel = selected is not None and exp == selected
        neg = q_k < 0
        fig.add_trace(go.Surface(
            x=np.vstack([k, k]), y=np.full((2, len(k)), days), z=np.vstack([np.zeros_like(q_k), q_k]),
            surfacecolor=np.vstack([q_k, q_k]), colorscale=[[0, "#0a3d22"], [1, CYAN if sel else TEAL]],
            showscale=False, opacity=0.9 if sel else 0.55, name=f"{exp:%Y-%m-%d}",
            hovertemplate=f"{exp:%Y-%m-%d} · {days:.0f}d<br>k %{{x:.3f}}<br>q(k) %{{z:.4f}}<extra></extra>",
            contours=dict(z=dict(show=False))))
        fig.add_trace(go.Scatter3d(x=k, y=np.full(len(k), days), z=q_k, mode="lines", showlegend=False,
                                   line=dict(color=CYAN if sel else "#8dffc0", width=4 if sel else 2), hoverinfo="skip"))
        if neg.any():
            fig.add_trace(go.Scatter3d(x=k[neg], y=np.full(int(neg.sum()), days), z=q_k[neg], mode="markers",
                                       marker=dict(color=RED, size=3), name="negative density", showlegend=False,
                                       hovertemplate="NEGATIVE density<br>k %{x:.3f}<br>%{z:.4f}<extra></extra>"))
    fig.update_layout(scene=scene_axes("log-moneyness k", "days to expiry", "density per unit k"))
    return _layout3d(fig, "vs3d:density",
                     "Risk-neutral density waterfall · q(k) = K·e^{rT}·∂²C/∂K² · each expiry over its observed range", height)


def calendar_surface(model, calendar_grid, height=440):
    if calendar_grid is None:
        return None
    kg, dW = calendar_grid
    mids = np.array([(a.T + b.T) / 2 * 365 for a, b in zip(model.slices[:-1], model.slices[1:])])
    m = np.nanmax(np.abs(dW)) if np.isfinite(dW).any() else 1
    fig = go.Figure(go.Surface(
        x=kg, y=mids, z=dW, cmin=-m, cmax=m, colorscale=[[0, RED], [0.5, "#06180e"], [1, TEAL]],
        colorbar=dict(title=dict(text="Δw", font=dict(color=MUTED, size=10)), thickness=10,
                      tickfont=dict(color=MUTED, size=10)),
        contours=dict(z=dict(show=True, start=0, end=0, size=1, color=AMBER, width=4)),
        hovertemplate="k %{x:.3f}<br>pair mid %{y:.0f}d<br>Δw %{z:.6f}<extra></extra>"))
    fig.update_layout(scene=scene_axes("log-moneyness k", "expiry pair (mid, days)", "Δw = w(T₂) − w(T₁)"))
    return _layout3d(fig, "vs3d:calendar", "Calendar Δw surface · amber contour = 0 · below 0 = violation", height)


def residual_stems(recon: pd.DataFrame, height=460):
    """Each quote's IV error as a stem from the zero plane at (k, T)."""
    r = recon[np.isfinite(recon.iv_model)]
    fig = go.Figure()
    xs, ys, zs = [], [], []
    for k, t, e in zip(r.k, r["T"] * 365, r.iv_err_volpts):
        xs += [k, k, None]
        ys += [t, t, None]
        zs += [0, e, None]
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color="rgba(141,255,192,0.35)", width=2),
                               hoverinfo="skip", showlegend=False))
    for used, name, sym in ((True, "fit set", "circle"), (False, "not fitted", "circle-open")):
        rr = r[r.used_in_fit == used]
        if rr.empty:
            continue
        fig.add_trace(go.Scatter3d(
            x=rr.k, y=rr["T"] * 365, z=rr.iv_err_volpts, mode="markers", name=name,
            marker=dict(size=3.2, symbol=sym, color=np.where(rr.iv_err_volpts >= 0, TEAL, RED)),
            customdata=np.c_[rr.strike, rr.in_band.astype(float)],
            hovertemplate="K %{customdata[0]:.2f}<br>%{y:.0f}d<br>model − mid %{z:.3f} vp"
                          "<br>in band %{customdata[1]:.0f}<extra></extra>"))
    kk = [float(r.k.min()), float(r.k.max())]
    tt = [float(r["T"].min() * 365), float(r["T"].max() * 365)]
    fig.add_trace(go.Surface(x=kk, y=tt, z=[[0, 0], [0, 0]], showscale=False, opacity=0.18, hoverinfo="skip",
                             colorscale=[[0, "#1f5a3a"], [1, "#1f5a3a"]]))
    fig.update_layout(scene=scene_axes("log-moneyness k", "days to expiry", "IV error (vol pts)"))
    return _layout3d(fig, "vs3d:resid", "IV reconstruction error stems · model − mid at (k, T)", height)


def book_landscape(history: list, bins_bps: float | None = None, half_width_bps: float | None = None,
                   n_bins: int = 40, height=460):
    """Displayed order-book depth over time as terrain: x = distance from the then-current mid in bps,
    y = seconds ago, z = displayed quantity in that bin. Top-20 levels only (visible resting liquidity,
    not total supply/demand). Seconds without a book sample stay empty."""
    if len(history) < 2:
        return None
    if half_width_bps is None:
        # adapt the window to the displayed book's actual reach (99th pct of level distance from mid)
        offs = [abs(p / mid - 1) * 1e4 for _, mid, bids, asks in history for p, _ in list(bids) + list(asks)]
        half_width_bps = float(np.percentile(offs, 99)) if offs else 1.0
        half_width_bps = max(half_width_bps, 1e-3)
    if bins_bps is None:
        bins_bps = 2 * half_width_bps / n_bins
    edges = np.linspace(-half_width_bps, half_width_bps, int(round(2 * half_width_bps / bins_bps)) + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    now = history[-1][0]
    secs = np.arange(history[0][0], now + 1)
    Z = np.full((len(secs), len(centers)), np.nan)
    by_sec = {h[0]: h for h in history}
    for i, sec in enumerate(secs):
        h = by_sec.get(sec)
        if h is None:
            continue
        _, mid, bids, asks = h
        row = np.zeros(len(centers))
        for p, q in list(bids) + list(asks):
            off = (p / mid - 1) * 1e4
            j = np.searchsorted(edges, off, side="right") - 1
            if 0 <= j < len(centers):
                row[j] += q
        Z[i] = row
    side = np.where(centers < 0, -1.0, 1.0)[None, :] * np.ones_like(Z)
    fig = go.Figure(go.Surface(
        x=centers, y=secs - now, z=Z, surfacecolor=side * np.sqrt(np.nan_to_num(Z)),
        colorscale=[[0, TEAL], [0.5, "#04170c"], [1, RED]], showscale=False,
        contours=dict(z=dict(show=False)), lighting=dict(ambient=0.6, diffuse=0.6, specular=0.15, roughness=0.6),
        hovertemplate="%{x:+.1f} bps from mid<br>t%{y}s<br>qty %{z:.4g}<extra></extra>"))
    fig.update_layout(scene=scene_axes("bps from mid (− bids · + asks)", "seconds ago", "displayed qty",
                                       aspectratio=dict(x=1.4, y=1.5, z=0.55)))
    return _layout3d(fig, "vs3d:book",
                     f"Order-book landscape · last {len(secs)}s · displayed top-20 depth · ±{half_width_bps:.3g} bp · bin {bins_bps:.3g} bp", height)
