"""Plotly figures for the Wave Climate Trend-Extreme Tool. One visual
language for every page: white background, muted grid, one colour per
record, scenario colours fixed."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import genpareto

import analysis as A

PALETTE = ["#0b5394", "#c0392b", "#1a7f5a", "#b7791f", "#6b4c9a", "#2a9d8f", "#d16d9e", "#555555"]
FIXED = {"RCP4.5": "#0b5394", "RCP8.5": "#c0392b", "Historical 2005": "#444444", "Brisbane 2024": "#1a7f5a",
         "reference": "#222222", "raw model": "#999999", "delta change": "#b7791f", "QDM": "#0b5394"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
LABEL = {"swh": "H<sub>s</sub> [m]", "tm": "T<sub>m</sub> [s]", "tp": "T<sub>p</sub> [s]"}


def color_for(name, i=0):
    for key, c in FIXED.items():
        if name.startswith(key):
            return c
    return PALETTE[i % len(PALETTE)]


def _layout(fig, height=380, title=None, xtitle=None, ytitle=None, legend_top=True):
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=50, r=20, t=50 if title else 25, b=80 if legend_top else 45),
        title=dict(text=title, x=0.01, font=dict(size=15)) if title else None,
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0, font=dict(size=11)) if legend_top
        else dict(font=dict(size=11)),
        font=dict(family="Inter, Segoe UI, Helvetica, Arial, sans-serif", size=12, color="#1f2933"),
        hovermode="x unified",
    )
    fig.update_xaxes(title=xtitle, gridcolor="#eef1f4", zeroline=False)
    fig.update_yaxes(title=ytitle, gridcolor="#eef1f4", zeroline=False)
    return fig


def _rgba(hex_color, a):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


def _display_series(s, max_points=60000):
    """Long records are shown as daily mean with a daily min-max band."""
    if len(s) <= max_points:
        return s, None, None
    d = s.resample("D")
    return d.mean().dropna(), d.min().dropna(), d.max().dropna()


def record(s, name, variable, color=None, height=320):
    color = color or color_for(name)
    m, lo, hi = _display_series(s)
    fig = go.Figure()
    if lo is not None:
        fig.add_trace(go.Scatter(x=hi.index, y=hi.values, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=lo.index, y=lo.values, fill="tonexty", mode="lines", fillcolor=_rgba(color, 0.18), line=dict(width=0),
                                 name="daily range", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=m.index, y=m.values, line=dict(color=color, width=1), name="daily mean"))
    else:
        fig.add_trace(go.Scatter(x=m.index, y=m.values, line=dict(color=color, width=0.8), name=name))
    return _layout(fig, height, None, None, LABEL.get(variable, variable))


def seasonal(records, variable, height=340):
    fig = go.Figure()
    for i, (name, s) in enumerate(records.items()):
        mc = A.monthly_climatology(s)
        fig.add_trace(go.Scatter(x=MONTHS, y=mc.values, mode="lines+markers", name=name,
                                 line=dict(color=color_for(name, i), width=2), marker=dict(size=7)))
    return _layout(fig, height, "Seasonal cycle (monthly means)", None, LABEL.get(variable, variable))


def annual_means(trends, variable, height=460):
    """trends: dict name -> {"trend": analysis.trend_analysis result or None, "annual": Series}."""
    fig = go.Figure()
    for i, (name, item) in enumerate(trends.items()):
        c = color_for(name, i)
        tr, am = item["trend"], item["annual"]
        fig.add_trace(go.Scatter(x=am.index, y=am.values, mode="lines+markers", name=name, legendgroup=name,
                                 line=dict(color=c, width=1.2), marker=dict(size=6)))
        if tr is not None:
            o = tr["ols"]
            per_dec = o["slope"] * (1000 if variable == "swh" else 10)
            unit = "cm/decade" if variable == "swh" else "s/decade"
            fig.add_trace(go.Scatter(x=am.index, y=o["intercept"] + o["slope"] * am.index, mode="lines", legendgroup=name, showlegend=False,
                                     name=f"{name} OLS trend", line=dict(color=c, width=2.5, dash="dash"),
                                     hovertemplate=f"{name} OLS trend: {per_dec:+.3f} {unit}, p = {o['p_value']:.2f}<extra></extra>"))
    fig = _layout(fig, height, "Annual means (markers) and OLS trends (dashed)", "Year", f"Annual mean {LABEL.get(variable, variable)}", legend_top=False)
    fig.update_layout(legend=dict(orientation="v", x=1.01, y=1, xanchor="left", font=dict(size=11)), margin=dict(r=10))
    return fig


def peaks(s, res, name, variable, height=360):
    c = color_for(name)
    m, lo, hi = _display_series(s)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=m.index, y=(hi if hi is not None else m).values, line=dict(color="#c7cdd4", width=0.8),
                             name="record (daily max)" if hi is not None else "record"))
    fig.add_trace(go.Scatter(x=res["peaks"].index, y=res["peaks"].values, mode="markers", name=f"storm peaks (n = {res['n_peaks']})",
                             marker=dict(color=c, size=6, line=dict(color="white", width=0.5))))
    fig.add_hline(y=res["threshold"], line=dict(color="#222222", dash="dash", width=1.2),
                  annotation_text=f"threshold {res['threshold']:.2f} ({res['percentile']:g}th percentile)", annotation_position="top left")
    return _layout(fig, height, "Declustered storm peaks", None, LABEL.get(variable, variable))


def gpd_fit(res, name, variable, height=340):
    c = color_for(name)
    y = res["exceedances"]
    xx = np.linspace(0, y.max() * 1.05, 200)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=y, histnorm="probability density", nbinsx=25, name="storm-peak exceedances",
                               marker=dict(color=_rgba(c, 0.45), line=dict(color=c, width=0.5))))
    fig.add_trace(go.Scatter(x=xx, y=genpareto.pdf(xx, res["xi"], 0, res["beta"]), name=f"GPD ξ = {res['xi']:+.3f}, β = {res['beta']:.2f}",
                             line=dict(color="#222222", width=2)))
    fig.update_layout(bargap=0.05)
    return _layout(fig, height, "GPD fit to exceedances", f"Exceedance over threshold [{'m' if variable == 'swh' else 's'}]", "Density")


def qq(res, name, variable, height=340):
    c = color_for(name)
    fq, eq = A.qq_points(res)
    lim = max(fq.max(), eq.max()) * 1.05
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines", line=dict(color="#222222", dash="dash", width=1), name="1:1"))
    fig.add_trace(go.Scatter(x=fq, y=eq, mode="markers", marker=dict(color=c, size=6), name="exceedances"))
    unit = "m" if variable == "swh" else "s"
    return _layout(fig, height, "Quantile-quantile plot", f"Fitted GPD quantile [{unit}]", f"Empirical quantile [{unit}]")


def return_levels(results, variable, height=420, ts_table=None):
    """results: dict name -> pot result. Fitted curve, bootstrap band, empirical points."""
    fig = go.Figure()
    for i, (name, res) in enumerate(results.items()):
        c = color_for(name, i)
        T = np.logspace(np.log10(1.0 / res["rate_per_year"] * 1.05), np.log10(max(res["return_periods"].max(), 200)), 200)
        curve = A.return_level(res["threshold"], res["xi"], res["beta"], res["rate_per_year"], T)
        rp = res["return_periods"]
        fig.add_trace(go.Scatter(x=np.concatenate([rp, rp[::-1]]), y=np.concatenate([res["rl_ci"][1], res["rl_ci"][0][::-1]]),
                                 fill="toself", mode="lines", fillcolor=_rgba(c, 0.15), line=dict(width=0), name=f"{name} bootstrap 95%", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=T, y=curve, mode="lines", name=f"{name} fitted", line=dict(color=c, width=2.5)))
        ex, eT = res["empirical"]
        fig.add_trace(go.Scatter(x=eT, y=ex, mode="markers", name=f"{name} storm peaks", marker=dict(color=c, size=5, opacity=0.7)))
    fig = _layout(fig, height, "Return levels", "Return period [years]", LABEL.get(variable, variable))
    fig.update_xaxes(type="log", tickvals=[0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500], ticktext=["0.5", "1", "2", "5", "10", "20", "50", "100", "200", "500"])
    fig.update_layout(hovermode="closest")
    return fig


def diagnostics(mrl, ps, threshold, variable, height=320):
    unit = "m" if variable == "swh" else "s"
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Mean residual life", "Shape ξ", "Modified scale β − ξu"))
    fig.add_trace(go.Scatter(x=mrl["threshold"], y=mrl["mean_excess"], mode="lines+markers", name="mean excess",
                             error_y=dict(type="data", array=1.96 * mrl["se"], visible=True, color="#999999"),
                             line=dict(color="#333333"), marker=dict(size=4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=ps["threshold"], y=ps["xi"], mode="lines+markers", name="shape ξ", line=dict(color="#0b5394"), marker=dict(size=4)), row=1, col=2)
    fig.add_trace(go.Scatter(x=ps["threshold"], y=ps["modified_scale"], mode="lines+markers", name="modified scale", line=dict(color="#c0392b"), marker=dict(size=4)), row=1, col=3)
    for col in (1, 2, 3):
        fig.add_vline(x=threshold, line=dict(color="#222222", dash="dot", width=1), row=1, col=col)
        fig.update_xaxes(title=f"Threshold [{unit}]", row=1, col=col, gridcolor="#eef1f4")
    fig.update_yaxes(title=f"Mean excess [{unit}]", row=1, col=1, gridcolor="#eef1f4")
    fig.update_yaxes(title="ξ", row=1, col=2, gridcolor="#eef1f4")
    fig.update_yaxes(title=f"β − ξu [{unit}]", row=1, col=3, gridcolor="#eef1f4")
    fig.update_layout(template="plotly_white", height=height, showlegend=False, margin=dict(l=50, r=20, t=40, b=45),
                      font=dict(family="Inter, Segoe UI, Helvetica, Arial, sans-serif", size=12))
    return fig


def validation_quantiles(series, variable, height=380):
    """Quantiles of each corrected series against the reference quantiles."""
    qs = np.concatenate([np.linspace(0.01, 0.99, 99), [0.995, 0.999]])
    ref = series["reference"].dropna()
    qr = np.quantile(ref.values, qs)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[qr.min(), qr.max()], y=[qr.min(), qr.max()], mode="lines", name="1:1", line=dict(color="#222222", dash="dash", width=1)))
    for name in ("raw model", "delta change", "QDM"):
        if name in series:
            qc = np.quantile(series[name].dropna().values, qs)
            fig.add_trace(go.Scatter(x=qr, y=qc, mode="lines+markers", name=name, line=dict(color=FIXED[name], width=2), marker=dict(size=4)))
    unit = LABEL.get(variable, variable)
    fig = _layout(fig, height, "Validation years: quantiles against the reference", f"Reference quantile {unit}", f"Model quantile {unit}")
    fig.update_layout(hovermode="closest")
    return fig


def validation_seasonal(series, variable, height=340):
    fig = go.Figure()
    for name, s in series.items():
        mc = A.monthly_climatology(s)
        fig.add_trace(go.Scatter(x=MONTHS, y=mc.values, mode="lines+markers", name=name,
                                 line=dict(color=FIXED.get(name, "#888888"), width=2.5 if name == "reference" else 1.8),
                                 marker=dict(size=6)))
    return _layout(fig, height, "Validation years: seasonal cycle", None, LABEL.get(variable, variable))


def projection_compare(records, variable, height=380):
    """Annual means of several projected series (raw, delta, QDM)."""
    fig = go.Figure()
    for i, (name, s) in enumerate(records.items()):
        am = A.annual_means(s)
        fig.add_trace(go.Scatter(x=am.index, y=am.values, mode="lines+markers", name=name, line=dict(color=color_for(name, i), width=1.5), marker=dict(size=5)))
    return _layout(fig, height, "Projected annual means by method", "Year", f"Annual mean {LABEL.get(variable, variable)}")


def ts_eva(ts, stationary, name, variable, T=100, height=400):
    t = ts["table"]
    c = color_for(name)
    col = f"RL_{T}"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.concatenate([t.index, t.index[::-1]]), y=np.concatenate([t[f"{col}_hi"], t[f"{col}_lo"][::-1]]),
                             fill="toself", mode="lines", fillcolor=_rgba(c, 0.15), line=dict(width=0), name="bootstrap 95%", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t.index, y=t[col], mode="lines", name=f"non-stationary {T}-year level", line=dict(color=c, width=2.5)))
    fig.add_hline(y=stationary, line=dict(color="#222222", dash="dash", width=1.5),
                  annotation_text=f"stationary {T}-year level {stationary:.2f}", annotation_position="bottom right")
    return _layout(fig, height, f"Time-varying {T}-year return level (transformed-stationary EVA, {ts['window_years']}-year window)",
                   "Year", LABEL.get(variable, variable))


def ts_moments(ts, name, variable, height=300):
    t = ts["table"]
    c = color_for(name)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Running mean μ(t)", "Running standard deviation σ(t)"))
    fig.add_trace(go.Scatter(x=t.index, y=t["mu"], mode="lines", line=dict(color=c, width=2), name="μ"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t.index, y=t["sigma"], mode="lines", line=dict(color=c, width=2), name="σ"), row=1, col=2)
    fig.update_layout(template="plotly_white", height=height, showlegend=False, margin=dict(l=50, r=20, t=40, b=45),
                      font=dict(family="Inter, Segoe UI, Helvetica, Arial, sans-serif", size=12))
    fig.update_xaxes(title="Year", gridcolor="#eef1f4")
    fig.update_yaxes(gridcolor="#eef1f4")
    return fig
