"""
Wave Climate Trend-Extreme Tool

Pick data, pick options, run once, read the report: bias correction, trends,
and stationary and non-stationary return levels of wave climate records and
projections.

Run locally:  streamlit run app.py
"""

import hashlib
import io
import os
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import analysis as A
import charts as C
import fetch as F
import pipeline as PL

st.set_page_config(page_title="Wave Climate Trend-Extreme Tool", page_icon="🌊", layout="wide",
                   initial_sidebar_state="collapsed")

ROLES = ["observed", "reference", "model historical", "model future", "projected"]
VARNAME = {"swh": "wave height", "tm": "wave period"}
SYM = {"swh": "Hs", "tm": "Tm"}
UNIT = {"swh": "m", "tm": "s"}
PAGES = {"setup": (":material/tune:", "Setup"), "results": (":material/insights:", "Results"), "about": (":material/groups:", "Team")}
CORRECTIONS = {"off": "Off", "delta": "Delta change", "qdm": "Quantile delta mapping", "compare": "Compare all"}

# Free photos from Unsplash (Unsplash License). Credits on the Team page.
PHOTOS = {
    "hero": ("https://images.unsplash.com/photo-1559827260-dc66d52bef19?w=1800&q=70&auto=format&fit=crop", "Silas Baisch", "https://unsplash.com/@silasbaisch"),
    "storm": ("https://images.unsplash.com/photo-1524679166686-04237b4667b4?w=1600&q=70&auto=format&fit=crop", "Torsten Dederichs", "https://unsplash.com/@tdederichs"),
    "aerial": ("https://images.unsplash.com/photo-1524275804141-5e9f2bc5a71d?w=1600&q=70&auto=format&fit=crop", "Ivana Cajina", "https://unsplash.com/@von_co"),
    "buoy": ("https://images.unsplash.com/photo-1590524761651-fc4070aa033b?w=1600&q=70&auto=format&fit=crop&crop=entropy", "Emmy C", "https://unsplash.com/@emmy__c"),
    "rocks": ("https://images.unsplash.com/photo-1612083162601-49e34ec39956?w=1600&q=70&auto=format&fit=crop", "Henson Tsai", "https://unsplash.com/@hensontsai"),
    "beach": ("https://images.unsplash.com/photo-1549616014-08d94f85e697?w=1600&q=70&auto=format&fit=crop", "Damon Hall", "https://unsplash.com/@damodigital"),
}

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
  html, body, [class*="css"], .stApp {font-family: Inter, "Segoe UI", Helvetica, Arial, sans-serif;}
  .block-container {padding-top: 4.2rem; padding-bottom: 2.5rem; max-width: 1280px;}
  header[data-testid="stHeader"] {background: #ffffff !important;}
  [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] {display: none;}
  h1, h2, h3 {letter-spacing: -0.01em;}
  @keyframes fadeUp {from {opacity: 0; transform: translateY(18px);} to {opacity: 1; transform: none;}}
  @keyframes floaty {0% {transform: translateY(0);} 50% {transform: translateY(-6px);} 100% {transform: translateY(0);}}
  .fade {animation: fadeUp .7s ease both;}
  .d1 {animation-delay: .10s} .d2 {animation-delay: .20s} .d3 {animation-delay: .30s} .d4 {animation-delay: .40s} .d5 {animation-delay: .50s}
  .st-key-tiles button {height: 120px; border-radius: 18px; border: 1px solid #e3e8ee; background: #fff; font-weight: 600;
      color: #0b5394; transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease; animation: fadeUp .7s ease both;}
  .st-key-tiles button:hover {transform: translateY(-5px); box-shadow: 0 14px 30px rgba(11,83,148,.18); border-color: #0b5394;}
  .st-key-tiles button p {font-size: 1.15rem;}
  .st-key-tiles [data-testid="stCaptionContainer"] {text-align: center;}
  .st-key-actions button {border-radius: 999px; padding: .55rem 1.3rem; font-weight: 600; transition: transform .15s ease;}
  .st-key-actions button:hover {transform: translateY(-2px);}
  .st-key-runbtn button {height: 3.2rem; font-size: 1.05rem; font-weight: 700; border-radius: 14px;}
  div[data-testid="stLayoutWrapper"]:has(> .st-key-topbar) {position: sticky; top: 3.75rem; z-index: 990; background: #ffffff;}
  .st-key-topbar {background: #ffffff; border-bottom: 1px solid #e9eef3; padding: .35rem 0 .45rem 0; margin-bottom: .6rem;}
  .st-key-topbar button {border-radius: 999px;}
  .banner {position: relative; border-radius: 18px; overflow: hidden; height: 170px; margin-bottom: 1rem; background-size: cover; background-position: center;}
  .banner .ov {position:absolute; inset:0; background: linear-gradient(90deg, rgba(5,38,70,.85), rgba(5,38,70,.25));}
  .banner h1 {position:absolute; left: 1.6rem; bottom: 1.0rem; color: #fff; margin: 0; font-size: 2rem; text-shadow: 0 2px 12px rgba(0,0,0,.4);}
  .banner .sub {position:absolute; left: 1.6rem; top: 1.0rem; color: rgba(255,255,255,.9); font-size: .95rem;}
  .banner .credit {position:absolute; right: .8rem; bottom: .45rem; color: rgba(255,255,255,.7); font-size: .58rem; letter-spacing: .01em;}
  .banner .credit a {color: rgba(255,255,255,.9);}
  .panel {border: 1px solid #e3e8ee; border-radius: 16px; padding: .2rem 1.1rem 1rem 1.1rem; background: #fff;}
  .card {border: 1px solid #e3e8ee; border-radius: 16px; padding: 1.1rem 1.2rem; background: #fff; height: 100%; animation: fadeUp .7s ease both;}
  .card h3 {margin: .2rem 0 .1rem 0; font-size: 1.15rem;}
  .card .role {color: #0b5394; font-weight: 600; font-size: .9rem;}
  .card p {margin: .35rem 0 0 0; color: #3e4c59; font-size: .9rem;}
  .card a {margin-right: .6rem; font-size: .88rem;}
  .card p.cr a {margin-right: 0;}
  .avatar {width: 64px; height: 64px; border-radius: 50%; background: linear-gradient(135deg,#0b5394,#1a9a9a); color: #fff; display:flex; align-items:center;
      justify-content:center; font-weight: 700; font-size: 1.35rem; margin-bottom: .5rem; animation: floaty 4s ease-in-out infinite;}
  .kl {display:grid; grid-template-columns: 170px repeat(4, 1fr); gap: .6rem; align-items: stretch; margin-bottom: .6rem; animation: fadeUp .6s ease both;}
  .kl .nm {display:flex; align-items:center; font-weight: 700; color: #1f2933; font-size: 1rem;}
  .kl .nm span {display:inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: .5rem;}
  .kl .c {border-radius: 14px; padding: .65rem .9rem; background: linear-gradient(180deg, #f5f9fd, #ffffff); border: 1px solid #dfe8f1;}
  .kl .c .t {font-size: .78rem; color: #52606d; font-weight: 600; text-transform: uppercase; letter-spacing: .04em;}
  .kl .c .v {font-size: 1.75rem; font-weight: 800; color: #0b5394; line-height: 1.2;}
  .kl .c .v small {font-size: .95rem; font-weight: 600; color: #52606d;}
  .kl .c .ci {font-size: .8rem; color: #52606d;}
  .lead {font-size: 1.05rem; color: #1f2933; background: #f5f9fd; border-left: 4px solid #0b5394; padding: .8rem 1rem; border-radius: 8px; margin: .3rem 0 1rem 0;}
  .sec-n {display:inline-flex; width: 30px; height: 30px; border-radius: 50%; background: #0b5394; color: #fff; align-items:center; justify-content:center;
      font-weight: 700; font-size: .95rem; margin-right: .5rem;}
  .sec-t {font-size: 1.3rem; font-weight: 700; color: #1f2933; vertical-align: middle;}
  .note {font-size: .88rem; color: #7a4b00; background: #fff8e6; border: 1px solid #ffe2a8; border-radius: 10px; padding: .55rem .8rem; margin: .3rem 0;}
  .small {font-size: .85rem; color: #52606d;}
  hr.sep {border: none; border-top: 1px solid #e9eef3; margin: 1.6rem 0 1.2rem 0;}
  footer, #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# State and helpers
# ---------------------------------------------------------------------------

def init_state():
    defaults = {"page": "home", "records": {}, "results": None, "log": [], "opts": dict(PL.DEFAULTS), "targets": None}
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()
S = st.session_state


def go(page):
    S.page = page
    st.rerun()


def add_record(name, df, info, role, source):
    S.records[name] = {"df": df, "info": info, "role": role, "source": source}
    S.log.append(f"{datetime.now():%H:%M} loaded {name} ({role}, {source})")


@st.cache_data(show_spinner=False)
def load_file(path):
    return A.load_series(path)


def load_demo():
    for entry in F.load_catalogue()["stations"]:
        for name, rec in entry["records"].items():
            df, info = load_file(F.resolve_catalogue_file(rec["file"]))
            add_record(name, df, info, rec["role"], entry["label"])


def default_targets():
    long = [n for n, r in S.records.items() if r["info"]["span_years"] >= 10]
    return long or list(S.records)


@st.cache_resource(show_spinner=False)
def results_store():
    return {}


def signature(targets, opts):
    h = hashlib.sha1()
    for n in sorted(S.records):
        r = S.records[n]
        df = r["df"]
        num = df.select_dtypes("number")
        h.update(f"{n}|{r['role']}|{len(df)}|{float(num.sum().sum()):.6f}".encode())
    h.update(repr(sorted(targets)).encode())
    h.update(repr(sorted(opts.items())).encode())
    return h.hexdigest()


def run_analysis(container):
    targets = [t for t in (S.targets or default_targets()) if t in S.records]
    if not targets:
        container.error("Choose at least one record to analyse.")
        return False
    key = signature(targets, S.opts)
    store = results_store()
    if key in store:
        S.results = store[key]
        return True
    with container.status("Running the analysis", expanded=True) as status:
        R = PL.run(S.records, targets, S.opts, progress=lambda m: status.write(m))
        status.update(label=f"Done in {R['seconds']:.0f} s", state="complete", expanded=False)
    if len(store) > 12:
        store.pop(next(iter(store)))
    store[key] = R
    S.results = R
    return True


def f2(x):
    return f"{x:.2f}"


def sg(x):
    """Signed, two decimals, no negative zero."""
    x = round(float(x), 2) + 0.0
    return f"{x:+.2f}" if x != 0 else "0.00"


def table2(df):
    """Show numbers with two decimals."""
    cfg = {c: st.column_config.NumberColumn(format="%.2f") for c in df.columns if pd.api.types.is_float_dtype(df[c])}
    st.dataframe(df, hide_index=True, width="stretch", column_config=cfg)


def RL_COLS(var):
    return {"return_period_years": "return period (yr)", "return_level": f"level [{UNIT[var]}]", "ci_low": "95% low", "ci_high": "95% high"}


def tab_html(df, index=True):
    return df.to_html(index=index, border=0, float_format=lambda x: f"{x:.2f}")


def banner(title, photo, sub=""):
    url, by, link = PHOTOS[photo]
    st.markdown(f"<div class='banner fade' style=\"background-image:url('{url}')\"><div class='ov'></div>"
                f"{'<div class=sub>' + sub + '</div>' if sub else ''}<h1>{title}</h1>"
                f"<div class='credit'>Photo: <a href='{link}' target='_blank'>{by}</a> / Unsplash</div></div>", unsafe_allow_html=True)


def topbar():
    with st.container(key="topbar"):
        c1, c2, c3 = st.columns([0.9, 5.2, 3])
        with c1:
            if st.button(":material/home: Home", key="home_btn"):
                go("home")
        with c2:
            keys = list(PAGES)
            pick = st.segmented_control("Navigate", keys, format_func=lambda k: f"{PAGES[k][0]} {PAGES[k][1]}",
                                        default=S.page if S.page in keys else None, label_visibility="collapsed", key=f"nav_{S.page}")
            if pick and pick != S.page:
                go(pick)
        with c3:
            n = len(S.records)
            last = S.results["finished"].strftime("%H:%M UTC") if S.results else "none"
            st.markdown(f"<div class='small' style='text-align:right; padding-top:.45rem'>{n} records · last run {last}</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Report text (short sentences built from the numbers)
# ---------------------------------------------------------------------------

def txt_highlights(R):
    var = R["opts"]["variable"]
    u = UNIT[var]
    lv = {n: PL.key_levels(r) for n, r in R["pot"].items()}
    if not lv:
        return "No record could be fitted."
    parts = [f"**{n}**: {f2(v[100][0])} {u} (95% interval {f2(v[100][1])} to {f2(v[100][2])} {u})" for n, v in lv.items()]
    s = f"The 100-year {VARNAME[var]} is " + "; ".join(parts) + "."
    if len(lv) >= 2:
        vals = [v[100][0] for v in lv.values()]
        half = min((v[100][2] - v[100][1]) / 2 for v in lv.values())
        spread = max(vals) - min(vals)
        s += (f" The records differ by {f2(spread)} {u}, " + ("well inside the sampling uncertainty." if spread < half else "more than half an interval width."))
    return s


def txt_return_levels(R):
    var = R["opts"]["variable"]
    u = UNIT[var]
    lines = ["Each line is the fitted Generalized Pareto distribution. The band is the bootstrap 95% interval. "
             "Dots are the storm peaks at their empirical return periods."]
    for n, r in R["pot"].items():
        ub = "no upper bound" if not np.isfinite(r["upper_bound"]) else f"upper bound {f2(r['upper_bound'])} {u}"
        lines.append(f"- **{n}**: threshold {f2(r['threshold'])} {u}, {r['n_peaks']} storms ({f2(r['rate_per_year'])} per year), "
                     f"shape ξ = {sg(r['xi'])} ({ub}).")
    return "\n".join(lines)


def txt_ts(R):
    var = R["opts"]["variable"]
    u = UNIT[var]
    lines = [f"The record is standardised by a {R['opts']['window']}-year running mean and standard deviation, fitted once, "
             "and mapped back to each year. Solid lines: time-varying 100-year level. Dashed: the stationary level."]
    for n, ts in R["ts"].items():
        st_rl = PL.key_levels(R["pot"][n])[100][0]
        sm = PL.P.ts_eva_summary(ts, st_rl, 100)
        lines.append(f"- **{n}**: {f2(sm['rl_first'])} {u} in {sm['first_year']} to {f2(sm['rl_last'])} {u} in {sm['last_year']} "
                     f"({sg(sm['change'])} {u}); never more than {f2(max(abs(sm['rl_min'] - st_rl), abs(sm['rl_max'] - st_rl)))} {u} from the stationary value.")
    return "\n".join(lines)


def trend_rows(R, var):
    f, unit = (100.0, "cm/decade") if var == "swh" else (1.0, "s/decade")
    rows = []
    for n, item in R["trends"][var].items():
        tr = item["trend"]
        if tr is None:
            continue
        o, se, mk = tr["ols"], tr["sen"], tr["mk"]
        rows.append({"record": n, f"mean [{UNIT[var]}]": float(item["annual"].mean()),
                     f"OLS slope [{unit}]": o["slope"] * 10 * f, "OLS 95% low": o["slope_ci"][0] * 10 * f, "OLS 95% high": o["slope_ci"][1] * 10 * f,
                     "OLS p": o["p_value"], f"Sen slope [{unit}]": se["slope"] * 10 * f, "Mann-Kendall p": mk["p_value"]})
    return pd.DataFrame(rows), unit


def txt_trends(R):
    lines = ["Annual means (markers) with ordinary least squares trends (dashed). A trend counts when the OLS and Mann-Kendall p-values are both below 0.05."]
    for var in ("swh", "tm"):
        df, unit = trend_rows(R, var)
        for _, r in df.iterrows():
            sig = r["OLS p"] < 0.05 and r["Mann-Kendall p"] < 0.05
            lines.append(f"- **{r['record']}, {SYM[var]}**: {sg(r[f'OLS slope [{unit}]'])} {unit} (p = {r['OLS p']:.2f}), "
                         + ("**a trend**." if sig else "no trend."))
    return "\n".join(lines)


def txt_bc(R):
    bc = R["bc"]
    v = bc["validation"]
    inp = bc["inputs"]
    u = UNIT[R["opts"]["variable"]]
    s = (f"Reference: **{inp['ref_name']}**; model historical: **{inp['hist_name']}**; {len(inp['overlap'])} overlap years, "
         f"split into {len(bc['val_info']['calibration_years'])} calibration and {len(bc['val_info']['validation_years'])} validation years. ")
    s += (f"On the held-out years the raw model has a mean bias of {sg(v.loc['raw model', 'mean_bias'])} {u} and a 99.5th-percentile bias of "
          f"{sg(v.loc['raw model', 'p995_bias'])} {u}. Delta change leaves {sg(v.loc['delta change', 'p995_bias'])} {u} at the 99.5th percentile; "
          f"QDM leaves {sg(v.loc['QDM', 'p995_bias'])} {u}. The plot compares quantiles of each series with the reference.")
    return s


def txt_compare(R):
    var = R["opts"]["variable"]
    u = UNIT[var]
    lines = ["The 100-year level of each projected series, by correction method, with its bootstrap 95% interval."]
    for name in R["bc"]["projected"]:
        vals = {m: PL.key_levels(r, (100,))[100][0] for (n, m), r in R["bc"]["compare_pot"].items() if n == name and r is not None}
        if vals:
            lines.append(f"- **{name}**: " + ", ".join(f"{PL.METHOD_LABEL[m]} {f2(x)} {u}" for m, x in vals.items())
                         + f"; spread {f2(max(vals.values()) - min(vals.values()))} {u}.")
    return "\n".join(lines)


def txt_checks(R, name):
    r = R["pot"][name]
    var = R["opts"]["variable"]
    u = UNIT[var]
    s = (f"Storm peaks of **{name}** above the {r['percentile']:g}th-percentile threshold, separated by at least {r['min_gap_hours']:.0f} hours. "
         "In the quantile-quantile plot, points on the 1:1 line mean the fitted distribution matches the storms.")
    if name in R["sens"]:
        g = R["sens"][name]
        s += (f" Over nine threshold and storm-gap settings the 100-year level runs from {f2(g['RL_100'].min())} to {f2(g['RL_100'].max())} {u}.")
    if name in R["diag"]:
        s += " The diagnostics below show where the mean excess is linear and the shape is stable; the chosen threshold should sit in that range."
    return s


def section(n, title, text, fig=None, ratio=(1, 1.7)):
    st.markdown(f"<hr class='sep'><span class='sec-n'>{n}</span><span class='sec-t'>{title}</span>", unsafe_allow_html=True)
    l, r = st.columns(ratio, gap="large")
    with l:
        st.markdown(text)
    with r:
        if fig is not None:
            st.plotly_chart(fig, width="stretch")
    return l, r


def keylevel_html(R):
    var = R["opts"]["variable"]
    u = UNIT[var]
    out = []
    for i, (n, res) in enumerate(R["pot"].items()):
        lv = PL.key_levels(res)
        cells = "".join(f"<div class='c'><div class='t'>{T}-year</div><div class='v'>{f2(v[0])} <small>{u}</small></div>"
                        f"<div class='ci'>95%: {f2(v[1])} to {f2(v[2])}</div></div>" for T, v in lv.items())
        out.append(f"<div class='kl' style='animation-delay:{0.1 * i:.1f}s'><div class='nm'><span style='background:{C.color_for(n, i)}'></span>{n}</div>{cells}</div>")
    return "".join(out)


# ---------------------------------------------------------------------------
# Figures and exports for the report
# ---------------------------------------------------------------------------

def report_figures(R):
    var = R["opts"]["variable"]
    F_ = {}
    if R["pot"]:
        F_["key"] = C.key_levels({n: PL.key_levels(r) for n, r in R["pot"].items()}, var)
        F_["rl"] = C.return_levels(R["pot"], var)
    if R["ts"]:
        F_["ts"] = C.ts_eva_multi(R["ts"], {n: PL.key_levels(R["pot"][n])[100][0] for n in R["ts"]}, var)
    F_["trend_swh"] = C.annual_means(R["trends"]["swh"], "swh") if R["trends"]["swh"] else None
    F_["trend_tm"] = C.annual_means(R["trends"]["tm"], "tm") if R["trends"]["tm"] else None
    if R["bc"] is not None:
        F_["bc_q"] = C.validation_quantiles(R["bc"]["val_series"], var)
        if R["opts"]["correction"] == "compare":
            data = {}
            for (n, m), r in R["bc"]["compare_pot"].items():
                if r is not None:
                    data.setdefault(n, {})[PL.METHOD_LABEL[m]] = PL.key_levels(r, (100,))[100]
            F_["bc_cmp"] = C.method_compare(data, var)
    return F_


def html_report(R, F_):
    var = R["opts"]["variable"]
    o = R["opts"]
    css = ("body{font-family:Inter,Segoe UI,Helvetica,Arial,sans-serif;max-width:1150px;margin:2rem auto;color:#1f2933;padding:0 1rem}"
           "h1{color:#0b5394;margin-bottom:.2rem}.meta{color:#52606d;font-size:.9rem}"
           ".kl{display:grid;grid-template-columns:170px repeat(4,1fr);gap:.6rem;margin:.5rem 0}.kl .nm{font-weight:700;display:flex;align-items:center}"
           ".kl .nm span{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:.5rem}"
           ".kl .c{border:1px solid #dfe8f1;border-radius:12px;padding:.6rem .8rem;background:#f5f9fd}.kl .t{font-size:.75rem;color:#52606d;font-weight:600;text-transform:uppercase}"
           ".kl .v{font-size:1.6rem;font-weight:800;color:#0b5394}.kl .v small{font-size:.9rem;color:#52606d}.kl .ci{font-size:.8rem;color:#52606d}"
           ".lead{background:#f5f9fd;border-left:4px solid #0b5394;padding:.8rem 1rem;border-radius:8px}"
           ".sec{display:grid;grid-template-columns:1fr 1.7fr;gap:1.5rem;border-top:1px solid #e9eef3;padding-top:1rem;margin-top:1.4rem}"
           ".sec h2{font-size:1.2rem;margin:.2rem 0}.note{background:#fff8e6;border:1px solid #ffe2a8;border-radius:8px;padding:.5rem .8rem;font-size:.9rem}"
           "table{border-collapse:collapse;font-size:.85rem;margin-top:.6rem}td,th{border:1px solid #e3e8ee;padding:3px 7px;text-align:right}th{background:#f3f6fa}")

    def md(s):
        import re
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        lines = s.split("\n")
        out, inlist = [], False
        for l in lines:
            if l.startswith("- "):
                if not inlist:
                    out.append("<ul>"); inlist = True
                out.append(f"<li>{l[2:]}</li>")
            else:
                if inlist:
                    out.append("</ul>"); inlist = False
                out.append(f"<p>{l}</p>")
        if inlist:
            out.append("</ul>")
        return "".join(out)

    first = [True]

    def fig_html(fig):
        if fig is None:
            return ""
        h = fig.to_html(include_plotlyjs="cdn" if first[0] else False, full_html=False)
        first[0] = False
        return h

    def sec(n, title, text, fig, extra=""):
        return f"<div class='sec'><div><h2>{n}. {title}</h2>{md(text)}{extra}</div><div>{fig_html(fig)}</div></div>"

    parts = [f"<html><head><meta charset='utf-8'><title>Wave climate report</title><style>{css}</style></head><body>",
             "<h1>Wave climate report</h1>",
             f"<div class='meta'>Wave Climate Trend-Extreme Tool · {R['finished']:%Y-%m-%d %H:%M} UTC · records: {', '.join(R['targets'])} · "
             f"threshold {o['percentile']:g}th percentile · storm gap {o['gap']} h · {o['n_boot']} bootstrap samples · correction: {CORRECTIONS[o['correction']]}</div>"]
    for nte in R["notes"]:
        parts.append(f"<p class='note'>{nte}</p>")
    if R["bc"] is not None and R["bc"]["inputs"]["synthetic"]:
        parts.append("<p class='note'>The bias correction used a synthetic reference (model × 1.2 + 0.1). It demonstrates the method; it is not data.</p>")
    parts.append("<h2>Key results</h2>" + keylevel_html(R) + f"<div class='lead'>{md(txt_highlights(R))}</div>")
    k = 1
    if "key" in F_:
        parts.append(sec(k, "Design levels at a glance", "The four design return periods with their 95% intervals, for each record.", F_["key"])); k += 1
        rl_tab = tab_html(pd.concat([A.return_level_table(r).assign(record=n) for n, r in R["pot"].items()]).rename(columns=RL_COLS(var)), index=False)
        parts.append(sec(k, "Return levels", txt_return_levels(R), F_["rl"], rl_tab)); k += 1
    if "ts" in F_:
        parts.append(sec(k, "Change through time", txt_ts(R), F_["ts"])); k += 1
    df_s, _ = trend_rows(R, "swh")
    df_t, _ = trend_rows(R, "tm")
    parts.append(sec(k, "Trends in the annual means", txt_trends(R), F_.get("trend_swh"),
                     tab_html(pd.concat([df_s.assign(variable="Hs"), df_t.assign(variable="Tm")]), index=False))); k += 1
    if F_.get("trend_tm") is not None:
        parts.append(f"<div class='sec'><div></div><div>{fig_html(F_['trend_tm'])}</div></div>")
    if R["bc"] is not None:
        parts.append(sec(k, "Bias correction", txt_bc(R), F_["bc_q"], tab_html(R["bc"]["validation"].drop(columns=["n"])))); k += 1
        if "bc_cmp" in F_:
            parts.append(sec(k, "Correction methods compared", txt_compare(R), F_["bc_cmp"])); k += 1
    parts.append("<p class='meta'>Methods: POT with runs declustering, GPD with location fixed at the threshold, bootstrap intervals (Coles 2001); "
                 "transformed-stationary EVA (Mentaschi et al. 2016); quantile delta mapping (Cannon et al. 2015); OLS, Mann-Kendall, Sen's slope.</p></body></html>")
    return "\n".join(parts)


def zip_bundle(R, F_, html):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("report.html", html)
        for n, r in R["pot"].items():
            zf.writestr(f"return_levels/{n}.csv", A.return_level_table(r).round(4).to_csv(index=False))
            zf.writestr(f"storm_peaks/{n}.csv", r["peaks"].rename(R["opts"]["variable"]).to_csv())
        for n, ts in R["ts"].items():
            zf.writestr(f"nonstationary/{n}.csv", ts["table"].round(4).to_csv())
        for n, g in R["sens"].items():
            zf.writestr(f"sensitivity/{n}.csv", g.round(4).to_csv(index=False))
        for var in ("swh", "tm"):
            df, _ = trend_rows(R, var)
            zf.writestr(f"trends_{var}.csv", df.round(4).to_csv(index=False))
        if R["bc"] is not None:
            zf.writestr("bias_correction/validation.csv", R["bc"]["validation"].round(4).to_csv())
            for n, proj in R["bc"]["projected"].items():
                for m, s in proj.items():
                    zf.writestr(f"bias_correction/projected_{n}_{m}.csv", s.rename(R["opts"]["variable"]).to_csv())
        for k, fig in F_.items():
            if fig is not None:
                zf.writestr(f"figures/{k}.html", fig.to_html(include_plotlyjs="cdn", full_html=True))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------

if S.page == "home":
    url, by, link = PHOTOS["hero"]
    hero_html = f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
      html, body {{margin:0; padding:0; font-family: Inter, 'Segoe UI', Helvetica, Arial, sans-serif;}}
      .hero {{position: relative; height: 430px; border-radius: 22px; overflow: hidden; background: #062a4a url('{url}') center/cover;}}
      .hero .ov {{position:absolute; inset:0; background: linear-gradient(180deg, rgba(4,30,56,.35) 0%, rgba(4,30,56,.78) 100%);}}
      canvas {{position:absolute; left:0; bottom:0; width:100%; height:100%;}}
      .txt {{position:absolute; left: 7%; top: 19%; color: #fff; max-width: 70%;}}
      .txt h1 {{font-size: 3rem; font-weight: 800; margin: 0 0 .5rem 0; letter-spacing: -.02em; line-height: 1.05; animation: up .9s ease both; text-shadow: 0 4px 24px rgba(0,0,0,.35);}}
      .txt p {{font-size: 1.25rem; margin: 0; opacity: .95; animation: up .9s .25s ease both; text-shadow: 0 2px 12px rgba(0,0,0,.35);}}
      .tags {{margin-top: 1.1rem; animation: up .9s .45s ease both;}}
      .tags span {{display:inline-block; margin: 0 .4rem .4rem 0; padding: .3rem .75rem; border-radius: 999px; background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.35); font-size: .85rem;}}
      .credit {{position:absolute; right: .9rem; bottom: .5rem; color: rgba(255,255,255,.65); font-size: .58rem; letter-spacing: .01em;}}
      .credit a {{color: rgba(255,255,255,.9); text-decoration: none;}}
      @keyframes up {{from {{opacity:0; transform: translateY(16px);}} to {{opacity:1; transform:none;}}}}
    </style>
    <div class="hero"><div class="ov"></div><canvas id="w"></canvas>
      <div class="txt"><h1>Wave Climate<br>Trend-Extreme Tool</h1>
        <p>From a wave record and a climate projection to a design number, with every choice visible.</p>
        <div class="tags"><span>Bias correction</span><span>Trends</span><span>Return levels</span><span>Non-stationary extremes</span><span>Open source</span></div>
      </div>
      <div class="credit">Photo: <a href="{link}" target="_blank">{by}</a> / Unsplash</div>
    </div>
    <script>
      const c = document.getElementById('w'); const ctx = c.getContext('2d');
      function size() {{ c.width = c.clientWidth; c.height = c.clientHeight; }} size(); window.addEventListener('resize', size);
      const layers = [ {{a: 14, l: 0.012, s: 0.9, y: 0.86, col: 'rgba(120,200,255,0.35)'}}, {{a: 22, l: 0.008, s: 0.6, y: 0.90, col: 'rgba(60,150,230,0.45)'}}, {{a: 30, l: 0.005, s: 0.4, y: 0.94, col: 'rgba(11,83,148,0.75)'}} ];
      let t = 0;
      function draw() {{ ctx.clearRect(0,0,c.width,c.height);
        for (const L of layers) {{ ctx.beginPath(); ctx.moveTo(0, c.height);
          for (let x = 0; x <= c.width; x += 4) {{ const y = c.height*L.y + Math.sin(x*L.l + t*L.s)*L.a + Math.sin(x*L.l*2.3 - t*L.s*1.3)*L.a*0.35; ctx.lineTo(x, y); }}
          ctx.lineTo(c.width, c.height); ctx.closePath(); ctx.fillStyle = L.col; ctx.fill(); }}
        t += 0.02; requestAnimationFrame(draw); }} draw();
    </script>
    """
    if hasattr(st, "iframe"):
        st.iframe(hero_html, height=450)
    else:
        components.html(hero_html, height=450)

    with st.container(key="actions"):
        a1, a2, a3, _ = st.columns([1.7, 1.4, 1.2, 4])
        demo = a1.button(":material/play_arrow: Try it with demo data", type="primary")
        if a2.button(":material/upload: Use my own data"):
            go("setup")
        if a3.button(":material/groups: The team"):
            go("about")
    if demo:
        if not S.records:
            load_demo()
        S.targets = default_targets()
        S.opts = dict(PL.DEFAULTS)
        if run_analysis(st):
            go("results")

    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
    with st.container(key="tiles"):
        cols = st.columns(3)
        tiles = [("setup", ":material/database:  1 · Data", "Demo station, your CSV, NOAA buoys or Copernicus"),
                 ("setup", ":material/tune:  2 · Options", "Bias correction, stationary or non-stationary"),
                 ("results", ":material/insights:  3 · Report", "Design levels, trends and checks, ready to download")]
        for i, (col, (target, label, cap)) in enumerate(zip(cols, tiles)):
            with col:
                st.markdown(f"<style>.st-key-tiles div[data-testid='stColumn']:nth-child({i + 1}) button{{animation-delay:{0.12 * i + 0.1:.2f}s}}</style>", unsafe_allow_html=True)
                if st.button(label, key=f"tile_{i}", width="stretch"):
                    go(target)
                st.caption(cap)
    st.markdown("<p class='small fade d5' style='text-align:center; margin-top:1.2rem'>Shahbazbegian and Amini · University of Tehran and Columbia University · MIT license</p>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Setup: data + options + run
# ---------------------------------------------------------------------------

elif S.page == "setup":
    topbar()
    banner("Setup", "buoy", "Pick the data, pick the options, run.")
    left, right = st.columns([1.15, 1], gap="large")

    with left:
        st.markdown("### :material/database: 1 · Data")
        src = st.segmented_control("Source", ["Demo station", "Upload", "NOAA buoy", "Copernicus"], default="Demo station", label_visibility="collapsed")
        if src == "Demo station":
            cat = F.load_catalogue()
            st.caption(" · ".join(e["label"] for e in cat["stations"]))
            if st.button(":material/download: Load demo records", type="primary"):
                load_demo()
                st.toast(f"{len(S.records)} records loaded", icon="✅")
                st.rerun()
        elif src == "Upload":
            ups = st.file_uploader("CSV or gzip CSV with a time column and Hs and/or Tm", type=["csv", "gz"], accept_multiple_files=True)
            role_up = st.selectbox("Role", ROLES, index=0, help="observed or reference: the truth for bias correction. model historical / model future: climate-model runs.")
            if ups and st.button(":material/download: Load files", type="primary"):
                for f in ups:
                    try:
                        df, info = A.load_series(f)
                        add_record(os.path.splitext(os.path.splitext(f.name)[0])[0], df, info, role_up, "upload")
                    except Exception as e:
                        st.error(f"{f.name}: {e}")
                st.rerun()
        elif src == "NOAA buoy":
            c1, c2, c3 = st.columns(3)
            station = c1.text_input("Station", "44025", help="e.g. 44025 Long Island, 46042 Monterey Bay, 51001 Hawaii")
            years = c2.text_input("Years", "2015-2024")
            role_nd = c3.selectbox("Role", ROLES, index=0, key="role_ndbc")
            if st.button(":material/download: Fetch", type="primary"):
                try:
                    with st.spinner("Downloading from NDBC..."):
                        df, notes = F.fetch_ndbc(station, F.years_from_spec(years))
                    buf = io.StringIO(); df.to_csv(buf, index=False); buf.seek(0); buf.name = "ndbc.csv"
                    df2, info = A.load_series(buf)
                    add_record(f"NDBC {station}", df2, info, role_nd, "NOAA NDBC")
                    st.rerun()
                except Exception as e:
                    st.error(f"NDBC fetch failed: {e}")
        else:
            key = st.text_input("CDS API key", type="password", help="Free account at cds.climate.copernicus.eu. Used for this request only.")
            c1, c2 = st.columns(2)
            lat = c1.number_input("Latitude", -90.0, 90.0, 40.5, 0.1)
            lon = c2.number_input("Longitude", -180.0, 180.0, 2.0, 0.1)
            prod = st.radio("Product", ["ERA5 at the point", "C3S coastal station"], horizontal=True, label_visibility="collapsed")
            if prod.startswith("ERA5"):
                yrs = st.text_input("Years", "1979-2005", key="era5_years")
                if st.button(":material/download: Request ERA5", type="primary"):
                    try:
                        prog = st.progress(0.0, "submitting")
                        df, point = F.fetch_era5_point(lat, lon, F.years_from_spec(yrs), key=key or None, progress=lambda f, m: prog.progress(f, m))
                        buf = io.StringIO(); df.to_csv(buf, index=False); buf.seek(0); buf.name = "era5.csv"
                        df2, info = A.load_series(buf)
                        add_record(f"ERA5 {point}", df2, info, "reference", "Copernicus ERA5")
                        st.rerun()
                    except Exception as e:
                        st.error(f"CDS request failed: {e}")
            else:
                exp = st.selectbox("Experiment", list(F.C3S_REQUEST["experiment"]), format_func=lambda k: {"historical": "Historical 1976-2005", "reanalysis": "ERA5-forced 2001-2017", "rcp45": "RCP4.5 2041-2100", "rcp85": "RCP8.5 2041-2100"}[k])
                if st.button(":material/download: Request C3S station", type="primary"):
                    try:
                        prog = st.progress(0.0, "submitting")
                        df, where = F.fetch_c3s_station(lat, lon, exp, key=key or None, progress=lambda f, m: prog.progress(f, m))
                        buf = io.StringIO(); df.to_csv(buf, index=False); buf.seek(0); buf.name = "c3s.csv"
                        df2, info = A.load_series(buf)
                        add_record(f"C3S {exp}", df2, info, "model historical" if exp == "historical" else "model future", "Copernicus C3S")
                        st.rerun()
                    except Exception as e:
                        st.error(f"CDS request failed: {e}")

        if S.records:
            rows = [{"record": n, "role": r["role"], "period": f"{r['info']['start']:%Y} to {r['info']['end']:%Y}",
                     "years": round(r["info"]["span_years"], 1), "variables": ", ".join(SYM.get(k, k) for k in r["info"]["columns"])}
                    for n, r in S.records.items()]
            edited = st.data_editor(pd.DataFrame(rows), hide_index=True, width="stretch",
                                    disabled=["record", "period", "years", "variables"],
                                    column_config={"role": st.column_config.SelectboxColumn("role", options=ROLES, required=True)})
            for _, row in edited.iterrows():
                S.records[row["record"]]["role"] = row["role"]
            with st.expander("Remove records"):
                rm = st.multiselect("Records", list(S.records), label_visibility="collapsed")
                if rm and st.button(":material/delete: Remove"):
                    for n in rm:
                        del S.records[n]
                    S.targets = None
                    st.rerun()
        else:
            st.info("No records yet.", icon=":material/info:")

    with right:
        st.markdown("### :material/tune: 2 · Options")
        o = S.opts
        if S.records:
            S.targets = st.multiselect("Records to analyse", list(S.records), default=[t for t in (S.targets or default_targets()) if t in S.records])
        bc_ok = bool(S.records) and PL.bias_available(S.records, o["variable"])
        corr_keys = list(CORRECTIONS)
        o["correction"] = st.segmented_control("Bias correction", corr_keys, format_func=lambda k: CORRECTIONS[k], default=o["correction"]) or "off"
        if o["correction"] != "off" and not bc_ok and not o["synthetic_reference"]:
            st.caption(":material/info: Needs a reference record (role observed or reference) that overlaps a model historical run. "
                       "Without one the raw model is analysed. A synthetic demo reference is in the advanced options.")
        ext = st.segmented_control("Extremes", ["stationary", "both"], format_func=lambda k: {"stationary": "Stationary", "both": "Stationary + non-stationary"}[k],
                                   default="both" if o["nonstationary"] else "stationary")
        o["nonstationary"] = (ext or "both") == "both"

        if st.toggle("Advanced options", key="adv"):
            with st.container(border=True):
                a1, a2 = st.columns(2)
                o["variable"] = a1.segmented_control("Extremes of", ["swh", "tm"], format_func=lambda k: VARNAME[k].capitalize(), default=o["variable"]) or "swh"
                o["percentile"] = a2.number_input("Threshold percentile", 90.0, 99.9, float(o["percentile"]), 0.1, format="%.1f")
                a3, a4 = st.columns(2)
                o["gap"] = int(a3.number_input("Storm gap (hours)", 0, 240, int(o["gap"]), 6))
                o["n_boot"] = int(a4.number_input("Bootstrap samples", 100, 5000, int(o["n_boot"]), 100))
                a5, a6 = st.columns(2)
                rp = a5.text_input("Return periods (years)", ", ".join(str(int(x)) for x in o["return_periods"]))
                try:
                    o["return_periods"] = tuple(sorted({int(float(x)) for x in rp.replace(";", ",").split(",") if x.strip()} | {10, 25, 50, 100}))
                except ValueError:
                    st.error("Return periods must be numbers.")
                o["window"] = int(a6.number_input("Non-stationary window (years)", 5, 60, int(o["window"]), 5))
                a7, a8 = st.columns(2)
                o["kind"] = a7.segmented_control("Correction type", ["multiplicative", "additive"], default=o["kind"]) or "multiplicative"
                o["split"] = a8.segmented_control("Validation split", ["alternate", "halves"], default=o["split"]) or "alternate"
                o["diagnostics"] = st.checkbox("Threshold diagnostics", value=o["diagnostics"])
                o["sensitivity"] = st.checkbox("Threshold and storm-gap sensitivity grid", value=o["sensitivity"])
                o["synthetic_reference"] = st.checkbox("Synthetic reference for bias correction (demo only; model × 1.2 + 0.1)", value=o["synthetic_reference"])
        S.opts = o
        st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)
        with st.container(key="runbtn"):
            run = st.button(":material/rocket_launch: Run analysis", type="primary", width="stretch", disabled=not S.records)
        if run and run_analysis(st):
            go("results")

# ---------------------------------------------------------------------------
# Results: the report
# ---------------------------------------------------------------------------

elif S.page == "results":
    topbar()
    R = S.results
    if R is None:
        banner("Report", "storm")
        st.info("No results yet. Run an analysis first.", icon=":material/info:")
        if st.button(":material/play_arrow: Run the demo now", type="primary"):
            if not S.records:
                load_demo()
            S.targets = default_targets()
            if run_analysis(st):
                st.rerun()
    else:
        o = R["opts"]
        var = o["variable"]
        banner("Report", "storm", f"{', '.join(R['targets'])} · bias correction: {CORRECTIONS[o['correction']].lower()} · {R['finished']:%Y-%m-%d %H:%M} UTC")
        if "_export" not in R:  # build figures and exports once per run
            F0 = report_figures(R)
            h0 = html_report(R, F0)
            R["_export"] = (F0, h0, zip_bundle(R, F0, h0))
        F_, html, zbytes = R["_export"]
        c1, c2, c3 = st.columns([3.2, 1.2, 1.2])
        c1.markdown(f"<div class='small'>Threshold {o['percentile']:g}th percentile · storm gap {o['gap']} h · {o['n_boot']} bootstrap samples · "
                    f"{'non-stationary window ' + str(o['window']) + ' years' if o['nonstationary'] else 'stationary only'}</div>", unsafe_allow_html=True)
        stamp = R["finished"].strftime("%Y%m%d_%H%M")
        c2.download_button(":material/description: HTML report", data=html, file_name=f"wave_climate_report_{stamp}.html", mime="text/html", type="primary", width="stretch")
        c3.download_button(":material/folder_zip: All results", data=zbytes, file_name=f"wave_climate_results_{stamp}.zip", mime="application/zip", width="stretch")
        for nte in R["notes"]:
            st.markdown(f"<div class='note'>{nte}</div>", unsafe_allow_html=True)
        if R["bc"] is not None and R["bc"]["inputs"]["synthetic"]:
            st.markdown("<div class='note'>The bias correction used a synthetic reference (model × 1.2 + 0.1). It demonstrates the method; it is not data.</div>", unsafe_allow_html=True)
        for n, why in R["skipped"].items():
            st.markdown(f"<div class='note'>{n}: {why}</div>", unsafe_allow_html=True)

        # Highlights
        if R["pot"]:
            st.markdown(f"### Design {VARNAME[var]}")
            st.markdown(keylevel_html(R), unsafe_allow_html=True)
            st.markdown(f"<div class='lead'>{txt_highlights(R).replace('**', '')}</div>", unsafe_allow_html=True)

        k = 1
        if R["pot"]:
            section(k, "Design levels at a glance", "The 10, 25, 50 and 100-year levels with their 95% intervals. Overlapping whiskers mean the records cannot be told apart.", F_["key"]); k += 1
            l, r = section(k, "Return levels", txt_return_levels(R), F_["rl"]); k += 1
            with l:
                with st.expander("Table"):
                    table2(pd.concat([A.return_level_table(res).assign(record=n) for n, res in R["pot"].items()])
                           .rename(columns=RL_COLS(var)))
        if R["ts"]:
            section(k, "Change through time", txt_ts(R), F_["ts"]); k += 1
        if F_.get("trend_swh") is not None:
            l, r = section(k, "Trends in the annual means", txt_trends(R)); k += 1
            with r:
                if F_.get("trend_tm") is not None:
                    th, tt = st.tabs(["Wave height", "Wave period"])
                    with th:
                        st.plotly_chart(F_["trend_swh"], width="stretch")
                    with tt:
                        st.plotly_chart(F_["trend_tm"], width="stretch")
                else:
                    st.plotly_chart(F_["trend_swh"], width="stretch")
            with l:
                with st.expander("Table"):
                    for v in ("swh", "tm"):
                        df, _ = trend_rows(R, v)
                        if len(df):
                            st.caption(SYM[v]); table2(df)
        if R["bc"] is not None:
            l, r = section(k, "Bias correction", txt_bc(R), F_["bc_q"]); k += 1
            with l:
                with st.expander("Validation table"):
                    table2(R["bc"]["validation"].drop(columns=["n"]).reset_index().rename(columns={"method": "series"}))
            if "bc_cmp" in F_:
                section(k, "Correction methods compared", txt_compare(R), F_["bc_cmp"]); k += 1

        if R["pot"]:
            st.markdown(f"<hr class='sep'><span class='sec-n'>{k}</span><span class='sec-t'>Checks on the fit</span>", unsafe_allow_html=True)
            pick = st.segmented_control("Record", list(R["pot"]), default=list(R["pot"])[0], label_visibility="collapsed", key="chk_pick") or list(R["pot"])[0]
            l, r = st.columns([1, 1.7], gap="large")
            with l:
                st.markdown(txt_checks(R, pick))
                if pick in R["sens"]:
                    with st.expander("Sensitivity grid"):
                        table2(R["sens"][pick].rename(columns={"gap_hours": "gap (h)", "n_peaks": "storms", "rate_per_year": "per year", "RL_100": "100-yr level"}))
            with r:
                res = R["pot"][pick]
                t1, t2, t3 = st.tabs(["Storm peaks", "Quantile-quantile", "Threshold diagnostics"])
                with t1:
                    st.plotly_chart(C.peaks(R["series"][pick], res, pick, var), width="stretch")
                with t2:
                    st.plotly_chart(C.qq(res, pick, var), width="stretch")
                with t3:
                    if pick in R["diag"]:
                        mrl, ps = R["diag"][pick]
                        st.plotly_chart(C.diagnostics(mrl, ps, res["threshold"], var), width="stretch")
                    else:
                        st.caption("Turn on threshold diagnostics in the advanced options.")
            k += 1

        st.markdown(f"<hr class='sep'><span class='sec-n'>{k}</span><span class='sec-t'>Data used</span>", unsafe_allow_html=True)
        l, r = st.columns([1, 1.7], gap="large")
        with l:
            rows = [{"record": n, "role": S.records[n]["role"], "period": f"{S.records[n]['info']['start']:%Y} to {S.records[n]['info']['end']:%Y}",
                     "years": round(S.records[n]["info"]["span_years"], 1)} for n in R["targets"] if n in S.records]
            srcs = sorted({S.records[n]["source"] for n in R["targets"] if n in S.records})
            st.markdown("The records behind this report and their seasonal cycle (monthly means). " + ("Source: " + "; ".join(srcs) + "." if srcs else ""))
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        with r:
            recs = {n: A.to_series(S.records[n]["df"], var) for n in R["targets"] if n in S.records and var in S.records[n]["df"].columns}
            if recs:
                st.plotly_chart(C.seasonal(recs, var), width="stretch")

# ---------------------------------------------------------------------------
# Team
# ---------------------------------------------------------------------------

elif S.page == "about":
    topbar()
    banner("Team", "aerial")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class='card d1'><div class='avatar'>AS</div><h3>Amirhussein Shahbazbegian</h3>
        <div class='role'>Lead developer and first author</div>
        <p>M.Sc. student, Energy Systems Engineering, School of Energy and Sustainable Resources Engineering, University of Tehran.
        Marine renewable energy, wave climate projections, machine learning.</p>
        <p><a href='https://www.linkedin.com/in/amirhussein-shahbazbegian' target='_blank'>LinkedIn</a>
        <a href='https://shahbazbegian.wixsite.com' target='_blank'>Website</a>
        <a href='mailto:shahbazbegian@ut.ac.ir'>Email</a></p></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class='card d2'><div class='avatar'>EA</div><h3>Erfan Amini</h3>
        <div class='role'>Supervisor and co-author</div>
        <p>Lecturer and Columbia Science Fellow, Department of Earth and Environmental Sciences, Columbia University.
        Coastal flood modeling, nature-based coastal defenses, optimization.</p>
        <p><a href='https://erfan-amini.super.site/' target='_blank'>Website</a>
        <a href='https://www.linkedin.com/in/erfan-amini/' target='_blank'>LinkedIn</a>
        <a href='https://x.com/erfan_amini_' target='_blank'>X</a>
        <a href='https://github.com/erfan-amini' target='_blank'>GitHub</a></p></div>""", unsafe_allow_html=True)
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown("""<div class='card d3'><h3>Methods</h3>
        <p>Quantile delta mapping and delta change (Cannon et al. 2015; Hay et al. 2000) · OLS, Mann-Kendall, Sen's slope · POT-GPD with runs declustering, bootstrap intervals and threshold diagnostics (Coles 2001; Scarrott and MacDonald 2012) · Transformed-stationary EVA (Mentaschi et al. 2016).</p></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown("""<div class='card d4'><h3>Cite</h3>
        <p class='cr'>Shahbazbegian, A. and Amini, E. <i>A Unified Trend-Extreme Analysis Tool for Wave Climate Projection under Climate Change.</i> Manuscript. Source code:
        <a href='https://github.com/erfan-amini/wave-climate-tool' target='_blank'>github.com/erfan-amini/wave-climate-tool</a>, MIT license.</p></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class='card d5'><h3>Credits</h3>
        <p class='cr'>Demonstration data: Copernicus Climate Change Service, <i>Ocean surface wave time series for the European coast</i> (Caires and Yan 2020). Generated using Copernicus Climate Change Service information 2020.
        Photos from Unsplash (Unsplash License): {", ".join(f"<a href='{v[2]}' target='_blank'>{v[1]}</a>" for v in PHOTOS.values())}. Icons: Material Symbols.</p></div>""", unsafe_allow_html=True)
