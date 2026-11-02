"""
Wave Climate Trend-Extreme Tool

Joint trend and extreme-value analysis of wave records and climate-model
projections, with bias correction against a reference record and both
stationary and non-stationary return levels.

Run locally:  streamlit run app.py
"""

import io
import json
import os
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

import analysis as A
import charts as C
import fetch as F
import projection as P

st.set_page_config(page_title="Wave Climate Trend-Extreme Tool", page_icon="🌊", layout="wide",
                   initial_sidebar_state="expanded")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROLES = ["observed", "reference", "model historical", "model future", "projected"]
VARNAME = {"swh": "Significant wave height", "tm": "Mean wave period"}
UNIT = {"swh": "m", "tm": "s"}
PAGES = ["Overview", "1 · Data", "2 · Projection", "3 · Trends", "4 · Extremes", "5 · Report"]

st.markdown("""
<style>
  .block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1250px;}
  h1, h2, h3 {letter-spacing: -0.01em;}
  .hero {padding: 1.4rem 1.6rem; border-radius: 14px; background: linear-gradient(135deg, #0b5394 0%, #1a7f9a 100%); color: white; margin-bottom: 1rem;}
  .hero h1 {color: white; margin: 0 0 0.4rem 0; font-size: 2rem;}
  .hero p {margin: 0; font-size: 1.05rem; opacity: 0.95;}
  .card {border: 1px solid #e3e8ee; border-radius: 12px; padding: 0.9rem 1.1rem; background: #fff; height: 100%;}
  .card h4 {margin: 0 0 0.3rem 0; font-size: 0.95rem; color: #0b5394;}
  .card p {margin: 0; font-size: 0.88rem; color: #3e4c59;}
  .step {display:flex; gap: 0.6rem; align-items: stretch; margin: 0.6rem 0 1rem 0;}
  .step div {flex: 1; border-radius: 10px; padding: 0.7rem 0.8rem; background: #f3f6fa; border: 1px solid #e3e8ee; font-size: 0.86rem;}
  .step div b {display:block; color:#0b5394; margin-bottom: 0.2rem;}
  .status {font-size: 0.85rem; color: #3e4c59;}
  .status .ok {color: #1a7f5a; font-weight: 600;}
  .status .no {color: #9aa5b1;}
  div[data-testid="stMetric"] {background: #f8fafc; border: 1px solid #e3e8ee; border-radius: 10px; padding: 0.6rem 0.8rem;}
  div[data-testid="stMetric"] label {color: #52606d;}
  .small {font-size: 0.85rem; color: #52606d;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def init_state():
    defaults = {"records": {}, "pot": {}, "ts": {}, "proj_validation": None, "proj_series": None,
                "trend_table": None, "figs": {}, "tables": {}, "diag": {}, "sens": {}, "log": []}
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()
S = st.session_state


def add_record(name, df, info, role, source):
    S.records[name] = {"df": df, "info": info, "role": role, "source": source}
    S.log.append(f"{datetime.now():%H:%M} loaded {name} ({role}, {source})")


def records_with(variable, roles=None):
    return [n for n, r in S.records.items() if variable in r["df"].columns and (roles is None or r["role"] in roles)]


def series(name, variable):
    return A.to_series(S.records[name]["df"], variable)


def keep(key, fig=None, table=None):
    if fig is not None:
        S.figs[key] = fig
    if table is not None:
        S.tables[key] = table


@st.cache_data(show_spinner=False)
def load_file(path):
    return A.load_series(path)


def load_catalogue_station(entry):
    for name, rec in entry["records"].items():
        df, info = load_file(F.resolve_catalogue_file(rec["file"]))
        add_record(name, df, info, rec["role"], entry["label"])


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🌊 Wave Climate\n### Trend-Extreme Tool")
    page = st.radio("Workflow", PAGES, label_visibility="collapsed", key="page")
    st.divider()
    variable = st.radio("Variable", ["swh", "tm"], format_func=lambda v: VARNAME[v], horizontal=True)
    st.divider()
    n_rec = len(S.records)
    st.markdown("**Session**")
    st.markdown(
        "<div class='status'>"
        f"<div>{'<span class=ok>●</span>' if n_rec else '<span class=no>○</span>'} {n_rec} record(s) loaded</div>"
        f"<div>{'<span class=ok>●</span>' if S.proj_validation is not None else '<span class=no>○</span>'} bias correction validated</div>"
        f"<div>{'<span class=ok>●</span>' if S.trend_table is not None else '<span class=no>○</span>'} trends computed</div>"
        f"<div>{'<span class=ok>●</span>' if S.pot else '<span class=no>○</span>'} {len(S.pot)} extreme-value fit(s)</div>"
        f"<div>{'<span class=ok>●</span>' if S.ts else '<span class=no>○</span>'} {len(S.ts)} non-stationary fit(s)</div>"
        "</div>", unsafe_allow_html=True)
    if n_rec and st.button("Reset session"):
        for k in list(S.keys()):
            del S[k]
        st.rerun()

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

if page == "Overview":
    st.markdown("""<div class='hero'><h1>Wave Climate Trend-Extreme Tool</h1>
    <p>From a wave record and a climate projection to a design number, with every choice on the table:
    bias correction, trend, threshold, stationarity.</p></div>""", unsafe_allow_html=True)

    st.markdown("""<div class='step'>
    <div><b>1 · Data</b>Load records from the built-in catalogue, your own CSV files, NOAA buoys, or Copernicus with your key.</div>
    <div><b>2 · Projection</b>Correct a model's bias against a reference record by delta change or quantile delta mapping, and validate on held-out years.</div>
    <div><b>3 · Trends</b>Annual means with OLS, Mann-Kendall and Sen's slope.</div>
    <div><b>4 · Extremes</b>Peaks over threshold, GPD, bootstrap intervals, diagnostics, and time-varying return levels.</div>
    <div><b>5 · Report</b>Every table and figure of the session in one HTML report and one ZIP.</div>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("#### Start")
        cat = F.load_catalogue()
        if st.button("Load the demonstration station (C3S southeast Greenland + Brisbane buoy)", type="primary"):
            for entry in cat["stations"]:
                load_catalogue_station(entry)
            st.success(f"{len(S.records)} records loaded. Continue with 1 · Data or go straight to 4 · Extremes.")
        st.markdown("<p class='small'>Or open <b>1 · Data</b> to upload your own records or fetch them online.</p>", unsafe_allow_html=True)
    with c2:
        st.markdown("#### Methods")
        st.markdown("""<div class='card'><p>
        <b>Bias correction</b>: delta change; quantile delta mapping (Cannon et al. 2015), by calendar month, with a split-sample test.<br>
        <b>Trends</b>: OLS with 95% interval; Mann-Kendall; Sen's slope.<br>
        <b>Extremes</b>: POT with runs declustering; GPD, location fixed at the threshold; bootstrap intervals; mean residual life, parameter stability, QQ; sensitivity grid.<br>
        <b>Non-stationary extremes</b>: transformed-stationary approach (Mentaschi et al. 2016).
        </p></div>""", unsafe_allow_html=True)
    st.markdown("<p class='small'>Shahbazbegian, A. and Amini, E. A Unified Trend-Extreme Analysis Tool for Wave Climate Projection under Climate Change. MIT license.</p>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 1 · Data
# ---------------------------------------------------------------------------

elif page == "1 · Data":
    st.markdown("## 1 · Data")
    st.markdown("<p class='small'>Every record needs a time column and a wave height and/or period column. Give each record a role: "
                "<b>observed</b> or <b>reference</b> (used as the truth for bias correction), <b>model historical</b>, <b>model future</b>, "
                "or <b>projected</b> (produced by step 2).</p>", unsafe_allow_html=True)
    t_cat, t_up, t_ndbc, t_cds = st.tabs(["Catalogue", "Upload CSV", "NOAA NDBC buoy", "Copernicus CDS"])

    with t_cat:
        cat = F.load_catalogue()
        labels = {e["label"]: e for e in cat["stations"]}
        pick = st.multiselect("Stations", list(labels), default=list(labels)[:1])
        st.caption(cat.get("note", ""))
        if st.button("Load selected stations", type="primary"):
            for lab in pick:
                load_catalogue_station(labels[lab])
            st.success(f"Loaded. {len(S.records)} records in the session.")

    with t_up:
        st.caption("CSV or gzip CSV. Column names are matched against common variants (swh, Hs, Hs (m); tm, mp1, Tz; tp). "
                   "Missing-value codes such as -99.9 become NaN.")
        ups = st.file_uploader("Files", type=["csv", "gz"], accept_multiple_files=True, label_visibility="collapsed")
        role_up = st.selectbox("Role for the uploaded files", ROLES, index=0)
        if ups and st.button("Load uploads"):
            for f in ups:
                try:
                    df, info = A.load_series(f)
                    name = os.path.splitext(os.path.splitext(f.name)[0])[0]
                    add_record(name, df, info, role_up, "upload")
                    st.success(f"{name}: {info['n_rows']:,} rows, {info['span_years']:.1f} years")
                except Exception as e:
                    st.error(f"{f.name}: {e}")

    with t_ndbc:
        st.caption("Historical standard meteorological files from the National Data Buoy Center (significant wave height WVHT, "
                   "dominant period DPD, average period APD). No key needed. Example station: 44025 (Long Island), 46042 (Monterey Bay), 51001 (Hawaii).")
        c1, c2, c3 = st.columns([1, 1, 1])
        station = c1.text_input("Station ID", "44025")
        years = c2.text_input("Years", "2015-2024", help="A range such as 2015-2024, or a list such as 2016,2018")
        role_nd = c3.selectbox("Role", ROLES, index=0, key="role_ndbc")
        if st.button("Fetch from NDBC", type="primary"):
            try:
                with st.spinner("Downloading..."):
                    df, notes = F.fetch_ndbc(station, F.years_from_spec(years))
                buf = io.StringIO(); df.to_csv(buf, index=False); buf.seek(0); buf.name = "ndbc.csv"
                df2, info = A.load_series(buf)
                add_record(f"NDBC {station} {years}", df2, info, role_nd, "NOAA NDBC")
                st.success(f"{len(df2):,} rows. " + "; ".join(notes))
            except Exception as e:
                st.error(f"NDBC fetch failed: {e}")

    with t_cds:
        st.caption("Requests to the Copernicus Climate Data Store with your own API key (free account at cds.climate.copernicus.eu). "
                   "The key is used for this request only and is not stored. CDS queues requests; ERA5 at a point takes minutes per year, "
                   "the C3S coastal product can take much longer because the files cover the whole domain.")
        key = st.text_input("CDS API key", type="password", help="Your personal access token from your CDS profile page")
        c1, c2 = st.columns(2)
        lat = c1.number_input("Latitude", -90.0, 90.0, 40.5, 0.1)
        lon = c2.number_input("Longitude", -180.0, 180.0, 2.0, 0.1)
        which = st.radio("Product", ["ERA5 reanalysis at the point", "C3S coastal wave time series (nearest station)"], horizontal=True)
        if which.startswith("ERA5"):
            yrs = st.text_input("Years", "1995-2005", key="era5_years")
            role_cds = st.selectbox("Role", ROLES, index=1, key="role_era5")
            if st.button("Request ERA5", type="primary"):
                try:
                    prog = st.progress(0.0, "submitting")
                    df, point = F.fetch_era5_point(lat, lon, F.years_from_spec(yrs), key=key or None,
                                                   progress=lambda f, m: prog.progress(f, m))
                    buf = io.StringIO(); df.to_csv(buf, index=False); buf.seek(0); buf.name = "era5.csv"
                    df2, info = A.load_series(buf)
                    add_record(f"ERA5 {point} {yrs}", df2, info, role_cds, "Copernicus ERA5")
                    st.success(f"{len(df2):,} rows at {point}")
                except Exception as e:
                    st.error(f"CDS request failed: {e}")
        else:
            exp = st.selectbox("Experiment", list(F.C3S_REQUEST["experiment"]), format_func=lambda k: {"historical": "Historical 1976-2005", "reanalysis": "ERA5-forced 2001-2017", "rcp45": "RCP4.5 2041-2100", "rcp85": "RCP8.5 2041-2100"}[k])
            role_c3s = st.selectbox("Role", ROLES, index=2 if exp == "historical" else 3, key="role_c3s")
            if st.button("Request C3S station", type="primary"):
                try:
                    prog = st.progress(0.0, "submitting")
                    df, where = F.fetch_c3s_station(lat, lon, exp, key=key or None, progress=lambda f, m: prog.progress(f, m))
                    buf = io.StringIO(); df.to_csv(buf, index=False); buf.seek(0); buf.name = "c3s.csv"
                    df2, info = A.load_series(buf)
                    add_record(f"C3S {exp} {where}", df2, info, role_c3s, "Copernicus C3S")
                    st.success(f"{len(df2):,} rows at station {where}")
                except Exception as e:
                    st.error(f"CDS request failed: {e}")

    st.markdown("#### Loaded records")
    if not S.records:
        st.info("Nothing loaded yet.")
    else:
        rows = []
        for n, r in S.records.items():
            info, df = r["info"], r["df"]
            row = {"record": n, "role": r["role"], "source": r["source"], "start": info["start"].date(), "end": info["end"].date(),
                   "years": round(info["span_years"], 2), "step (h)": info["step_hours"], "rows": info["n_rows"]}
            if variable in df.columns:
                st_ = A.summary_stats(A.to_series(df, variable))
                row.update({f"mean {variable}": round(st_["mean"], 3), f"p99 {variable}": round(st_["p99"], 3), f"max {variable}": round(st_["max"], 3)})
            rows.append(row)
        table = pd.DataFrame(rows)
        edited = st.data_editor(table, hide_index=True, width="stretch", disabled=[c for c in table.columns if c != "role"],
                                column_config={"role": st.column_config.SelectboxColumn("role", options=ROLES, required=True)})
        for _, row in edited.iterrows():
            S.records[row["record"]]["role"] = row["role"]
        keep("records", table=table)
        rm = st.multiselect("Remove records", list(S.records))
        if rm and st.button("Remove"):
            for n in rm:
                del S.records[n]
            st.rerun()

        names = records_with(variable)
        if names:
            left, right = st.columns([1, 1])
            with left:
                fig = C.seasonal({n: series(n, variable) for n in names}, variable)
                st.plotly_chart(fig, width="stretch"); keep(f"seasonal_{variable}", fig)
            with right:
                pick = st.selectbox("Preview", names)
                fig = C.record(series(pick, variable), pick, variable)
                st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# 2 · Projection
# ---------------------------------------------------------------------------

elif page == "2 · Projection":
    st.markdown("## 2 · Projection: bias correction against a reference")
    st.markdown("<p class='small'>A climate model's wave series carries a bias. Two corrections are offered. <b>Delta change</b> applies the model's "
                "change in the monthly mean to the reference record. <b>Quantile delta mapping</b> (QDM) matches every quantile of the model to the "
                "reference and preserves the model's projected change at each quantile, so the extremes keep the climate signal. "
                "Both are fitted by calendar month and tested on held-out years before they are applied.</p>", unsafe_allow_html=True)
    refs = records_with(variable, ["observed", "reference"])
    hists = records_with(variable, ["model historical"])
    futs = records_with(variable, ["model future"])
    demo = any(n.startswith("Synthetic reference") for n in refs)
    if not hists:
        st.warning("This step needs a model historical run (role model historical), a reference record over the same years "
                   "(role observed or reference), and one or more model future runs. Load them in 1 · Data.")
    else:
        with st.expander("No reference record that overlaps the model historical run? Build a synthetic pair to try the interface"):
            st.caption("Takes the first twenty years of the longest loaded record as a synthetic model historical run, and builds a synthetic "
                       "reference from it (times 1.2 plus 0.1, with small noise). Both are labelled as such everywhere. "
                       "They demonstrate the workflow; they are not data.")
            if st.button("Build synthetic pair (demo)"):
                longest = max(records_with(variable), key=lambda n: S.records[n]["info"]["span_years"])
                base = series(longest, variable)
                yrs = sorted(set(base.index.year))[:20]
                h = base[np.isin(base.index.year, yrs)]
                rng = np.random.default_rng(0)
                syn = h * 1.2 + 0.1 + rng.normal(0, 0.02, len(h))
                for name, ser, role, src in (("Synthetic model historical (demo)", h, "model historical", f"first {len(yrs)} years of {longest}"),
                                             ("Synthetic reference (demo)", syn, "reference", "synthetic: historical × 1.2 + 0.1")):
                    df = pd.DataFrame({"time": ser.index, variable: ser.values})
                    info = {"columns": {variable: "synthetic"}, "n_rows": len(df), "start": df.time.min(), "end": df.time.max(),
                            "step_hours": S.records[longest]["info"]["step_hours"], "duplicates_removed": 0, "span_years": A.record_length_years(df.time)}
                    add_record(name, df, info, role, src)
                st.rerun()
        if not refs:
            st.warning("No reference record loaded (role observed or reference).")
    if refs and hists:
        c1, c2, c3 = st.columns(3)
        hist_name = c2.selectbox("Model historical run", hists, index=len(hists) - 1)
        hy = set(series(hist_name, variable).index.year)
        overlapping = [r for r in refs if set(series(r, variable).index.year) & hy]
        ref_name = c1.selectbox("Reference record", refs, index=refs.index(overlapping[0]) if overlapping else 0)
        fut_names = c3.multiselect("Model future runs to project", futs, default=futs)
        c4, c5 = st.columns(2)
        kind = c4.radio("Correction type", ["multiplicative", "additive"], horizontal=True,
                        help="Multiplicative scales values (natural for a positive variable such as wave height); additive shifts them.")
        mode = c5.radio("Split-sample test", ["alternate", "halves"], horizontal=True,
                        help="alternate: odd years calibrate, even years validate; halves: first half calibrates, second validates.")
        ref_s, hist_s = series(ref_name, variable), series(hist_name, variable)
        overlap = sorted(set(ref_s.index.year) & set(hist_s.index.year))
        if len(overlap) < 2:
            st.error(f"The reference ({ref_s.index.year.min()} to {ref_s.index.year.max()}) and the model historical run "
                     f"({hist_s.index.year.min()} to {hist_s.index.year.max()}) share fewer than two years. Bias correction needs an overlap.")
        else:
            st.caption(f"Overlap: {overlap[0]} to {overlap[-1]} ({len(overlap)} years). " + ("Synthetic demonstration data." if demo else ""))
            ref_o = ref_s[np.isin(ref_s.index.year, overlap)]
            hist_o = hist_s[np.isin(hist_s.index.year, overlap)]
            if st.button("Validate the corrections on held-out years", type="primary"):
                with st.spinner("Fitting and scoring..."):
                    tab, ser, info = P.validate_corrections(ref_o, hist_o, kind=kind, mode=mode)
                S.proj_validation = {"table": tab, "series": ser, "info": info, "variable": variable, "ref": ref_name, "hist": hist_name, "kind": kind}
            if S.proj_validation is not None and S.proj_validation["variable"] == variable:
                v = S.proj_validation
                st.markdown("#### Validation on held-out years")
                st.caption(f"Calibration years: {len(v['info']['calibration_years'])}; validation years: {len(v['info']['validation_years'])}. "
                           f"Metrics in {UNIT[variable]}; closer to zero is better (std ratio closer to one).")
                tab = v["table"].rename(columns={"mean_bias": "mean bias", "std_ratio": "std ratio", "seasonal_rmse": "seasonal RMSE",
                                                 "quantile_rmse": "quantile RMSE", "p99_bias": "p99 bias", "p995_bias": "p99.5 bias",
                                                 "annual_max_bias": "annual max bias"}).drop(columns=["n"])
                st.dataframe(tab.style.format("{:.3f}").background_gradient(cmap="RdYlGn_r", subset=["quantile RMSE", "seasonal RMSE"]), width="stretch")
                keep("validation", table=tab.reset_index())
                l, r = st.columns(2)
                with l:
                    fig = C.validation_quantiles(v["series"], variable); st.plotly_chart(fig, width="stretch"); keep("validation_quantiles", fig)
                with r:
                    fig = C.validation_seasonal(v["series"], variable); st.plotly_chart(fig, width="stretch"); keep("validation_seasonal", fig)
            st.markdown("#### Project the future runs")
            methods = st.multiselect("Methods", ["raw", "delta", "QDM"], default=["raw", "delta", "QDM"])
            if fut_names and st.button("Produce projected series", type="primary"):
                out = {}
                with st.spinner("Projecting..."):
                    for fn in fut_names:
                        fut_s = series(fn, variable)
                        for m in methods:
                            proj = P.project(ref_o, hist_o, fut_s, method=m, kind=kind)
                            label = f"{fn} · {m}"
                            df = pd.DataFrame({"time": proj.index, variable: proj.values})
                            info = {"columns": {variable: m}, "n_rows": len(df), "start": df.time.min(), "end": df.time.max(),
                                    "step_hours": S.records[fn]["info"]["step_hours"], "duplicates_removed": 0, "span_years": A.record_length_years(df.time)}
                            add_record(label, df, info, "projected", f"{m} correction of {fn} against {ref_name}")
                            out[label] = proj
                S.proj_series = {"variable": variable, "names": list(out)}
                st.success(f"{len(out)} projected series added as records (role projected). They are available in 3 · Trends and 4 · Extremes.")
            if S.proj_series and S.proj_series["variable"] == variable:
                names = [n for n in S.proj_series["names"] if n in S.records]
                if names:
                    fig = C.projection_compare({n: series(n, variable) for n in names}, variable)
                    st.plotly_chart(fig, width="stretch"); keep("projection_compare", fig)

# ---------------------------------------------------------------------------
# 3 · Trends
# ---------------------------------------------------------------------------

elif page == "3 · Trends":
    st.markdown("## 3 · Trends in the annual means")
    st.markdown("<p class='small'>Ordinary least squares slope with its 95% interval and p-value, the Mann-Kendall test, and Sen's slope with its interval. "
                "Years with less than 80% coverage are dropped; records shorter than ten years are shown without a trend.</p>", unsafe_allow_html=True)
    names = records_with(variable)
    if not names:
        st.info("Load data first.")
    else:
        long = [n for n in names if S.records[n]["info"]["span_years"] >= 10] or names
        pick = st.multiselect("Records", names, default=long)
        trends, rows = {}, []
        for n in pick:
            s = series(n, variable)
            tr = A.trend_analysis(s)
            am = tr["annual_means"] if tr else A.annual_means(s)
            trends[n] = {"trend": tr, "annual": am}
            row = {"record": n, "years": len(am), "mean": round(float(s.mean()), 3)}
            if tr:
                o, k, se = tr["ols"], tr["mk"], tr["sen"]
                f = 1000 if variable == "swh" else 10
                u = "cm/decade" if variable == "swh" else "s/decade"
                row.update({f"OLS slope [{u}]": round(o["slope"] * f, 3), "OLS 95% CI": f"[{o['slope_ci'][0] * f:+.3f}, {o['slope_ci'][1] * f:+.3f}]",
                            "OLS p": round(o["p_value"], 3), "R²": round(o["r2"], 3), f"Sen slope [{u}]": round(se["slope"] * f, 3),
                            "Sen 95% CI": f"[{se['slope_ci'][0] * f:+.3f}, {se['slope_ci'][1] * f:+.3f}]", "Mann-Kendall p": round(k["p_value"], 3),
                            f"change over record [{UNIT[variable]}]": round(tr["change_over_record"]["ols"], 3)})
            rows.append(row)
        table = pd.DataFrame(rows)
        S.trend_table = table
        keep("trends", table=table)
        fig = C.annual_means(trends, variable)
        st.plotly_chart(fig, width="stretch"); keep(f"annual_means_{variable}", fig)
        st.dataframe(table, hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# 4 · Extremes
# ---------------------------------------------------------------------------

elif page == "4 · Extremes":
    st.markdown("## 4 · Extremes: return levels")
    st.markdown("<p class='small'>Peaks over threshold: the threshold is a percentile of the record; exceedances separated by more than the storm gap are "
                "independent storms, each kept as one peak; a Generalized Pareto Distribution is fitted by maximum likelihood with the location fixed at the "
                "threshold; return levels use the storm rate, with bootstrap 95% intervals. The non-stationary section standardises the record by its running "
                "mean and standard deviation first and maps the return levels back to each year.</p>", unsafe_allow_html=True)
    names = records_with(variable)
    if not names:
        st.info("Load data first.")
    else:
        c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1.3])
        target = c1.selectbox("Record", names)
        pct = c2.number_input("Threshold percentile", 90.0, 99.9, 99.5, 0.1, format="%.1f")
        gap = c3.number_input("Storm gap (h)", 0, 240, 72, 6)
        n_boot = c4.number_input("Bootstrap samples", 100, 5000, 500, 100)
        rp_text = c5.text_input("Return periods (yr)", "2, 5, 10, 25, 50, 100, 200")
        s = series(target, variable)
        span = S.records[target]["info"]["span_years"]
        if span < 10:
            st.warning(f"This record spans {span:.1f} years. Return levels from fewer than about ten years carry wide uncertainty and are not design values.")
        if st.button("Run stationary analysis", type="primary"):
            try:
                rps = tuple(sorted({float(x) for x in rp_text.replace(";", ",").split(",") if x.strip()}))
                with st.spinner("Fitting..."):
                    S.pot[(target, variable)] = A.pot_analysis(s, pct, gap, return_periods=rps, n_boot=int(n_boot))
            except ValueError as e:
                st.error(str(e))
        res = S.pot.get((target, variable))
        if res is not None:
            u = UNIT[variable]
            m = st.columns(6)
            m[0].metric("Threshold", f"{res['threshold']:.2f} {u}")
            m[1].metric("Storm peaks", res["n_peaks"], help=f"{res['n_raw_exceedances']} raw exceedances before declustering")
            m[2].metric("Storms per year", f"{res['rate_per_year']:.2f}")
            m[3].metric("Shape ξ", f"{res['xi']:+.3f}", help=f"bootstrap 95%: [{res['xi_ci'][0]:+.2f}, {res['xi_ci'][1]:+.2f}]")
            m[4].metric("Scale β", f"{res['beta']:.3f} {u}")
            m[5].metric("Upper bound", "none (ξ ≥ 0)" if not np.isfinite(res["upper_bound"]) else f"{res['upper_bound']:.1f} {u}")
            tbl = A.return_level_table(res).rename(columns={"return_period_years": "return period (yr)", "return_level": f"return level [{u}]",
                                                             "ci_low": "95% low", "ci_high": "95% high"})
            keep(f"return_levels_{target}_{variable}", table=tbl)
            l, r = st.columns([1, 1.4])
            with l:
                st.dataframe(tbl, hide_index=True, width="stretch")
            with r:
                compare = {n: rr for (n, v), rr in S.pot.items() if v == variable}
                fig = C.return_levels(compare, variable); st.plotly_chart(fig, width="stretch"); keep(f"return_levels_{variable}", fig)
            l, r = st.columns(2)
            with l:
                fig = C.peaks(s, res, target, variable); st.plotly_chart(fig, width="stretch"); keep(f"peaks_{target}_{variable}", fig)
                fig = C.gpd_fit(res, target, variable); st.plotly_chart(fig, width="stretch"); keep(f"gpd_{target}_{variable}", fig)
            with r:
                fig = C.qq(res, target, variable); st.plotly_chart(fig, width="stretch"); keep(f"qq_{target}_{variable}", fig)
                with st.container():
                    st.markdown("**Threshold diagnostics**")
                    st.caption("Pick the lowest threshold above which the mean excess is about linear and the shape and modified scale are about constant.")
                    if st.button("Compute diagnostics"):
                        with st.spinner("Computing..."):
                            S.diag[(target, variable)] = (A.mean_residual_life(s, np.arange(90, 99.91, 0.25)),
                                                          A.parameter_stability(s, np.arange(97.0, 99.91, 0.2), gap))
                    if (target, variable) in S.diag:
                        mrl, ps = S.diag[(target, variable)]
                        fig = C.diagnostics(mrl, ps, res["threshold"], variable); st.plotly_chart(fig, width="stretch"); keep(f"diagnostics_{target}_{variable}", fig)
            with st.expander("Sensitivity of the 100-year level to threshold and storm gap"):
                if st.button("Compute sensitivity grid"):
                    with st.spinner("Computing nine fits..."):
                        S.sens[(target, variable)] = A.sensitivity_grid(s, (99.0, 99.5, 99.8), (24, 48, 72), T=100)
                if (target, variable) in S.sens:
                    g = S.sens[(target, variable)].round(3)
                    st.dataframe(g, hide_index=True, width="stretch"); keep(f"sensitivity_{target}_{variable}", table=g)

            st.markdown("#### Non-stationary return levels (transformed-stationary approach)")
            st.caption("The record is standardised by a running mean and standard deviation; the stationary analysis runs on the standardised series; "
                       "the return levels are mapped back year by year. The window sets how slowly the mean and spread may change.")
            c1, c2 = st.columns([1, 3])
            window = c1.number_input("Window (years)", 5, 60, 30, 5)
            if span < 1.5 * window:
                c2.info(f"The record spans {span:.0f} years; a {window}-year window leaves little room for the running statistics to vary. Consider a shorter window.")
            if st.button("Run non-stationary analysis"):
                try:
                    rps = tuple(sorted({float(x) for x in rp_text.replace(";", ",").split(",") if x.strip()}))
                    with st.spinner("Transforming and fitting..."):
                        S.ts[(target, variable)] = P.ts_eva(s, window_years=int(window), percentile=pct, min_gap_hours=gap,
                                                            return_periods=rps, n_boot=int(n_boot))
                except ValueError as e:
                    st.error(str(e))
            ts = S.ts.get((target, variable))
            if ts is not None:
                T = 100 if 100 in ts["result_y"]["return_periods"] else int(ts["result_y"]["return_periods"][-1])
                stationary_rl = float(A.return_level(res["threshold"], res["xi"], res["beta"], res["rate_per_year"], T))
                sm = P.ts_eva_summary(ts, stationary_rl, T)
                m = st.columns(5)
                m[0].metric(f"{T}-yr level, {sm['first_year']}", f"{sm['rl_first']:.2f} {u}")
                m[1].metric(f"{T}-yr level, {sm['last_year']}", f"{sm['rl_last']:.2f} {u}", delta=f"{sm['change']:+.2f} {u}", delta_color="off")
                m[2].metric("Lowest to highest", f"{sm['rl_min']:.2f}–{sm['rl_max']:.2f}")
                m[3].metric("Stationary level", f"{sm['stationary']:.2f} {u}")
                m[4].metric("Shape ξ (standardised)", f"{ts['result_y']['xi']:+.3f}")
                fig = C.ts_eva(ts, stationary_rl, target, variable, T); st.plotly_chart(fig, width="stretch"); keep(f"ts_eva_{target}_{variable}", fig)
                fig = C.ts_moments(ts, target, variable); st.plotly_chart(fig, width="stretch"); keep(f"ts_moments_{target}_{variable}", fig)
                keep(f"ts_table_{target}_{variable}", table=ts["table"].round(3).reset_index())

# ---------------------------------------------------------------------------
# 5 · Report
# ---------------------------------------------------------------------------

elif page == "5 · Report":
    st.markdown("## 5 · Report")
    if not S.records:
        st.info("Nothing to report yet.")
    else:
        st.markdown("#### Session summary")
        for k, t in S.tables.items():
            st.markdown(f"**{k.replace('_', ' ')}**")
            st.dataframe(t, hide_index=True, width="stretch")
        st.markdown(f"<p class='small'>{len(S.figs)} figures and {len(S.tables)} tables in this session.</p>", unsafe_allow_html=True)

        def build_html():
            parts = ["<html><head><meta charset='utf-8'><title>Wave Climate Trend-Extreme Tool report</title>",
                     "<style>body{font-family:Inter,Segoe UI,Helvetica,Arial,sans-serif;max-width:1100px;margin:2rem auto;color:#1f2933}"
                     "table{border-collapse:collapse;font-size:0.9rem;margin:0.5rem 0 1.5rem 0}td,th{border:1px solid #e3e8ee;padding:4px 8px}"
                     "th{background:#f3f6fa}h2{margin-top:2rem;color:#0b5394}</style></head><body>",
                     f"<h1>Wave Climate Trend-Extreme Tool</h1><p>Report generated {datetime.now():%Y-%m-%d %H:%M}.</p>",
                     "<h2>Records</h2><ul>" + "".join(f"<li><b>{n}</b>: {r['role']}, {r['source']}, {r['info']['start'].date()} to {r['info']['end'].date()}</li>" for n, r in S.records.items()) + "</ul>"]
            for k, t in S.tables.items():
                parts.append(f"<h2>{k.replace('_', ' ')}</h2>" + t.to_html(index=False, border=0))
            first = True
            for k, fig in S.figs.items():
                parts.append(f"<h2>{k.replace('_', ' ')}</h2>" + fig.to_html(include_plotlyjs="cdn" if first else False, full_html=False))
                first = False
            parts.append("<h2>Log</h2><pre>" + "\n".join(S.log) + "</pre></body></html>")
            return "\n".join(parts)

        def build_zip():
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for k, t in S.tables.items():
                    zf.writestr(f"tables/{k}.csv", t.to_csv(index=False))
                for n, r in S.records.items():
                    if r["role"] == "projected":
                        zf.writestr(f"projected/{n.replace(' ', '_').replace('·', '-')}.csv", r["df"].to_csv(index=False))
                for (n, v), res in S.pot.items():
                    zf.writestr(f"storm_peaks/{n}_{v}.csv", res["peaks"].rename(v).to_csv())
                for k, fig in S.figs.items():
                    zf.writestr(f"figures/{k}.html", fig.to_html(include_plotlyjs="cdn", full_html=True))
                zf.writestr("log.txt", "\n".join(S.log))
            return buf.getvalue()

        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        c1, c2 = st.columns(2)
        c1.download_button("⬇️ HTML report (interactive figures)", data=build_html(), file_name=f"wave_climate_report_{stamp}.html", mime="text/html", type="primary")
        c2.download_button("⬇️ ZIP of tables, series and figures", data=build_zip(), file_name=f"wave_climate_session_{stamp}.zip", mime="application/zip")

st.markdown("<hr style='margin-top:2rem;border-color:#e3e8ee'><p class='small' style='text-align:center'>Wave Climate Trend-Extreme Tool · "
            "Amirhussein Shahbazbegian and Erfan Amini · MIT License</p>", unsafe_allow_html=True)
