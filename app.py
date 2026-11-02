"""
Wave Climate Trend-Extreme Tool

Bias correction, trends and extreme-value analysis of wave climate records
and projections in one guided workflow.

Run locally:  streamlit run app.py
"""

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
import projection as P

st.set_page_config(page_title="Wave Climate Trend-Extreme Tool", page_icon="🌊", layout="wide",
                   initial_sidebar_state="collapsed")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROLES = ["observed", "reference", "model historical", "model future", "projected"]
VARNAME = {"swh": "Wave height", "tm": "Wave period"}
UNIT = {"swh": "m", "tm": "s"}
PAGES = {"data": (":material/database:", "Data"), "projection": (":material/tune:", "Projection"),
         "trends": (":material/trending_up:", "Trends"), "extremes": (":material/waves:", "Extremes"),
         "report": (":material/description:", "Report"), "about": (":material/groups:", "Team")}

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
  /* landing tiles */
  .st-key-tiles button {height: 128px; border-radius: 18px; border: 1px solid #e3e8ee; background: #fff; font-size: 1.08rem; font-weight: 600;
      color: #0b5394; transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease; animation: fadeUp .7s ease both;}
  .st-key-tiles button:hover {transform: translateY(-5px); box-shadow: 0 14px 30px rgba(11,83,148,.18); border-color: #0b5394;}
  .st-key-tiles button p {font-size: 1.08rem;}
  .st-key-tiles [data-testid="stCaptionContainer"] {text-align: center;}
  .st-key-actions button {border-radius: 999px; padding: .55rem 1.3rem; font-weight: 600; transition: transform .15s ease;}
  .st-key-actions button:hover {transform: translateY(-2px);}
  /* top bar */
  div[data-testid="stLayoutWrapper"]:has(> .st-key-topbar) {position: sticky; top: 3.75rem; z-index: 990; background: #ffffff;}
  .st-key-topbar { background: #ffffff;
      border-bottom: 1px solid #e9eef3; padding: .35rem 0 .3rem 0; margin-bottom: .6rem;}
  .st-key-topbar button {border-radius: 999px;}
  .chip {display:inline-block; padding: .18rem .62rem; border-radius: 999px; background: #eef4fa; color: #0b5394; font-size: .78rem; font-weight: 600; margin-right: .35rem;}
  .chip.off {background: #f1f3f5; color: #9aa5b1; font-weight: 500;}
  .banner {position: relative; border-radius: 18px; overflow: hidden; height: 190px; margin-bottom: 1rem; background-size: cover; background-position: center;}
  .banner .ov {position:absolute; inset:0; background: linear-gradient(90deg, rgba(5,38,70,.85), rgba(5,38,70,.25));}
  .banner h1 {position:absolute; left: 1.6rem; bottom: 1.1rem; color: #fff; margin: 0; font-size: 2rem; text-shadow: 0 2px 12px rgba(0,0,0,.4);}
  .banner .credit {position:absolute; right: .9rem; bottom: .6rem; color: rgba(255,255,255,.75); font-size: .72rem;}
  .banner .credit a {color: rgba(255,255,255,.9);}
  .card {border: 1px solid #e3e8ee; border-radius: 16px; padding: 1.1rem 1.2rem; background: #fff; height: 100%; animation: fadeUp .7s ease both;}
  .card h3 {margin: .2rem 0 .1rem 0; font-size: 1.15rem;}
  .card .role {color: #0b5394; font-weight: 600; font-size: .9rem;}
  .card p {margin: .35rem 0 0 0; color: #3e4c59; font-size: .9rem;}
  .card a {margin-right: .6rem; font-size: .88rem;}
  .avatar {width: 64px; height: 64px; border-radius: 50%; background: linear-gradient(135deg,#0b5394,#1a9a9a); color: #fff; display:flex; align-items:center;
      justify-content:center; font-weight: 700; font-size: 1.35rem; margin-bottom: .5rem; animation: floaty 4s ease-in-out infinite;}
  div[data-testid="stMetric"] {background: #f8fafc; border: 1px solid #e3e8ee; border-radius: 12px; padding: .55rem .8rem;}
  .small {font-size: .85rem; color: #52606d;}
  footer, #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def init_state():
    defaults = {"page": "home", "variable": "swh", "records": {}, "pot": {}, "ts": {}, "proj_validation": None, "proj_series": None,
                "trend_table": None, "figs": {}, "tables": {}, "diag": {}, "sens": {}, "log": []}
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


def load_demo():
    for entry in F.load_catalogue()["stations"]:
        load_catalogue_station(entry)


def page_header(key, caption, how=None, photo=None):
    icon, label = PAGES[key]
    if photo:
        url, by, link = PHOTOS[photo]
        st.markdown(f"<div class='banner fade' style=\"background-image:url('{url}')\"><div class='ov'></div><h1>{label}</h1>"
                    f"<div class='credit'>Photo: <a href='{link}' target='_blank'>{by}</a> / Unsplash</div></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"## {icon} {label}")
    if caption:
        st.caption(caption)
    if how:
        with st.expander("How this works"):
            st.markdown(how)


def topbar():
    with st.container(key="topbar"):
        c1, c2, c3 = st.columns([0.9, 6, 2.2])
        with c1:
            if st.button(":material/home: Home", key="home_btn"):
                go("home")
        with c2:
            keys = list(PAGES)
            pick = st.segmented_control("Navigate", keys, format_func=lambda k: f"{PAGES[k][0]} {PAGES[k][1]}", default=S.page if S.page in keys else None,
                                        label_visibility="collapsed", key=f"nav_{S.page}")
            if pick and pick != S.page:
                go(pick)
        with c3:
            v = st.segmented_control("Variable", ["swh", "tm"], format_func=lambda k: VARNAME[k], default=S.variable, label_visibility="collapsed", key=f"var_{S.page}")
            if v and v != S.variable:
                S.variable = v
                st.rerun()
        global CHIPS
        CHIPS = st.empty()   # filled at the end of the run, so it shows this run's results


variable = S.variable
CHIPS = None

# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------

if S.page == "home":
    url, by, link = PHOTOS["hero"]
    components.html(f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
      html, body {{margin:0; padding:0; font-family: Inter, 'Segoe UI', Helvetica, Arial, sans-serif;}}
      .hero {{position: relative; height: 440px; border-radius: 22px; overflow: hidden; background: #062a4a url('{url}') center/cover;}}
      .hero .ov {{position:absolute; inset:0; background: linear-gradient(180deg, rgba(4,30,56,.35) 0%, rgba(4,30,56,.78) 100%);}}
      canvas {{position:absolute; left:0; bottom:0; width:100%; height:100%;}}
      .txt {{position:absolute; left: 7%; top: 20%; color: #fff; max-width: 70%;}}
      .txt h1 {{font-size: 3rem; font-weight: 800; margin: 0 0 .5rem 0; letter-spacing: -.02em; line-height: 1.05; animation: up .9s ease both; text-shadow: 0 4px 24px rgba(0,0,0,.35);}}
      .txt p {{font-size: 1.25rem; margin: 0; opacity: .95; animation: up .9s .25s ease both; text-shadow: 0 2px 12px rgba(0,0,0,.35);}}
      .tags {{margin-top: 1.1rem; animation: up .9s .45s ease both;}}
      .tags span {{display:inline-block; margin: 0 .4rem .4rem 0; padding: .3rem .75rem; border-radius: 999px; background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.35); font-size: .85rem; backdrop-filter: blur(4px);}}
      .credit {{position:absolute; right: 1rem; bottom: .7rem; color: rgba(255,255,255,.7); font-size: .72rem;}}
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
    """, height=460)

    with st.container(key="actions"):
        a1, a2, a3, a4 = st.columns([1.6, 1.4, 1.2, 4])
        if a1.button(":material/play_arrow: Try it with demo data", type="primary"):
            load_demo(); go("extremes")
        if a2.button(":material/upload: Use my own data"):
            go("data")
        if a3.button(":material/groups: The team"):
            go("about")

    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
    with st.container(key="tiles"):
        cols = st.columns(5)
        tiles = [("data", "Load records: catalogue, CSV, buoys, Copernicus"), ("projection", "Correct a model's bias and validate it"),
                 ("trends", "Annual means with three trend tests"), ("extremes", "Return levels with intervals and diagnostics"),
                 ("report", "Everything in one report")]
        for i, (col, (key, cap)) in enumerate(zip(cols, tiles)):
            with col:
                st.markdown(f"<style>.st-key-tiles div[data-testid='column']:nth-child({i + 1}) button{{animation-delay:{0.1 * i + 0.1:.2f}s}}</style>", unsafe_allow_html=True)
                if st.button(f"{PAGES[key][0]}  {i + 1} · {PAGES[key][1]}", key=f"tile_{key}", width="stretch"):
                    go(key)
                st.caption(cap)

    st.markdown("<p class='small fade d5' style='text-align:center; margin-top:1.2rem'>Shahbazbegian and Amini · Columbia University and University of Tehran · MIT license</p>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

elif S.page == "data":
    topbar()
    page_header("data", "Load records and give each one a role. Roles drive the next steps.",
                "Each record needs a time column and a wave height and/or period column. Column names are matched against common variants; "
                "missing-value codes become NaN. Roles: **observed** or **reference** (the truth for bias correction), **model historical**, "
                "**model future**, and **projected** (made in the Projection step).", photo="buoy")
    t_cat, t_up, t_ndbc, t_cds = st.tabs([":material/public: Catalogue", ":material/upload_file: Upload", ":material/sailing: NOAA buoy", ":material/cloud_download: Copernicus"])

    with t_cat:
        cat = F.load_catalogue()
        labels = {e["label"]: e for e in cat["stations"]}
        pick = st.multiselect("Stations", list(labels), default=list(labels)[:1], label_visibility="collapsed")
        if st.button(":material/download: Load stations", type="primary"):
            for lab in pick:
                load_catalogue_station(labels[lab])
            st.toast(f"{len(S.records)} records in the session", icon="✅")
            st.rerun()

    with t_up:
        ups = st.file_uploader("CSV or gzip CSV", type=["csv", "gz"], accept_multiple_files=True, label_visibility="collapsed")
        role_up = st.selectbox("Role", ROLES, index=0, help="observed or reference for the truth; model historical and model future for climate-model runs")
        if ups and st.button(":material/download: Load files"):
            for f in ups:
                try:
                    df, info = A.load_series(f)
                    name = os.path.splitext(os.path.splitext(f.name)[0])[0]
                    add_record(name, df, info, role_up, "upload")
                    st.toast(f"{name}: {info['n_rows']:,} rows", icon="✅")
                except Exception as e:
                    st.error(f"{f.name}: {e}")

    with t_ndbc:
        c1, c2, c3 = st.columns([1, 1, 1])
        station = c1.text_input("Station ID", "44025", help="Any NDBC station, e.g. 44025 Long Island, 46042 Monterey Bay, 51001 Hawaii")
        years = c2.text_input("Years", "2015-2024", help="A range such as 2015-2024 or a list such as 2016,2018")
        role_nd = c3.selectbox("Role", ROLES, index=0, key="role_ndbc")
        if st.button(":material/download: Fetch buoy record", type="primary"):
            try:
                with st.spinner("Downloading from NDBC..."):
                    df, notes = F.fetch_ndbc(station, F.years_from_spec(years))
                buf = io.StringIO(); df.to_csv(buf, index=False); buf.seek(0); buf.name = "ndbc.csv"
                df2, info = A.load_series(buf)
                add_record(f"NDBC {station} {years}", df2, info, role_nd, "NOAA NDBC")
                st.success(f"{len(df2):,} rows. " + "; ".join(notes))
            except Exception as e:
                st.error(f"NDBC fetch failed: {e}")

    with t_cds:
        st.caption("Needs your own CDS key (free account). The key is used for this request only. ERA5 takes minutes per year; the C3S product can take much longer.")
        key = st.text_input("CDS API key", type="password")
        c1, c2 = st.columns(2)
        lat = c1.number_input("Latitude", -90.0, 90.0, 40.5, 0.1)
        lon = c2.number_input("Longitude", -180.0, 180.0, 2.0, 0.1)
        which = st.radio("Product", ["ERA5 at the point", "C3S coastal station"], horizontal=True, label_visibility="collapsed")
        if which.startswith("ERA5"):
            yrs = st.text_input("Years", "1995-2005", key="era5_years")
            role_cds = st.selectbox("Role", ROLES, index=1, key="role_era5")
            if st.button(":material/download: Request ERA5", type="primary"):
                try:
                    prog = st.progress(0.0, "submitting")
                    df, point = F.fetch_era5_point(lat, lon, F.years_from_spec(yrs), key=key or None, progress=lambda f, m: prog.progress(f, m))
                    buf = io.StringIO(); df.to_csv(buf, index=False); buf.seek(0); buf.name = "era5.csv"
                    df2, info = A.load_series(buf)
                    add_record(f"ERA5 {point} {yrs}", df2, info, role_cds, "Copernicus ERA5")
                    st.success(f"{len(df2):,} rows at {point}")
                except Exception as e:
                    st.error(f"CDS request failed: {e}")
        else:
            exp = st.selectbox("Experiment", list(F.C3S_REQUEST["experiment"]), format_func=lambda k: {"historical": "Historical 1976-2005", "reanalysis": "ERA5-forced 2001-2017", "rcp45": "RCP4.5 2041-2100", "rcp85": "RCP8.5 2041-2100"}[k])
            role_c3s = st.selectbox("Role", ROLES, index=2 if exp == "historical" else 3, key="role_c3s")
            if st.button(":material/download: Request C3S station", type="primary"):
                try:
                    prog = st.progress(0.0, "submitting")
                    df, where = F.fetch_c3s_station(lat, lon, exp, key=key or None, progress=lambda f, m: prog.progress(f, m))
                    buf = io.StringIO(); df.to_csv(buf, index=False); buf.seek(0); buf.name = "c3s.csv"
                    df2, info = A.load_series(buf)
                    add_record(f"C3S {exp} {where}", df2, info, role_c3s, "Copernicus C3S")
                    st.success(f"{len(df2):,} rows at station {where}")
                except Exception as e:
                    st.error(f"CDS request failed: {e}")

    if not S.records:
        st.info("No records yet. Pick a station above, or go back home and try the demo.", icon=":material/info:")
    else:
        st.markdown("#### Records")
        rows = []
        for n, r in S.records.items():
            info, df = r["info"], r["df"]
            row = {"record": n, "role": r["role"], "source": r["source"], "start": info["start"].date(), "end": info["end"].date(),
                   "years": round(info["span_years"], 2), "step (h)": info["step_hours"], "rows": info["n_rows"]}
            if variable in df.columns:
                st_ = A.summary_stats(A.to_series(df, variable))
                row.update({f"mean [{UNIT[variable]}]": round(st_["mean"], 3), f"p99 [{UNIT[variable]}]": round(st_["p99"], 3), f"max [{UNIT[variable]}]": round(st_["max"], 3)})
            rows.append(row)
        table = pd.DataFrame(rows)
        edited = st.data_editor(table, hide_index=True, width="stretch", disabled=[c for c in table.columns if c != "role"],
                                column_config={"role": st.column_config.SelectboxColumn("role", options=ROLES, required=True, help="Click to change")})
        for _, row in edited.iterrows():
            S.records[row["record"]]["role"] = row["role"]
        keep("records", table=table)
        with st.expander("Remove records"):
            rm = st.multiselect("Records to remove", list(S.records), label_visibility="collapsed")
            if rm and st.button(":material/delete: Remove"):
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
                pick = st.selectbox("Preview", names, label_visibility="collapsed")
                fig = C.record(series(pick, variable), pick, variable)
                st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

elif S.page == "projection":
    topbar()
    page_header("projection", "Correct a model against a reference, test it on held-out years, then project.",
                "**Delta change** applies the model's change in the monthly mean to the reference record. **Quantile delta mapping** (Cannon et al. 2015) "
                "matches every quantile of the model to the reference and preserves the model's projected change at each quantile, so the extremes keep the "
                "climate signal. Both are fitted by calendar month. The split-sample test fits on half the overlap years and scores raw model, delta and QDM "
                "on the other half.", photo="beach")
    refs = records_with(variable, ["observed", "reference"])
    hists = records_with(variable, ["model historical"])
    futs = records_with(variable, ["model future"])
    if not hists:
        st.warning("Needs a model historical run, a reference over the same years, and model future runs. Load them in Data.", icon=":material/warning:")
    else:
        with st.expander("No reference that overlaps the model run? Build a synthetic pair to try the interface"):
            st.caption("Uses the first twenty years of the longest record as a synthetic model run and a distorted copy (×1.2 + 0.1) as reference. Labelled as such everywhere; not data.")
            if st.button(":material/science: Build synthetic pair"):
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
            st.warning("No reference record (role observed or reference).", icon=":material/warning:")
    if refs and hists:
        c1, c2, c3 = st.columns(3)
        hist_name = c2.selectbox("Model historical run", hists, index=len(hists) - 1)
        hy = set(series(hist_name, variable).index.year)
        overlapping = [r for r in refs if set(series(r, variable).index.year) & hy]
        ref_name = c1.selectbox("Reference record", refs, index=refs.index(overlapping[0]) if overlapping else 0)
        fut_names = c3.multiselect("Model future runs", futs, default=futs)
        c4, c5 = st.columns(2)
        kind = c4.segmented_control("Correction", ["multiplicative", "additive"], default="multiplicative",
                                    help="Multiplicative scales values (natural for wave height); additive shifts them.")
        mode = c5.segmented_control("Split", ["alternate", "halves"], default="alternate",
                                    help="alternate: odd years calibrate, even validate; halves: first half calibrates")
        ref_s, hist_s = series(ref_name, variable), series(hist_name, variable)
        overlap = sorted(set(ref_s.index.year) & set(hist_s.index.year))
        if len(overlap) < 2:
            st.error(f"Reference ({ref_s.index.year.min()} to {ref_s.index.year.max()}) and model run ({hist_s.index.year.min()} to {hist_s.index.year.max()}) share fewer than two years.")
        else:
            st.caption(f"Overlap {overlap[0]} to {overlap[-1]} ({len(overlap)} years).")
            ref_o = ref_s[np.isin(ref_s.index.year, overlap)]
            hist_o = hist_s[np.isin(hist_s.index.year, overlap)]
            if st.button(":material/fact_check: Validate on held-out years", type="primary"):
                with st.spinner("Fitting and scoring..."):
                    tab, ser, info = P.validate_corrections(ref_o, hist_o, kind=kind or "multiplicative", mode=mode or "alternate")
                S.proj_validation = {"table": tab, "series": ser, "info": info, "variable": variable, "ref": ref_name, "hist": hist_name, "kind": kind}
            if S.proj_validation is not None and S.proj_validation["variable"] == variable:
                v = S.proj_validation
                st.caption(f"{len(v['info']['calibration_years'])} calibration years, {len(v['info']['validation_years'])} validation years. Metrics in {UNIT[variable]}; closer to zero is better, std ratio closer to one.")
                tab = v["table"].rename(columns={"mean_bias": "mean bias", "std_ratio": "std ratio", "seasonal_rmse": "seasonal RMSE", "quantile_rmse": "quantile RMSE",
                                                 "p99_bias": "p99 bias", "p995_bias": "p99.5 bias", "annual_max_bias": "annual max bias"}).drop(columns=["n"])
                st.dataframe(tab.style.format("{:.3f}").background_gradient(cmap="RdYlGn_r", subset=["quantile RMSE", "seasonal RMSE"]), width="stretch")
                keep("validation", table=tab.reset_index())
                l, r = st.columns(2)
                with l:
                    fig = C.validation_quantiles(v["series"], variable); st.plotly_chart(fig, width="stretch"); keep("validation_quantiles", fig)
                with r:
                    fig = C.validation_seasonal(v["series"], variable); st.plotly_chart(fig, width="stretch"); keep("validation_seasonal", fig)
            methods = st.multiselect("Methods to project", ["raw", "delta", "QDM"], default=["raw", "delta", "QDM"])
            if fut_names and st.button(":material/rocket_launch: Produce projected series", type="primary"):
                out = {}
                with st.spinner("Projecting..."):
                    for fn in fut_names:
                        fut_s = series(fn, variable)
                        for m in methods:
                            proj = P.project(ref_o, hist_o, fut_s, method=m, kind=kind or "multiplicative")
                            label = f"{fn} · {m}"
                            df = pd.DataFrame({"time": proj.index, variable: proj.values})
                            info = {"columns": {variable: m}, "n_rows": len(df), "start": df.time.min(), "end": df.time.max(),
                                    "step_hours": S.records[fn]["info"]["step_hours"], "duplicates_removed": 0, "span_years": A.record_length_years(df.time)}
                            add_record(label, df, info, "projected", f"{m} correction of {fn} against {ref_name}")
                            out[label] = proj
                S.proj_series = {"variable": variable, "names": list(out)}
                st.toast(f"{len(out)} projected series added", icon="✅")
            if S.proj_series and S.proj_series["variable"] == variable:
                names = [n for n in S.proj_series["names"] if n in S.records]
                if names:
                    fig = C.projection_compare({n: series(n, variable) for n in names}, variable)
                    st.plotly_chart(fig, width="stretch"); keep("projection_compare", fig)

# ---------------------------------------------------------------------------
# Trends
# ---------------------------------------------------------------------------

elif S.page == "trends":
    topbar()
    page_header("trends", "Annual means with an OLS trend, the Mann-Kendall test and Sen's slope.",
                "Years with less than 80% coverage are dropped. Records shorter than ten years get no trend. OLS gives the slope with a 95% interval and p-value; "
                "Mann-Kendall tests for a monotonic trend without assuming normality; Sen's slope is the median of pairwise slopes.", photo="rocks")
    names = records_with(variable)
    if not names:
        st.info("Load data first.", icon=":material/info:")
    else:
        long = [n for n in names if S.records[n]["info"]["span_years"] >= 10] or names
        pick = st.multiselect("Records", names, default=long, label_visibility="collapsed")
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
# Extremes
# ---------------------------------------------------------------------------

elif S.page == "extremes":
    topbar()
    page_header("extremes", "Storm peaks over a threshold, a GPD fit, return levels with intervals, and their change through time.",
                "Threshold = a percentile of the record. Exceedances separated by more than the storm gap are independent storms; each keeps its peak. "
                "A Generalized Pareto Distribution is fitted by maximum likelihood with the location fixed at the threshold. Return levels use the storm rate; "
                "intervals come from a bootstrap of the peaks. The non-stationary fit standardises the record by its running mean and standard deviation, "
                "fits once, and maps the levels back to each year (Mentaschi et al. 2016).", photo="storm")
    names = records_with(variable)
    if not names:
        st.info("Load data first.", icon=":material/info:")
    else:
        c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1.3])
        longest = max(names, key=lambda n: S.records[n]["info"]["span_years"])
        target = c1.selectbox("Record", names, index=names.index(longest))
        pct = c2.number_input("Threshold percentile", 90.0, 99.9, 99.5, 0.1, format="%.1f", help="Percentile of the whole record used as threshold")
        gap = c3.number_input("Storm gap (h)", 0, 240, 72, 6, help="Minimum gap between independent storms")
        n_boot = c4.number_input("Bootstrap samples", 100, 5000, 500, 100)
        rp_text = c5.text_input("Return periods (yr)", "2, 5, 10, 25, 50, 100, 200")
        s = series(target, variable)
        span = S.records[target]["info"]["span_years"]
        if span < 10:
            st.warning(f"This record spans {span:.1f} years. Return levels from under ten years are not design values.", icon=":material/warning:")
        if st.button(":material/play_arrow: Run stationary analysis", type="primary"):
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
            tbl = A.return_level_table(res).rename(columns={"return_period_years": "return period (yr)", "return_level": f"return level [{u}]", "ci_low": "95% low", "ci_high": "95% high"})
            keep(f"return_levels_{target}_{variable}", table=tbl)
            l, r = st.columns([1, 1.4])
            with l:
                st.dataframe(tbl, hide_index=True, width="stretch")
            with r:
                compare = {n: rr for (n, v), rr in S.pot.items() if v == variable}
                fig = C.return_levels(compare, variable); st.plotly_chart(fig, width="stretch"); keep(f"return_levels_{variable}", fig)
            t_fit, t_diag, t_sens, t_ns = st.tabs([":material/show_chart: Fit", ":material/troubleshoot: Threshold diagnostics", ":material/grid_on: Sensitivity", ":material/timeline: Non-stationary"])
            with t_fit:
                l, r = st.columns(2)
                with l:
                    fig = C.peaks(s, res, target, variable); st.plotly_chart(fig, width="stretch"); keep(f"peaks_{target}_{variable}", fig)
                    fig = C.gpd_fit(res, target, variable); st.plotly_chart(fig, width="stretch"); keep(f"gpd_{target}_{variable}", fig)
                with r:
                    fig = C.qq(res, target, variable); st.plotly_chart(fig, width="stretch"); keep(f"qq_{target}_{variable}", fig)
            with t_diag:
                st.caption("Pick the lowest threshold above which the mean excess is about linear and the shape and modified scale are about constant.")
                if st.button(":material/troubleshoot: Compute diagnostics"):
                    with st.spinner("Computing..."):
                        S.diag[(target, variable)] = (A.mean_residual_life(s, np.arange(90, 99.91, 0.25)), A.parameter_stability(s, np.arange(97.0, 99.91, 0.2), gap))
                if (target, variable) in S.diag:
                    mrl, ps = S.diag[(target, variable)]
                    fig = C.diagnostics(mrl, ps, res["threshold"], variable); st.plotly_chart(fig, width="stretch"); keep(f"diagnostics_{target}_{variable}", fig)
            with t_sens:
                st.caption("The 100-year level over nine threshold and storm-gap settings.")
                if st.button(":material/grid_on: Compute grid"):
                    with st.spinner("Nine fits..."):
                        S.sens[(target, variable)] = A.sensitivity_grid(s, (99.0, 99.5, 99.8), (24, 48, 72), T=100)
                if (target, variable) in S.sens:
                    g = S.sens[(target, variable)].round(3)
                    st.dataframe(g, hide_index=True, width="stretch"); keep(f"sensitivity_{target}_{variable}", table=g)
            with t_ns:
                c1, c2 = st.columns([1, 3])
                window = c1.number_input("Window (years)", 5, 60, 30, 5, help="How slowly the running mean and spread may change")
                if span < 1.5 * window:
                    c2.info(f"The record spans {span:.0f} years; consider a shorter window.", icon=":material/info:")
                if st.button(":material/timeline: Run non-stationary analysis", type="primary"):
                    try:
                        rps = tuple(sorted({float(x) for x in rp_text.replace(";", ",").split(",") if x.strip()}))
                        with st.spinner("Transforming and fitting..."):
                            S.ts[(target, variable)] = P.ts_eva(s, window_years=int(window), percentile=pct, min_gap_hours=gap, return_periods=rps, n_boot=int(n_boot))
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
# Report
# ---------------------------------------------------------------------------

elif S.page == "report":
    topbar()
    page_header("report", "Every table and figure of this session, in one interactive HTML file and one ZIP.", photo="hero")
    if not S.records:
        st.info("Nothing to report yet.", icon=":material/info:")
    else:
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
        c1, c2, c3 = st.columns([1.3, 1.3, 3])
        c1.download_button(":material/download: HTML report", data=build_html(), file_name=f"wave_climate_report_{stamp}.html", mime="text/html", type="primary")
        c2.download_button(":material/folder_zip: ZIP of everything", data=build_zip(), file_name=f"wave_climate_session_{stamp}.zip", mime="application/zip")
        st.caption(f"{len(S.figs)} figures and {len(S.tables)} tables.")
        for k, t in S.tables.items():
            with st.expander(k.replace("_", " ")):
                st.dataframe(t, hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# Team
# ---------------------------------------------------------------------------

elif S.page == "about":
    topbar()
    page_header("about", "", photo="aerial")
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
        <p><a href='https://www.linkedin.com/in/erfan-amini/' target='_blank'>LinkedIn</a>
        <a href='https://x.com/erfan_amini_' target='_blank'>X</a>
        <a href='https://github.com/erfan-amini' target='_blank'>GitHub</a></p></div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown("""<div class='card d3'><h3>Methods</h3>
        <p>Quantile delta mapping and delta change (Cannon et al. 2015; Hay et al. 2000) · OLS, Mann-Kendall, Sen's slope · POT-GPD with runs declustering, bootstrap intervals and threshold diagnostics (Coles 2001; Scarrott and MacDonald 2012) · Transformed-stationary EVA (Mentaschi et al. 2016).</p></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown("""<div class='card d4'><h3>Cite</h3>
        <p>Shahbazbegian, A. and Amini, E. <i>A Unified Trend-Extreme Analysis Tool for Wave Climate Projection under Climate Change.</i> Manuscript. Source code:
        <a href='https://github.com/erfan-amini/wave-climate-tool' target='_blank'>github.com/erfan-amini/wave-climate-tool</a>, MIT license.</p></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class='card d5'><h3>Credits</h3>
        <p>Demonstration data: Copernicus Climate Change Service, <i>Ocean surface wave time series for the European coast</i> (Caires and Yan 2020). Generated using Copernicus Climate Change Service information 2020.
        Photos from Unsplash (Unsplash License): {", ".join(f"<a href='{v[2]}' target='_blank'>{v[1]}</a>" for v in PHOTOS.values())}.
        Icons: Material Symbols.</p></div>""", unsafe_allow_html=True)

if CHIPS is not None:
    n = len(S.records)
    chips = [(f"{n} records", n > 0), ("validated", S.proj_validation is not None), ("trends", S.trend_table is not None),
             (f"{len(S.pot)} EVA fits", bool(S.pot)), (f"{len(S.ts)} non-stationary", bool(S.ts))]
    CHIPS.markdown("".join(f"<span class='chip{'' if ok else ' off'}'>{t}</span>" for t, ok in chips), unsafe_allow_html=True)
