"""
Reproduce every figure, table and number in the manuscript from the data
bundled with the Wave Climate Trend-Extreme Tool.

Usage:
    python make_figures.py /path/to/app   # the folder holding analysis.py and data/

Writes figures to images/, LaTeX table bodies to tables/rows.tex, and all
numbers quoted in the text to numbers.tex (as \\N{key}) and numbers.json.

Revision 1 (JMSE): new demonstration station off western Sicily, ERA5 reference
for 1976-2005, the full C3S historical run, quantile delta mapping applied
before the extreme-value analysis, storm directions, a cross-check against
pyextremes, and an illustration of the GPD shape and scale.
"""

import glob as _glob
import json
import os
import re as _re
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import genpareto

warnings.filterwarnings("ignore")
_up = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
APP = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else (
    _up if os.path.exists(os.path.join(_up, "analysis.py")) else os.path.join(_up, "app"))  # repo: reproduce/..
sys.path.insert(0, APP)
import analysis as A  # noqa: E402
import projection as P  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "images")
TAB = os.path.join(HERE, "tables")
os.makedirs(IMG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)
DATA = os.path.join(APP, "data")

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "legend.fontsize": 8, "figure.dpi": 100,
                     "savefig.dpi": 200, "axes.grid": True, "grid.alpha": 0.3})
C = {"ref": "#222222", "hist": "#8a8a8a", "rcp45": "#0b5394", "rcp85": "#c0392b", "bri": "#1a7f5a",
     "raw": "#999999", "delta": "#b7791f", "qdm": "#0b5394"}
LBL = {"ref": "ERA5 1976-2005", "hist": "Model historical 1976-2005", "rcp45": "RCP4.5", "rcp85": "RCP8.5", "bri": "Brisbane 2024"}
PCT, GAP, SEED = 99.5, 72, 0
NBOOT = int(os.environ.get("NBOOT", 2000))  # 2000 for the paper; smaller for quick checks
RPS = (2, 5, 10, 25, 50, 100, 200)
SC = ("rcp45", "rcp85")
N = {}      # numbers (JSON)
ROWS = {}   # LaTeX table bodies


def save(fig, name):
    fig.savefig(os.path.join(IMG, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def mdates(am):
    return pd.to_datetime(am.index.astype(str)) + pd.Timedelta(days=182)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
STATION = json.load(open(os.path.join(DATA, "station.json")))
ref_df, iref = A.load_series(os.path.join(DATA, "reference_era5_1976_2005.csv.gz"))
hist_df, ihist = A.load_series(os.path.join(DATA, "historical_1976_2005.csv.gz"))
r45_df, i45 = A.load_series(os.path.join(DATA, "rcp45_2041_2100.csv.gz"))
r85_df, i85 = A.load_series(os.path.join(DATA, "rcp85_2041_2100.csv.gz"))
bri_df, ib = A.load_series(os.path.join(DATA, "brisbane_2024.csv.gz"))
DF = {"ref": ref_df, "hist": hist_df, "rcp45": r45_df, "rcp85": r85_df, "bri": bri_df}
INFO = {"ref": iref, "hist": ihist, "rcp45": i45, "rcp85": i85, "bri": ib}
S = {k: {v: A.to_series(df, v) for v in ("swh", "tm", "dir") if v in df.columns} for k, df in DF.items()}

# Table 1: records
rows = []
for k in ("ref", "hist", "rcp45", "rcp85", "bri"):
    st = A.summary_stats(S[k]["swh"]); info = INFO[k]
    tm = S[k]["tm"].mean() if "tm" in S[k] else np.nan
    period = f"{info['start'].year}" if info["span_years"] < 1.5 else f"{info['start'].year} to {info['end'].year}"
    rows.append((LBL[k].replace(" 1976-2005", "").replace(" 2024", ""), period, f"{info['step_hours']:g}", f"{st['n']:,}",
                 f"{st['mean']:.2f}", f"{st['p99']:.2f}", f"{st['max']:.2f}", "--" if not np.isfinite(tm) else f"{tm:.1f}"))
    N[f"{k}_mean_hs"] = round(st["mean"], 3); N[f"{k}_max_hs"] = round(st["max"], 2)
    N[f"{k}_p99_hs"] = round(st["p99"], 2); N[f"{k}_n"] = st["n"]
    N[f"{k}_mean_tm"] = round(float(tm), 2) if np.isfinite(tm) else None
ROWS["Datasets"] = "\n".join(" & ".join(r) + r" \\" for r in rows)

# ---------------------------------------------------------------------------
# Figure: site map (Europe domain and a zoom on western Sicily)
# ---------------------------------------------------------------------------
try:
    from global_land_mask import globe
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.35, 1]})
    cmap = matplotlib.colors.ListedColormap(["#dbe9f6", "#d9d4c7"])
    lons = np.arange(-40, 45.01, 0.1); lats = np.arange(28, 80.01, 0.1)
    LON, LAT = np.meshgrid(lons, lats)
    axes[0].imshow(np.where(globe.is_land(LAT, LON), 1.0, 0.0), extent=[lons[0], lons[-1], lats[0], lats[-1]], origin="lower",
                   cmap=cmap, aspect=1 / np.cos(np.radians(50)), interpolation="nearest")
    axes[0].add_patch(plt.Rectangle((-37.9, 30), 43.6 + 37.9, 79.1 - 30, fill=False, ls="--", lw=1, ec="#555555"))
    axes[0].plot(STATION["lon"], STATION["lat"], "o", ms=7, mfc="#c0392b", mec="k", zorder=5)
    axes[0].add_patch(plt.Rectangle((11.4, 37.3), 1.6, 1.2, fill=False, lw=1.2, ec="#c0392b"))
    axes[0].text(-36.5, 31, "C3S coastal wave projection domain", fontsize=8, color="#555555")
    axes[0].set_xlabel("Longitude [°E]"); axes[0].set_ylabel("Latitude [°N]"); axes[0].set_title("(a) Domain", loc="left")
    zl = np.arange(11.4, 13.0001, 0.005); zt = np.arange(37.3, 38.5001, 0.005)
    ZL, ZT = np.meshgrid(zl, zt)
    axes[1].imshow(np.where(globe.is_land(ZT, ZL), 1.0, 0.0), extent=[zl[0], zl[-1], zt[0], zt[-1]], origin="lower",
                   cmap=cmap, aspect=1 / np.cos(np.radians(38)), interpolation="nearest")
    axes[1].plot(STATION["lon"], STATION["lat"], "o", ms=9, mfc="#c0392b", mec="k", zorder=5, label="C3S station")
    axes[1].plot(STATION["era5_lon"], STATION["era5_lat"], "s", ms=9, mfc="#ffffff", mec="k", zorder=5, label="ERA5 point")
    for nm, lo, la, off in (("Marettimo", 12.06, 37.97, (-6, -14)), ("Favignana", 12.33, 37.93, (6, -12)), ("Levanzo", 12.34, 38.01, (-14, 10)),
                            ("Trapani", 12.51, 38.02, (8, 4)), ("Marsala", 12.44, 37.80, (8, 0))):
        axes[1].annotate(nm, (lo, la), xytext=off, textcoords="offset points", fontsize=7.5, color="#333333")
    axes[1].set_xlim(zl[0], zl[-1]); axes[1].set_ylim(zt[0], zt[-1])
    axes[1].set_xlabel("Longitude [°E]"); axes[1].set_title("(b) Egadi Islands, western Sicily", loc="left"); axes[1].legend(loc="lower left", fontsize=7.5)
    fig.tight_layout(); save(fig, "fig_site_map.png")
except ImportError:
    print("global_land_mask not installed; site map skipped")

# ---------------------------------------------------------------------------
# Reference period: ERA5 against the model's historical run
# ---------------------------------------------------------------------------
qs = np.concatenate([np.linspace(0.01, 0.99, 99), [0.995, 0.999]])
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
for k in ("ref", "hist"):
    mc = A.monthly_climatology(S[k]["swh"])
    axes[0].plot(mc.index, mc.values, "o-", ms=4, color=C[k], label=LBL[k])
axes[0].set_xticks(range(1, 13)); axes[0].set_xticklabels(list("JFMAMJJASOND")); axes[0].set_ylabel("Monthly mean $H_s$ [m]")
axes[0].set_title("(a) Seasonal cycle", loc="left"); axes[0].legend()
qr = np.quantile(S["ref"]["swh"].values, qs); qh = np.quantile(S["hist"]["swh"].values, qs)
axes[1].plot([0, qr.max() * 1.05], [0, qr.max() * 1.05], "k--", lw=1, label="1:1")
axes[1].plot(qr, qh, "o-", ms=2.5, color=C["hist"], label="model historical against ERA5")
axes[1].set_xlabel("ERA5 quantile [m]"); axes[1].set_ylabel("Model quantile [m]"); axes[1].set_title("(b) Quantiles, 1st to 99.9th", loc="left"); axes[1].legend()
fig.tight_layout(); save(fig, "fig_reference_period.png")
mr, mh = A.monthly_climatology(S["ref"]["swh"]), A.monthly_climatology(S["hist"]["swh"])
N["ref_month_max"] = int(mr.idxmax()); N["ref_month_min"] = int(mr.idxmin())
N["ref_winter"] = round(float(mr.loc[[12, 1, 2]].mean()), 2); N["ref_summer"] = round(float(mr.loc[[6, 7, 8]].mean()), 2)
N["hist_winter"] = round(float(mh.loc[[12, 1, 2]].mean()), 2); N["hist_summer"] = round(float(mh.loc[[6, 7, 8]].mean()), 2)
N["hist_minus_ref_mean"] = round(N["hist_mean_hs"] - N["ref_mean_hs"], 3)
N["hist_ref_q99_ratio"] = round(float(np.quantile(S["hist"]["swh"], 0.99) / np.quantile(S["ref"]["swh"], 0.99)), 3)
N["hist_ref_q50_ratio"] = round(float(np.quantile(S["hist"]["swh"], 0.5) / np.quantile(S["ref"]["swh"], 0.5)), 3)

# ---------------------------------------------------------------------------
# Bias correction: split-sample validation, then projections by method
# ---------------------------------------------------------------------------
ref_s, hist_s = S["ref"]["swh"], S["hist"]["swh"]
overlap = sorted(set(ref_s.index.year) & set(hist_s.index.year))
N["bc_overlap_first"], N["bc_overlap_last"], N["bc_overlap_years"] = overlap[0], overlap[-1], len(overlap)
vtab, vser, vinfo = P.validate_corrections(ref_s, hist_s, kind="multiplicative", mode="alternate")
ROWS["Validation"] = "\n".join(f"{m} & {r.mean_bias:+.3f} & {r.std_ratio:.3f} & {r.seasonal_rmse:.3f} & {r.quantile_rmse:.3f} & "
                               f"{r.p99_bias:+.3f} & {r.p995_bias:+.3f} & {r.annual_max_bias:+.3f} \\\\" for m, r in vtab.iterrows())
for m, r in vtab.iterrows():
    key = {"raw model": "raw", "delta change": "delta", "QDM": "qdm"}[m]
    for col in ("mean_bias", "std_ratio", "seasonal_rmse", "quantile_rmse", "p99_bias", "p995_bias", "annual_max_bias"):
        N[f"val_{key}_{col}"] = round(float(r[col]), 3)
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
qr = np.quantile(vser["reference"].dropna().values, qs)
axes[0].plot([qr.min(), qr.max()], [qr.min(), qr.max()], "k--", lw=1, label="1:1")
for m, key in (("raw model", "raw"), ("delta change", "delta"), ("QDM", "qdm")):
    axes[0].plot(qr, np.quantile(vser[m].dropna().values, qs), "-o", ms=2.5, color=C[key], label=m)
    mc = A.monthly_climatology(vser[m]); axes[1].plot(mc.index, mc.values, "o-", ms=4, color=C[key], label=m)
mcr = A.monthly_climatology(vser["reference"]); axes[1].plot(mcr.index, mcr.values, "o-", ms=5, color=C["ref"], lw=2.2, label="ERA5")
axes[0].set_xlabel("ERA5 quantile [m]"); axes[0].set_ylabel("Corrected model quantile [m]"); axes[0].set_title("(a) Held-out years: quantiles", loc="left"); axes[0].legend()
axes[1].set_xticks(range(1, 13)); axes[1].set_xticklabels(list("JFMAMJJASOND")); axes[1].set_ylabel("$H_s$ [m]")
axes[1].set_title("(b) Held-out years: seasonal cycle", loc="left"); axes[1].legend()
fig.tight_layout(); save(fig, "fig_bc_validation.png")

PROJ = {k: {m: P.project(ref_s, hist_s, S[k]["swh"], method={"raw": "raw", "delta": "delta", "qdm": "QDM"}[m], kind="multiplicative")
            for m in ("raw", "delta", "qdm")} for k in SC}
for k in SC:
    S[k]["qdm"] = PROJ[k]["qdm"]
# sensitivity of the corrected tail to where the QDM correction factor is held constant
for k in SC:
    for cap, tag in ((0.995, "cap995"), (None, "nocap")):
        q = P.quantile_delta_mapping(ref_s, hist_s, S[k]["swh"], kind="multiplicative", tail_prob=cap)
        r = A.pot_analysis(q, PCT, GAP, (100,), n_boot=0)
        N[f"{k}_rl100_{tag}"] = round(float(r["return_levels"][0]), 2); N[f"{k}_xi_{tag}"] = round(r["xi"], 3)
# QDM keeps the model's own change in each quantile
for k in SC:
    for q in (0.5, 0.99, 0.999):
        mch = np.quantile(S[k]["swh"], q) / np.quantile(hist_s, q)
        qch = np.quantile(PROJ[k]["qdm"], q) / np.quantile(ref_s, q)
        N[f"{k}_model_change_q{int(q * 1000)}"] = round(float(mch), 3); N[f"{k}_qdm_change_q{int(q * 1000)}"] = round(float(qch), 3)

# ---------------------------------------------------------------------------
# Projected mean conditions and trends (Hs corrected by QDM; Tm as modelled)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True)
for ax, k in zip(axes, SC):
    for m in ("raw", "delta", "qdm"):
        am = A.annual_means(PROJ[k][m])
        lab = {"raw": "raw model", "delta": "delta change", "qdm": "QDM"}[m]
        x = am.index if m != "delta" else None
        if m == "delta":  # the delta-change series keeps the reference years; drawn as its mean level only
            ax.axhline(PROJ[k][m].mean(), color=C["delta"], lw=1.6, ls="-.", label=f"delta change (mean {PROJ[k][m].mean():.2f} m)")
        else:
            ax.plot(x, am.values, "o-", ms=3, lw=1, color=C[m] if m != "qdm" else C[k], label=lab)
    ax.axhline(ref_s.mean(), color=C["ref"], lw=1, ls=":", label=f"ERA5 1976-2005 mean ({ref_s.mean():.2f} m)")
    ax.set_ylabel("Annual mean $H_s$ [m]"); ax.set_title(LBL[k], loc="left"); ax.legend(ncol=2, fontsize=7)
axes[1].set_xlabel("Year")
save(fig, "fig_bc_projections.png")

trend_rows = []
TR = {}
fig, axes = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True)
HAS_TM = all("tm" in S[k] for k in SC)
for ax, (v, src) in zip(axes, (("swh", "qdm"), ("tm", "tm"))):
    if v == "tm" and not HAS_TM:
        ax.text(0.5, 0.5, "mean wave period not yet available", ha="center", transform=ax.transAxes); continue
    f_dec, u_dec, f_chg, u_chg, d, dc = (1000.0, "cm/decade", 100.0, "cm", 2, 1) if v == "swh" else (10.0, "s/decade", 1.0, "s", 3, 2)
    for k in SC:
        series = S[k]["qdm"] if src == "qdm" else S[k]["tm"]
        tr = A.trend_analysis(series); TR[(k, v)] = tr
        am, o, mk, se = tr["annual_means"], tr["ols"], tr["mk"], tr["sen"]
        ax.plot(am.index, am.values, "o-", ms=3, lw=1, color=C[k], label=f"{LBL[k]} annual mean")
        ax.plot(am.index, o["intercept"] + o["slope"] * am.index, "--", lw=1.8, color=C[k],
                label=f"{LBL[k]} OLS: {o['slope'] * f_dec:+.{d}f} {u_dec}, p = {o['p_value']:.2f}")
        name = "$H_s$ (QDM)" if v == "swh" else "$T_m$ (model)"
        trend_rows.append((name, LBL[k], f"{am.mean():.2f}", f"{o['slope'] * f_dec:+.{d}f}",
                           f"[{o['slope_ci'][0] * f_dec:+.{d}f}, {o['slope_ci'][1] * f_dec:+.{d}f}]", f"{o['p_value']:.2f}",
                           f"{se['slope'] * f_dec:+.{d}f}", f"[{se['slope_ci'][0] * f_dec:+.{d}f}, {se['slope_ci'][1] * f_dec:+.{d}f}]",
                           f"{mk['p_value']:.2f}", f"{tr['change_over_record']['ols'] * f_chg:+.{dc}f} {u_chg}"))
        N[f"{k}_{v}_ols_per_decade"] = round(o["slope"] * f_dec, d)
        N[f"{k}_{v}_ols_ci_per_decade"] = [round(o["slope_ci"][0] * f_dec, d), round(o["slope_ci"][1] * f_dec, d)]
        N[f"{k}_{v}_ols_p"] = round(o["p_value"], 3); N[f"{k}_{v}_mk_p"] = round(mk["p_value"], 3)
        N[f"{k}_{v}_sen_per_decade"] = round(se["slope"] * f_dec, d)
        N[f"{k}_{v}_sen_ci_per_decade"] = [round(se["slope_ci"][0] * f_dec, d), round(se["slope_ci"][1] * f_dec, d)]
        N[f"{k}_{v}_change_over_record"] = round(tr["change_over_record"]["ols"] * f_chg, dc)
        N[f"{k}_{v}_record_years"] = tr["change_over_record"]["years"]
        N[f"{k}_{v}_annual_mean"] = round(float(am.mean()), 3); N[f"{k}_{v}_annual_std"] = round(float(am.std()), 3)
        N[f"{k}_{v}_annual_min"] = round(float(am.min()), 2); N[f"{k}_{v}_annual_max"] = round(float(am.max()), 2)
    ax.set_ylabel("Annual mean $H_s$, QDM [m]" if v == "swh" else "Annual mean $T_m$ [s]")
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.45 * (hi - lo))   # room for the legend above the data
    ax.legend(ncol=2, fontsize=7, loc="upper center")
axes[1].set_xlabel("Year")
save(fig, "fig_annual_means.png")
# raw-model Hs trends, for the comparison in the text
for k in SC:
    tr = A.trend_analysis(S[k]["swh"])
    N[f"{k}_rawhs_ols_per_decade"] = round(tr["ols"]["slope"] * 1000, 2); N[f"{k}_rawhs_ols_p"] = round(tr["ols"]["p_value"], 3)
ROWS["Trends"] = "\n".join(" & ".join(r) + r" \\" for r in trend_rows)
am45, am85 = A.annual_means(S["rcp45"]["qdm"]), A.annual_means(S["rcp85"]["qdm"])
N["scenario_mean_diff_cm"] = round(float(am85.mean() - am45.mean()) * 100, 1)
N["qdm_mean_change_rcp45_cm"] = round(float(S["rcp45"]["qdm"].mean() - ref_s.mean()) * 100, 1)
N["qdm_mean_change_rcp85_cm"] = round(float(S["rcp85"]["qdm"].mean() - ref_s.mean()) * 100, 1)

# Seasonal cycle: ERA5, model historical, and the corrected projections
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
for k, ser, lab in (("ref", S["ref"]["swh"], LBL["ref"]), ("hist", S["hist"]["swh"], LBL["hist"]),
                    ("rcp45", S["rcp45"]["qdm"], "RCP4.5, QDM"), ("rcp85", S["rcp85"]["qdm"], "RCP8.5, QDM")):
    mc = A.monthly_climatology(ser); axes[0].plot(mc.index, mc.values, "o-", ms=4, color=C[k], label=lab)
for k in ("hist", "rcp45", "rcp85"):
    if "tm" in S[k]:
        mc = A.monthly_climatology(S[k]["tm"]); axes[1].plot(mc.index, mc.values, "o-", ms=4, color=C[k], label=LBL[k])
for ax in axes:
    ax.set_xticks(range(1, 13)); ax.set_xticklabels(list("JFMAMJJASOND")); ax.legend(fontsize=7)
axes[0].set_ylabel("$H_s$ [m]"); axes[1].set_ylabel("$T_m$ [s]")
save(fig, "fig_seasonal_cycle.png")
mc45, mc85 = A.monthly_climatology(S["rcp45"]["qdm"]), A.monthly_climatology(S["rcp85"]["qdm"])
N["seasonal_max_abs_diff_cm"] = round(float((mc85 - mc45).abs().max()) * 100, 1)
N["seasonal_max_abs_diff_month"] = int((mc85 - mc45).abs().idxmax())
N["winter_change_rcp85_cm"] = round(float(mc85.loc[[12, 1, 2]].mean() - mr.loc[[12, 1, 2]].mean()) * 100, 1)
N["summer_change_rcp85_cm"] = round(float(mc85.loc[[6, 7, 8]].mean() - mr.loc[[6, 7, 8]].mean()) * 100, 1)

# ---------------------------------------------------------------------------
# Extremes: ERA5 present climate and the QDM-corrected projections
# ---------------------------------------------------------------------------
EX = {"ref": S["ref"]["swh"], "rcp45": S["rcp45"]["qdm"], "rcp85": S["rcp85"]["qdm"]}
R = {k: A.pot_analysis(s, PCT, GAP, RPS, n_boot=NBOOT, seed=SEED) for k, s in EX.items()}
for k, r in R.items():
    N[f"{k}_threshold"] = round(r["threshold"], 2); N[f"{k}_n_raw"] = r["n_raw_exceedances"]; N[f"{k}_n_peaks"] = r["n_peaks"]
    N[f"{k}_rate"] = round(r["rate_per_year"], 2); N[f"{k}_xi"] = round(r["xi"], 3); N[f"{k}_xi_ci"] = [round(x, 2) for x in r["xi_ci"]]
    N[f"{k}_beta"] = round(r["beta"], 3)
    N[f"{k}_upper_bound"] = round(r["upper_bound"], 1) if np.isfinite(r["upper_bound"]) else None
    for T, rl, lo, hi in zip(r["return_periods"], r["return_levels"], *r["rl_ci"]):
        N[f"{k}_rl{int(T)}"] = round(float(rl), 2); N[f"{k}_rl{int(T)}_ci"] = [round(float(lo), 2), round(float(hi), 2)]
    N[f"{k}_max_peak"] = round(float(r["peaks"].max()), 2)
# the interface default of 500 resamples against the paper's NBOOT, for the 100-year interval
_d500 = []
for k, s in EX.items():
    r5 = A.pot_analysis(s, PCT, GAP, (100,), n_boot=500, seed=SEED)
    _d500 += [abs(float(r5["rl_ci"][0][0]) - N[f"{k}_rl100_ci"][0]), abs(float(r5["rl_ci"][1][0]) - N[f"{k}_rl100_ci"][1])]
N["boot500_max_diff"] = round(max(_d500), 2)
for k in SC:
    N[f"{k}_rl100_change_cm"] = round((N[f"{k}_rl100"] - N["ref_rl100"]) * 100, 0)
N["rl100_diff_cm"] = round((N["rcp85_rl100"] - N["rcp45_rl100"]) * 100, 0)
# raw (uncorrected) model extremes, for the comparison
RAW = {k: A.pot_analysis(S[k]["swh"], PCT, GAP, (10, 100), n_boot=500, seed=SEED) for k in ("hist", "rcp45", "rcp85")}
for k, r in RAW.items():
    N[f"raw_{k}_rl100"] = round(float(r["return_levels"][1]), 2); N[f"raw_{k}_rl100_ci"] = [round(float(r["rl_ci"][0][1]), 2), round(float(r["rl_ci"][1][1]), 2)]
    N[f"raw_{k}_xi"] = round(r["xi"], 3)

gpd_rows = []
for k in ("ref", "rcp45", "rcp85"):
    r = R[k]; lab = LBL[k] if k == "ref" else f"{LBL[k]}, QDM"
    ub = "none ($\\xi \\ge 0$)" if not np.isfinite(r["upper_bound"]) else f"{r['upper_bound']:.1f}"
    gpd_rows.append(f"{lab} & {r['threshold']:.2f} & {r['n_raw_exceedances']} & {r['n_peaks']} & {r['rate_per_year']:.2f} & "
                    f"{r['xi']:+.3f} & [{r['xi_ci'][0]:+.2f}, {r['xi_ci'][1]:+.2f}] & {r['beta']:.3f} & {ub} \\\\")
ROWS["Gpd"] = "\n".join(gpd_rows)
rl_rows = []
for T in (10, 25, 50, 100, 200):
    cells = [str(T)]
    for k in ("ref", "rcp45", "rcp85"):
        cells += [f"{N[f'{k}_rl{T}']:.2f}", f"[{N[f'{k}_rl{T}_ci'][0]:.2f}, {N[f'{k}_rl{T}_ci'][1]:.2f}]"]
    rl_rows.append(" & ".join(cells) + r" \\")
ROWS["ReturnLevels"] = "\n".join(rl_rows)

# Storm peaks
fig, axes = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True)
for ax, k in zip(axes, SC):
    s, r = EX[k], R[k]
    ax.plot(s.index, s.values, lw=0.25, color="#aaaaaa", label="hourly $H_s$, QDM")
    ax.plot(r["peaks"].index, r["peaks"].values, "o", ms=3, color=C[k], label=f"storm peaks (n = {r['n_peaks']})")
    ax.axhline(r["threshold"], ls="--", lw=1.2, color="k", label=f"threshold {r['threshold']:.2f} m ({PCT:g}th percentile)")
    ax.set_ylabel("$H_s$ [m]"); ax.set_title(LBL[k], loc="left"); ax.legend(loc="upper right")
axes[1].set_xlabel("Year")
save(fig, "fig_storm_peaks.png")

# GPD fit and QQ
fig, axes = plt.subplots(2, 2, figsize=(9, 6))
for i, k in enumerate(SC):
    r = R[k]; y = r["exceedances"]
    ax = axes[i, 0]
    ax.hist(y, bins=25, density=True, alpha=0.45, color=C[k], label="storm-peak exceedances")
    xx = np.linspace(0, y.max() * 1.05, 200)
    ax.plot(xx, genpareto.pdf(xx, r["xi"], 0, r["beta"]), "k-", lw=1.5, label=f"GPD: $\\xi$ = {r['xi']:+.3f}, $\\beta$ = {r['beta']:.2f} m")
    ax.set_xlabel("Exceedance over threshold [m]"); ax.set_ylabel("Density"); ax.set_title(f"{LBL[k]}: fitted density", loc="left"); ax.legend()
    ax = axes[i, 1]
    fq, eq = A.qq_points(r)
    ax.plot(fq, eq, "o", ms=3, color=C[k]); lim = [0, max(fq.max(), eq.max()) * 1.05]; ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("Fitted GPD quantile [m]"); ax.set_ylabel("Empirical quantile [m]"); ax.set_title(f"{LBL[k]}: QQ plot", loc="left")
fig.tight_layout(); save(fig, "fig_gpd_fit.png")

# Return levels
fig, ax = plt.subplots(figsize=(7, 4.2))
for k in ("ref", "rcp45", "rcp85"):
    r = R[k]; lab = LBL[k] if k == "ref" else f"{LBL[k]}, QDM"
    T = np.logspace(np.log10(1.0 / r["rate_per_year"] * 1.05), np.log10(300), 300)
    ax.plot(T, A.return_level(r["threshold"], r["xi"], r["beta"], r["rate_per_year"], T), "-", lw=2, color=C[k], label=f"{lab} fitted")
    ax.fill_between(r["return_periods"], r["rl_ci"][0], r["rl_ci"][1], color=C[k], alpha=0.12, label=f"{lab} 95%")
    ex, eT = r["empirical"]
    ax.plot(eT, ex, "o", ms=2.5, color=C[k], mec="k", mew=0.3, alpha=0.7)
ax.set_xscale("log"); ax.set_xlabel("Return period [years]"); ax.set_ylabel("$H_s$ return level [m]")
ax.grid(alpha=0.3, which="both"); ax.legend(ncol=2, fontsize=7)
save(fig, "fig_return_levels.png")

# Threshold diagnostics
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
for k in SC:
    s = EX[k]
    mrl = A.mean_residual_life(s, np.arange(90, 99.91, 0.25)); ps = A.parameter_stability(s, np.arange(97.0, 99.91, 0.2), GAP)
    axes[0].errorbar(mrl["threshold"], mrl["mean_excess"], yerr=1.96 * mrl["se"], fmt="o-", ms=2.5, lw=0.8, color=C[k], label=LBL[k])
    axes[1].plot(ps["threshold"], ps["xi"], "o-", ms=3, color=C[k], label=LBL[k])
    axes[2].plot(ps["threshold"], ps["modified_scale"], "o-", ms=3, color=C[k], label=LBL[k])
    for ax in axes:
        ax.axvline(R[k]["threshold"], color=C[k], ls=":", lw=1)
    N[f"{k}_ps_xi_range"] = [round(float(ps["xi"].min()), 2), round(float(ps["xi"].max()), 2)]
axes[0].set_xlabel("Threshold $u$ [m]"); axes[0].set_ylabel("Mean excess [m]"); axes[0].set_title("Mean residual life", loc="left")
axes[1].set_xlabel("Threshold $u$ [m]"); axes[1].set_ylabel("Shape $\\xi$"); axes[1].set_title("Shape stability", loc="left")
axes[2].set_xlabel("Threshold $u$ [m]"); axes[2].set_ylabel("Modified scale $\\beta - \\xi u$ [m]"); axes[2].set_title("Scale stability", loc="left")
axes[0].legend(); fig.tight_layout(); save(fig, "fig_threshold_diagnostics.png")

# Sensitivity grid
grids = {k: A.sensitivity_grid(EX[k], (99.0, 99.5, 99.8), (24, 48, 72), T=100) for k in SC}
sens_rows = []
for p in (99.0, 99.5, 99.8):
    for g in (24, 48, 72):
        cells = [f"{p:.1f}", str(g)]
        for k in SC:
            row = grids[k][(grids[k].percentile == p) & (grids[k].gap_hours == g)].iloc[0]
            cells += [str(int(row.n_peaks)), f"{row.xi:+.3f}", f"{row.RL_100:.2f}"]
        sens_rows.append(" & ".join(cells) + r" \\")
ROWS["Sensitivity"] = "\n".join(sens_rows)
for k in SC:
    g = grids[k]
    N[f"{k}_rl100_grid_min"] = round(float(g.RL_100.min()), 2); N[f"{k}_rl100_grid_max"] = round(float(g.RL_100.max()), 2)
    N[f"{k}_xi_grid_min"] = round(float(g.xi.min()), 3); N[f"{k}_xi_grid_max"] = round(float(g.xi.max()), 3)
    d = g[(g.percentile == PCT) & (g.gap_hours == GAP)].iloc[0]
    N[f"{k}_grid_xi_maxdev"] = round(float((g.xi - d.xi).abs().max()), 3)
    hi = g[g.percentile == 99.8]
    N[f"{k}_rl100_998_min"] = round(float(hi.RL_100.min()), 2); N[f"{k}_rl100_998_max"] = round(float(hi.RL_100.max()), 2)
    N[f"{k}_n_998_min"] = int(hi.n_peaks.min()); N[f"{k}_n_998_max"] = int(hi.n_peaks.max())
    for p, tag in ((99.0, "990"), (99.5, "995")):
        sub = g[g.percentile == p]
        N[f"{k}_rl100_{tag}_min"] = round(float(sub.RL_100.min()), 2); N[f"{k}_rl100_{tag}_max"] = round(float(sub.RL_100.max()), 2)
        N[f"{k}_n_{tag}_min"] = int(sub.n_peaks.min()); N[f"{k}_n_{tag}_max"] = int(sub.n_peaks.max())
        N[f"{k}_xi_{tag}_max"] = round(float(sub.xi.max()), 3)
N["grid_rl100_range_cm"] = round(max(N[f"{k}_rl100_grid_max"] - N[f"{k}_rl100_grid_min"] for k in SC) * 100, 0)

# Correction methods compared: mean, trend, shape and 100-year level
bc_rows = []
for k in SC:
    for m in ("raw", "delta", "qdm"):
        proj = PROJ[k][m]
        r = A.pot_analysis(proj, PCT, GAP, (100,), n_boot=NBOOT, seed=SEED)
        tr = A.trend_analysis(proj) if m != "delta" else None
        trend_txt = f"{tr['ols']['slope'] * 1000:+.2f}" if tr else "--"
        lab = {"raw": "raw model", "delta": "delta change", "qdm": "QDM"}[m]
        bc_rows.append(f"{LBL[k]} & {lab} & {proj.mean():.2f} & {trend_txt} & {r['xi']:+.3f} & {r['return_levels'][0]:.2f} & "
                       f"\\mbox{{[{r['rl_ci'][0][0]:.2f}, {r['rl_ci'][1][0]:.2f}]}} \\\\")
        N[f"bc_{k}_{m}_rl100"] = round(float(r["return_levels"][0]), 2); N[f"bc_{k}_{m}_mean"] = round(float(proj.mean()), 3)
        N[f"bc_{k}_{m}_rl100_ci"] = [round(float(r["rl_ci"][0][0]), 2), round(float(r["rl_ci"][1][0]), 2)]
ROWS["BcProjections"] = "\n".join(bc_rows)
for k in SC:
    vals = [N[f"bc_{k}_{m}_rl100"] for m in ("raw", "delta", "qdm")]
    N[f"bc_rl100_spread_{k}"] = round(max(vals) - min(vals), 2)

# ---------------------------------------------------------------------------
# Non-stationary extremes (transformed-stationary), QDM-corrected projections
# ---------------------------------------------------------------------------
WINDOW = 30
TS = {k: P.ts_eva(EX[k], window_years=WINDOW, percentile=PCT, min_gap_hours=GAP, return_periods=(10, 50, 100), n_boot=NBOOT, seed=SEED) for k in SC}
fig, axes = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True, gridspec_kw={"height_ratios": [1.6, 1]})
for k in SC:
    t = TS[k]["table"]; st_rl = N[f"{k}_rl100"]
    axes[0].fill_between(t.index, t["RL_100_lo"], t["RL_100_hi"], color=C[k], alpha=0.12)
    axes[0].plot(t.index, t["RL_100"], color=C[k], lw=2.2, label=f"{LBL[k]} time-varying 100-year level")
    axes[0].axhline(st_rl, color=C[k], ls="--", lw=1.3, label=f"{LBL[k]} stationary 100-year level ({st_rl:.2f} m)")
    axes[1].plot(t.index, t["mu"], color=C[k], lw=1.8, label=f"{LBL[k]} running mean $\\mu(t)$")
    axes[1].plot(t.index, t["sigma"], color=C[k], lw=1.8, ls=":", label=f"{LBL[k]} running std $\\sigma(t)$")
    sm = P.ts_eva_summary(TS[k], st_rl, 100)
    N[f"{k}_ts_rl100_first"] = round(sm["rl_first"], 2); N[f"{k}_ts_rl100_last"] = round(sm["rl_last"], 2)
    N[f"{k}_ts_rl100_change"] = round(sm["change"], 2); N[f"{k}_ts_rl100_min"] = round(sm["rl_min"], 2); N[f"{k}_ts_rl100_max"] = round(sm["rl_max"], 2)
    N[f"{k}_ts_xi"] = round(TS[k]["result_y"]["xi"], 3); N[f"{k}_ts_n_peaks"] = TS[k]["result_y"]["n_peaks"]
    N[f"{k}_ts_mu_first"] = round(float(t["mu"].iloc[0]), 3); N[f"{k}_ts_mu_last"] = round(float(t["mu"].iloc[-1]), 3)
    N[f"{k}_ts_sigma_first"] = round(float(t["sigma"].iloc[0]), 3); N[f"{k}_ts_sigma_last"] = round(float(t["sigma"].iloc[-1]), 3)
    N[f"{k}_ts_band_last"] = [round(float(t["RL_100_lo"].iloc[-1]), 2), round(float(t["RL_100_hi"].iloc[-1]), 2)]
axes[0].set_ylabel("100-year $H_s$ [m]"); axes[0].legend(ncol=2, fontsize=7)
axes[1].set_ylabel("[m]"); axes[1].set_xlabel("Year"); axes[1].legend(ncol=2, fontsize=7)
save(fig, "fig_ts_eva.png")
ROWS["TsEva"] = "\n".join(f"{LBL[k]} & {N[f'{k}_ts_n_peaks']} & {N[f'{k}_ts_xi']:+.3f} & {N[f'{k}_rl100']:.2f} & {N[f'{k}_ts_rl100_first']:.2f} & "
                          f"{N[f'{k}_ts_rl100_last']:.2f} & {N[f'{k}_ts_rl100_change']:+.2f} & \\mbox{{{N[f'{k}_ts_rl100_min']:.2f} to {N[f'{k}_ts_rl100_max']:.2f}}} \\\\"
                          for k in SC)
N["ts_window"] = WINDOW
N["ts_max_dev_from_stationary"] = round(max(abs(N[f"{k}_ts_rl100_{e}"] - N[f"{k}_rl100"]) for k in SC for e in ("min", "max")), 2)

# ---------------------------------------------------------------------------
# Storm directions (ERA5 and model mean wave direction at the storm peaks)
# ---------------------------------------------------------------------------
DIRS = {}
for k, pot in (("ref", R["ref"]), ("hist", RAW["hist"]), ("rcp45", R["rcp45"]), ("rcp85", R["rcp85"])):
    if "dir" in S[k]:
        DIRS[k] = A.storm_directions(S[k]["dir"], pot)
if DIRS:
    fig, axes = plt.subplots(1, len(DIRS), figsize=(2.6 * len(DIRS), 3.0), subplot_kw={"projection": "polar"})
    for ax, (k, d) in zip(np.atleast_1d(axes), DIRS.items()):
        th = np.radians(np.arange(16) * 22.5)
        ax.bar(th, d["storm_share"], width=np.radians(22.5) * 0.9, color=C[k], alpha=0.55, edgecolor=C[k], label="storm peaks")
        ax.plot(np.append(th, th[0]), np.append(d["all_share"], d["all_share"][0]), ":", color="k", lw=1, label="all hours")
        ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
        ax.set_xticks(np.radians([0, 90, 180, 270])); ax.set_xticklabels(["N", "E", "S", "W"])
        ax.set_yticklabels([]); ax.set_title(LBL[k].replace(" 1976-2005", "\n1976-2005"), fontsize=8.5)
        N[f"{k}_dir_dominant"] = d["dominant"]; N[f"{k}_dir_dominant_share"] = round(d["dominant_share"], 1)
        N[f"{k}_dir_within45"] = round(d["within_45"], 1); N[f"{k}_dir_storm_mean"] = round(d["storm_mean"], 0)
        N[f"{k}_dir_all_mean"] = round(d["all_mean"], 0); N[f"{k}_dir_n"] = d["n_storms"]
        top = A.SECTORS.index(d["dominant"])
        N[f"{k}_dir_all_dominant_share"] = round(float(d["all_share"][top]), 1)
        N[f"{k}_dir_w_nw"] = round(float(d["storm_share"][12:15].sum()), 1)        # W, WNW, NW
        N[f"{k}_dir_all_w_nw"] = round(float(d["all_share"][12:15].sum()), 1)
        if "ref" in DIRS:
            c0 = A.SECTORS.index(DIRS["ref"]["dominant"]) * 22.5
            pdir = d["peak_dir"].values.astype(float)
            dd = np.abs(((pdir - c0 + 180.0) % 360.0) - 180.0)
            N[f"{k}_dir_within45_ref"] = round(float(100.0 * np.nanmean(dd <= 45.0)), 1)
    # does the share of storms between W and NW differ between the historical run and each projection?
    if "hist" in DIRS:
        from scipy.stats import fisher_exact
        def _wnw_counts(d):
            pdir = d["peak_dir"].dropna().values.astype(float)
            sec = np.floor(((pdir % 360.0) + 11.25) % 360.0 / 22.5).astype(int)
            k_in = int(np.isin(sec, (12, 13, 14)).sum()); return k_in, len(pdir) - k_in
        h_in, h_out = _wnw_counts(DIRS["hist"])
        for k in SC:
            if k in DIRS:
                p_in, p_out = _wnw_counts(DIRS[k])
                N[f"{k}_dir_wnw_fisher_p"] = round(float(fisher_exact([[h_in, h_out], [p_in, p_out]])[1]), 4)
    np.atleast_1d(axes)[0].legend(loc="lower left", bbox_to_anchor=(-0.25, -0.32), fontsize=7, ncol=2)
    fig.tight_layout(); save(fig, "fig_directions.png")

# ---------------------------------------------------------------------------
# Cross-check against pyextremes (same series, threshold, declustering, MLE)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(APP, "reproduce"))
try:
    from compare_pyextremes import compare
    cmp_rows = []; maxdiff = 0.0
    for k in ("ref", "rcp45", "rcp85"):
        tab, u = compare(EX[k], PCT, GAP, periods=(10, 50, 100))
        a, b = tab["this tool"], tab["pyextremes"]
        lab = LBL[k] if k == "ref" else f"{LBL[k]}, QDM"
        cmp_rows.append(f"{lab} & {int(a['storm peaks'])} & {int(b['storm peaks'])} & {a['shape xi']:+.4f} & {b['shape xi']:+.4f} & "
                        f"{a['scale beta (m)']:.4f} & {b['scale beta (m)']:.4f} & {a['100-yr level (m)']:.3f} & {b['100-yr level (m)']:.3f} \\\\")
        maxdiff = max(maxdiff, float((a - b).abs().drop("storm peaks").max()))
        N[f"pyx_{k}_npeaks_equal"] = bool(a["storm peaks"] == b["storm peaks"])
    ROWS["Pyextremes"] = "\n".join(cmp_rows)
    N["pyx_max_abs_diff"] = maxdiff
    import pyextremes
    N["pyx_version"] = pyextremes.__version__
except Exception as e:  # keep the rest of the script useful without pyextremes
    print("pyextremes comparison skipped:", e)
    ROWS["Pyextremes"] = "\\multicolumn{9}{c}{pyextremes not installed} \\\\"
    N["pyx_max_abs_diff"] = float("nan"); N["pyx_version"] = "n/a"

# ---------------------------------------------------------------------------
# Illustration of the GPD shape and scale (no data)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
cols = {-0.15: "#7b3294", -0.10: "#0b5394", -0.05: "#1a9a9a", 0.0: "#c0392b"}
y = np.linspace(0, 10.5, 700)
for xi, c in cols.items():
    for beta, ls in ((1.0, "-"), (1.5, "--")):
        sf = genpareto.sf(y, xi, 0, beta)
        axes[0].plot(y, np.where(sf > 0, sf, np.nan), ls, color=c, lw=1.6,
                     label=(f"$\\xi$ = {xi:+.2f}" if beta == 1.0 else None))
        T = np.logspace(0, 3, 200); lam = 3.0
        axes[1].plot(T, A.return_level(0.0, xi if xi != 0 else 0.0, beta, lam, T), ls, color=c, lw=1.6)
        if xi < 0 and beta == 1.0:
            axes[0].axvline(-beta / xi, color=c, lw=0.6, ls=":")
axes[0].set_yscale("log"); axes[0].set_ylim(1e-4, 1.2); axes[0].set_xlim(0, 10.5)
axes[0].set_xlabel("Exceedance over the threshold, $y$ [m]"); axes[0].set_ylabel("Probability of exceeding $y$")
axes[0].set_title("(a) Tail of the GPD", loc="left")
axes[0].plot([], [], "k-", label="$\\beta$ = 1.0 m"); axes[0].plot([], [], "k--", label="$\\beta$ = 1.5 m"); axes[0].legend(fontsize=7, ncol=2)
axes[1].set_xscale("log"); axes[1].set_xlabel("Return period $T$ [years]"); axes[1].set_ylabel("$H_T - u$ [m]")
axes[1].set_title("(b) Return level above the threshold, 3 storms per year", loc="left")
fig.tight_layout(); save(fig, "fig_gpd_illustration.png")

# ---------------------------------------------------------------------------
# Brisbane: format and regime check, and the refusal rule
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
axes[0].plot(S["bri"]["swh"].index, S["bri"]["swh"].values, lw=0.4, color=C["bri"]); axes[0].set_ylabel("$H_s$ [m]")
axes[0].set_title("Brisbane 2024: $H_s$", loc="left"); axes[0].tick_params(axis="x", labelrotation=45)
axes[1].plot(S["bri"]["tm"].index, S["bri"]["tm"].values, lw=0.4, color=C["bri"]); axes[1].set_ylabel("$T_z$ [s]")
axes[1].set_title("Brisbane 2024: $T_z$", loc="left"); axes[1].tick_params(axis="x", labelrotation=45)
for k in ("ref", "bri"):
    mc = A.monthly_climatology(S[k]["swh"]); axes[2].plot(mc.index, mc.values, "o-", ms=4, color=C[k], label=LBL[k])
axes[2].set_xticks(range(1, 13)); axes[2].set_xticklabels(list("JFMAMJJASOND")); axes[2].set_ylabel("Monthly mean $H_s$ [m]")
axes[2].set_title("Seasonal cycle", loc="left"); axes[2].legend()
fig.tight_layout(); save(fig, "fig_brisbane.png")
mb = A.monthly_climatology(S["bri"]["swh"])
N["bri_month_max"] = int(mb.idxmax()); N["bri_month_min"] = int(mb.idxmin())
N["bri_nan_hs"] = int(bri_df["swh"].isna().sum()); N["bri_step_min"] = int(ib["step_hours"] * 60)
sb = S["bri"]["swh"]; u_b = float(np.percentile(sb, PCT))
N["bri_threshold"] = round(u_b, 2); N["bri_n_raw"] = int((sb > u_b).sum())
N["bri_n_peaks"] = int(len(A.decluster(sb, u_b, GAP)))
try:
    A.pot_analysis(sb, PCT, GAP, (10,), n_boot=10); N["bri_pot_refused"] = False
except ValueError as e:
    N["bri_pot_refused"] = True; N["bri_pot_message"] = str(e)
N["bri_trend_none"] = A.trend_analysis(sb) is None
# most peaks the Brisbane year gives at any setting of the sensitivity grid
N["bri_n_peaks_grid_max"] = int(max(len(A.decluster(sb, float(np.percentile(sb, p)), g)) for p in (99.0, 99.5, 99.8) for g in (24, 48, 72)))
_bri_fit = [(p, g) for p in (99.0, 99.5, 99.8) for g in (24, 48, 72) if len(A.decluster(sb, float(np.percentile(sb, p)), g)) >= 10]
N["bri_grid_fit"] = [list(x) for x in _bri_fit]

# ---------------------------------------------------------------------------
# GUI figure from the screenshots taken by shot_paper.py
# ---------------------------------------------------------------------------
shots = {k: os.path.join(HERE, f"app_{k}.png") for k in ("home", "setup", "report", "section")}
if all(os.path.exists(v) for v in shots.values()):
    from PIL import Image, ImageDraw, ImageFont
    panels = [Image.open(shots[k]).convert("RGB") for k in ("home", "setup", "report", "section")]
    pw = max(p.width for p in panels)
    panels = [p if p.width == pw else p.resize((pw, int(p.height * pw / p.width))) for p in panels]
    gap, strip = 50, 110
    rows_ = [(panels[0], panels[1]), (panels[2], panels[3])]
    heights = [max(a.height, b.height) for a, b in rows_]
    out = Image.new("RGB", (2 * pw + gap, sum(heights) + 2 * strip + gap), "white")
    d = ImageDraw.Draw(out); font = ImageFont.load_default(size=64)
    yy, labels = 0, iter("abcd")
    for (a, b), h in zip(rows_, heights):
        for x, p in ((0, a), (pw + gap, b)):
            d.text((x + 6, yy + 18), f"({next(labels)})", fill="#0b5394", font=font)
            out.paste(p, (x, yy + strip))
        yy += strip + h + gap
    out.save(os.path.join(IMG, "fig_gui.png")); print("wrote fig_gui.png", out.size)

# ---------------------------------------------------------------------------
# LaTeX macros
# ---------------------------------------------------------------------------
MONTH = ["", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def fmt(x, d):
    return f"{x:.{d}f}"


def sgn(x, d):
    s = f"{x:+.{d}f}"
    return s.replace("-", "$-$") if s.startswith("-") else s


TXT = {}
TXT["station_lat"] = fmt(STATION["lat"], 1); TXT["station_lon"] = fmt(STATION["lon"], 1)
TXT["era5_lat"] = fmt(STATION["era5_lat"], 2); TXT["era5_lon"] = fmt(STATION["era5_lon"], 2)
TXT["era5_dist_km"] = fmt(STATION["era5_dist_km"], 0); TXT["station_index"] = str(STATION["index"])
for k in ("ref", "hist", "rcp45", "rcp85", "bri"):
    TXT[f"{k}_mean_hs"] = fmt(N[f"{k}_mean_hs"], 2); TXT[f"{k}_max_hs"] = fmt(N[f"{k}_max_hs"], 2); TXT[f"{k}_p99_hs"] = fmt(N[f"{k}_p99_hs"], 2)
for k in ("hist", "rcp45", "rcp85"):
    TXT[f"{k}_mean_tm"] = fmt(N[f"{k}_mean_tm"], 1) if N[f"{k}_mean_tm"] is not None else "n/a"
TXT["ref_month_max"] = MONTH[N["ref_month_max"]]; TXT["ref_month_min"] = MONTH[N["ref_month_min"]]
for k in ("ref_winter", "ref_summer", "hist_winter", "hist_summer"):
    TXT[k] = fmt(N[k], 2)
TXT["hist_minus_ref_cm"] = sgn(N["hist_minus_ref_mean"] * 100, 0)
TXT["hist_ref_q99_pct"] = sgn((N["hist_ref_q99_ratio"] - 1) * 100, 0); TXT["hist_ref_q50_pct"] = sgn((N["hist_ref_q50_ratio"] - 1) * 100, 0)
TXT["bc_overlap_first"] = str(N["bc_overlap_first"]); TXT["bc_overlap_last"] = str(N["bc_overlap_last"]); TXT["bc_overlap_years"] = str(N["bc_overlap_years"])
for key in ("raw", "delta", "qdm"):
    for col in ("mean_bias", "p99_bias", "p995_bias", "annual_max_bias"):
        TXT[f"val_{key}_{col}"] = sgn(N[f"val_{key}_{col}"], 2)
    TXT[f"val_{key}_quantile_rmse"] = fmt(N[f"val_{key}_quantile_rmse"], 2); TXT[f"val_{key}_std_ratio"] = fmt(N[f"val_{key}_std_ratio"], 2)
    TXT[f"val_{key}_seasonal_rmse"] = fmt(N[f"val_{key}_seasonal_rmse"], 2)
for k in SC:
    for q in (500, 990, 999):
        TXT[f"{k}_model_change_q{q}_pct"] = sgn((N[f"{k}_model_change_q{q}"] - 1) * 100, 1)
        TXT[f"{k}_qdm_change_q{q}_pct"] = sgn((N[f"{k}_qdm_change_q{q}"] - 1) * 100, 1)
    TXT[f"{k}_mean_hs_qdm"] = fmt(N[f"{k}_swh_annual_mean"], 2)
    if not HAS_TM:
        for kk in ("tm_annual_mean", "tm_ols_per_decade", "tm_ols_p", "tm_mk_p", "tm_change_over_record"):
            N[f"{k}_{kk}"] = float("nan")
    TXT[f"{k}_mean_tm_annual"] = fmt(N[f"{k}_tm_annual_mean"], 2)
    TXT[f"{k}_hs_ols"] = sgn(N[f"{k}_swh_ols_per_decade"], 1); TXT[f"{k}_hs_ols_p"] = fmt(N[f"{k}_swh_ols_p"], 2); TXT[f"{k}_hs_mk_p"] = fmt(N[f"{k}_swh_mk_p"], 2)
    TXT[f"{k}_hs_change"] = sgn(N[f"{k}_swh_change_over_record"], 1)
    TXT[f"{k}_hs_ols_abs"] = fmt(abs(N[f"{k}_swh_ols_per_decade"]), 1)
    TXT[f"{k}_rawhs_ols"] = sgn(N[f"{k}_rawhs_ols_per_decade"], 1); TXT[f"{k}_rawhs_ols_p"] = fmt(N[f"{k}_rawhs_ols_p"], 2)
    TXT[f"{k}_tm_ols"] = sgn(N[f"{k}_tm_ols_per_decade"], 3); TXT[f"{k}_tm_ols_abs"] = fmt(abs(N[f"{k}_tm_ols_per_decade"]), 3)
    TXT[f"{k}_tm_ols_p"] = fmt(N[f"{k}_tm_ols_p"], 2); TXT[f"{k}_tm_mk_p"] = fmt(N[f"{k}_tm_mk_p"], 2)
    TXT[f"{k}_tm_change"] = sgn(N[f"{k}_tm_change_over_record"], 2)
    TXT[f"{k}_tm_change_abs"] = fmt(abs(N[f"{k}_tm_change_over_record"]), 2) if np.isfinite(N[f"{k}_tm_change_over_record"]) else "n/a"
    TXT[f"{k}_tm_change_pct"] = fmt(abs(N[f"{k}_tm_change_over_record"]) / N[f"{k}_tm_annual_mean"] * 100, 0) if np.isfinite(N[f"{k}_tm_annual_mean"]) else "n/a"
    TXT[f"{k}_hs_annual_std"] = fmt(N[f"{k}_swh_annual_std"], 2)
    TXT[f"qdm_mean_change_{k}_cm"] = sgn(N[f"qdm_mean_change_{k}_cm"], 0)
for k in ("ref", "rcp45", "rcp85"):
    TXT[f"{k}_threshold"] = fmt(N[f"{k}_threshold"], 2); TXT[f"{k}_n_raw"] = str(N[f"{k}_n_raw"]); TXT[f"{k}_n_peaks"] = str(N[f"{k}_n_peaks"])
    TXT[f"{k}_rate"] = fmt(N[f"{k}_rate"], 1); TXT[f"{k}_xi"] = sgn(N[f"{k}_xi"], 3); TXT[f"{k}_beta"] = fmt(N[f"{k}_beta"], 2)
    TXT[f"{k}_upper_bound"] = "none" if N[f"{k}_upper_bound"] is None else fmt(N[f"{k}_upper_bound"], 1)
    for T in (10, 25, 50, 100):
        TXT[f"{k}_rl{T}"] = fmt(N[f"{k}_rl{T}"], 2); TXT[f"{k}_rl{T}_lo"] = fmt(N[f"{k}_rl{T}_ci"][0], 2); TXT[f"{k}_rl{T}_hi"] = fmt(N[f"{k}_rl{T}_ci"][1], 2)
    TXT[f"{k}_rl100_ci_width"] = fmt(N[f"{k}_rl100_ci"][1] - N[f"{k}_rl100_ci"][0], 2)
for k in SC:
    TXT[f"{k}_rl100_change_cm"] = sgn(N[f"{k}_rl100_change_cm"], 0)
    TXT[f"raw_{k}_rl100"] = fmt(N[f"raw_{k}_rl100"], 2)
    for m in ("raw", "delta", "qdm"):
        TXT[f"bc_{k}_{m}_rl100"] = fmt(N[f"bc_{k}_{m}_rl100"], 2)
    TXT[f"bc_rl100_spread_{k}"] = fmt(N[f"bc_rl100_spread_{k}"], 2)
    TXT[f"{k}_rl100_grid_min"] = fmt(N[f"{k}_rl100_grid_min"], 2); TXT[f"{k}_rl100_grid_max"] = fmt(N[f"{k}_rl100_grid_max"], 2)
    TXT[f"{k}_grid_xi_maxdev"] = fmt(N[f"{k}_grid_xi_maxdev"], 2)
    TXT[f"{k}_rl100_998_min"] = fmt(N[f"{k}_rl100_998_min"], 2); TXT[f"{k}_rl100_998_max"] = fmt(N[f"{k}_rl100_998_max"], 2)
    TXT[f"{k}_n_998_min"] = str(N[f"{k}_n_998_min"]); TXT[f"{k}_n_998_max"] = str(N[f"{k}_n_998_max"])
    for tag in ("990", "995"):
        TXT[f"{k}_rl100_{tag}_min"] = fmt(N[f"{k}_rl100_{tag}_min"], 2); TXT[f"{k}_rl100_{tag}_max"] = fmt(N[f"{k}_rl100_{tag}_max"], 2)
        TXT[f"{k}_n_{tag}_min"] = str(N[f"{k}_n_{tag}_min"]); TXT[f"{k}_n_{tag}_max"] = str(N[f"{k}_n_{tag}_max"])
        TXT[f"{k}_xi_{tag}_max"] = sgn(N[f"{k}_xi_{tag}_max"], 2)
    TXT[f"{k}_ps_xi_lo"] = sgn(N[f"{k}_ps_xi_range"][0], 2); TXT[f"{k}_ps_xi_hi"] = sgn(N[f"{k}_ps_xi_range"][1], 2)
    for key in ("ts_rl100_first", "ts_rl100_last", "ts_rl100_min", "ts_rl100_max", "ts_mu_first", "ts_mu_last", "ts_sigma_first", "ts_sigma_last"):
        TXT[f"{k}_{key}"] = fmt(N[f"{k}_{key}"], 2)
    TXT[f"{k}_ts_rl100_change"] = sgn(N[f"{k}_ts_rl100_change"], 2); TXT[f"{k}_ts_xi"] = sgn(N[f"{k}_ts_xi"], 3)
    TXT[f"{k}_ts_n_peaks"] = str(N[f"{k}_ts_n_peaks"])
    TXT[f"{k}_ts_band_last_lo"] = fmt(N[f"{k}_ts_band_last"][0], 2); TXT[f"{k}_ts_band_last_hi"] = fmt(N[f"{k}_ts_band_last"][1], 2)
TXT["raw_hist_rl100"] = fmt(N["raw_hist_rl100"], 2)
TXT["raw_hist_rl100_lo"] = fmt(N["raw_hist_rl100_ci"][0], 2); TXT["raw_hist_rl100_hi"] = fmt(N["raw_hist_rl100_ci"][1], 2)
for k in SC:
    TXT[f"raw_change_{k}"] = sgn(N[f"raw_{k}_rl100"] - N["raw_hist_rl100"], 2)
    TXT[f"qdm_minus_raw_{k}"] = fmt(abs(N[f"bc_{k}_qdm_rl100"] - N[f"bc_{k}_raw_rl100"]), 2)  # quoted with "lowers ... by"
    TXT[f"{k}_minus_ref_rl100"] = fmt(abs(N[f"{k}_rl100"] - N["ref_rl100"]), 2)  # quoted with "above"
    for v in ("swh", "tm"):
        dd = 1 if v == "swh" else 3
        ci = N.get(f"{k}_{v}_sen_ci_per_decade")
        TXT[f"{k}_{v}_sen"] = sgn(N[f"{k}_{v}_sen_per_decade"], dd) if N.get(f"{k}_{v}_sen_per_decade") is not None and np.isfinite(N[f"{k}_{v}_sen_per_decade"]) else "n/a"
        TXT[f"{k}_{v}_sen_lo"] = sgn(ci[0], dd) if ci else "n/a"; TXT[f"{k}_{v}_sen_hi"] = sgn(ci[1], dd) if ci else "n/a"
        ci = N.get(f"{k}_{v}_ols_ci_per_decade")
        TXT[f"{k}_{v}_ols_lo"] = sgn(ci[0], dd) if ci else "n/a"; TXT[f"{k}_{v}_ols_hi"] = sgn(ci[1], dd) if ci else "n/a"
TXT["rcp85_minus_rcp45_rl100"] = fmt(abs(N["rcp85_rl100"] - N["rcp45_rl100"]), 2)  # quoted as a distance
TXT["rcp45_hs_annual_std_cm"] = fmt(N["rcp45_swh_annual_std"] * 100, 1); TXT["rcp85_hs_annual_std_cm"] = fmt(N["rcp85_swh_annual_std"] * 100, 1)
for k in SC:
    for tag in ("cap995", "nocap"):
        TXT[f"{k}_rl100_{tag}"] = fmt(N[f"{k}_rl100_{tag}"], 2); TXT[f"{k}_xi_{tag}"] = sgn(N[f"{k}_xi_{tag}"], 3)
TXT["qdm_tail_pct"] = f"{P.QDM_TAIL_PROB * 100:g}"
# what the capped correction leaves of the model's tail bias
TXT["qdm_q99_factor"] = fmt(1.0 / N["hist_ref_q99_ratio"], 2)
TXT["hist_minus_ref_rl100"] = fmt(abs(N["raw_hist_rl100"] - N["ref_rl100"]), 2)  # quoted with "above"
for k in SC:
    refraw = N["ref_rl100"] + (N[f"raw_{k}_rl100"] - N["raw_hist_rl100"])
    N[f"{k}_ref_plus_raw_change_rl100"] = round(refraw, 2)
    TXT[f"{k}_ref_plus_raw_change_rl100"] = fmt(refraw, 2)
    TXT[f"{k}_qdm_minus_refraw_rl100"] = fmt(abs(N[f"bc_{k}_qdm_rl100"] - refraw), 2)  # quoted with "higher"
    five = [N[f"bc_{k}_raw_rl100"], N[f"bc_{k}_delta_rl100"], N[f"bc_{k}_qdm_rl100"], N[f"{k}_rl100_cap995"], N[f"{k}_rl100_nocap"]]
    N[f"bc_rl100_span5_{k}"] = round(max(five) - min(five), 2)
    TXT[f"bc_rl100_span5_{k}"] = fmt(max(five) - min(five), 2)
    TXT[f"bc_rl100_span5_{k}_lo"] = fmt(min(five), 2); TXT[f"bc_rl100_span5_{k}_hi"] = fmt(max(five), 2)
    TXT[f"qdm_mean_change_{k}_cm"] = sgn(N[f"qdm_mean_change_{k}_cm"], 1)
    TXT[f"{k}_model_change_q990_pct"] = sgn((N[f"{k}_model_change_q990"] - 1) * 100, 1)
TXT["rl100_diff_cm"] = str(int(abs(N["rl100_diff_cm"])))
TXT["scenario_mean_diff_cm"] = sgn(N["scenario_mean_diff_cm"], 1)
TXT["seasonal_max_abs_diff_cm"] = str(int(round(N["seasonal_max_abs_diff_cm"]))); TXT["seasonal_max_abs_diff_month"] = MONTH[N["seasonal_max_abs_diff_month"]]
TXT["winter_change_rcp85_cm"] = sgn(N["winter_change_rcp85_cm"], 1); TXT["summer_change_rcp85_cm"] = sgn(N["summer_change_rcp85_cm"], 1)
TXT["seasonal_max_abs_diff_cm"] = fmt(N["seasonal_max_abs_diff_cm"], 1)
TXT["grid_rl100_range_m"] = fmt(N["grid_rl100_range_cm"] / 100, 2)
TXT["ts_window"] = str(WINDOW); TXT["ts_max_dev_from_stationary"] = fmt(N["ts_max_dev_from_stationary"], 2)
for k in DIRS:
    TXT[f"{k}_dir_dominant"] = N[f"{k}_dir_dominant"]; TXT[f"{k}_dir_dominant_share"] = fmt(N[f"{k}_dir_dominant_share"], 0)
    TXT[f"{k}_dir_within45"] = fmt(N[f"{k}_dir_within45"], 0); TXT[f"{k}_dir_storm_mean"] = fmt(N[f"{k}_dir_storm_mean"], 0)
    TXT[f"{k}_dir_all_mean"] = fmt(N[f"{k}_dir_all_mean"], 0); TXT[f"{k}_dir_n"] = str(N[f"{k}_dir_n"])
    if f"{k}_dir_wnw_fisher_p" in N:
        _pv = N[f"{k}_dir_wnw_fisher_p"]; TXT[f"{k}_dir_wnw_fisher_p"] = "< 0.001" if _pv < 0.001 else f"= {_pv:.3f}"
    for key in ("dir_all_dominant_share", "dir_w_nw", "dir_all_w_nw", "dir_within45_ref"):
        if f"{k}_{key}" in N:
            TXT[f"{k}_{key}"] = fmt(N[f"{k}_{key}"], 0)
TXT["pyx_version"] = str(N["pyx_version"])
TXT["pyx_max_abs_diff"] = "less than $10^{-3}$" if N["pyx_max_abs_diff"] < 1e-3 else fmt(N["pyx_max_abs_diff"], 3)
TXT["bri_nan_hs"] = str(N["bri_nan_hs"]); TXT["bri_mean_hs"] = fmt(N["bri_mean_hs"], 2); TXT["bri_max_hs"] = fmt(N["bri_max_hs"], 2)
TXT["bri_month_max"] = MONTH[N["bri_month_max"]]; TXT["bri_month_min"] = MONTH[N["bri_month_min"]]
TXT["bri_threshold"] = fmt(N["bri_threshold"], 2); TXT["bri_n_raw"] = str(N["bri_n_raw"]); TXT["bri_n_peaks"] = str(N["bri_n_peaks"])
TXT["bri_n_peaks_grid_max"] = str(N["bri_n_peaks_grid_max"])
_bf = N["bri_grid_fit"]
if _bf:
    _pcts = sorted({p for p, g in _bf}); _gaps = sorted({g for p, g in _bf})
    TXT["bri_grid_fit_settings"] = (f"the {_pcts[0]:g}th percentile with a gap of " + " or ".join(str(g) for g in _gaps) + " hours") if len(_pcts) == 1 else "; ".join(f"{p:g}th percentile, {g} hours" for p, g in _bf)
    TXT["bri_grid_fit_n"] = {1: "one", 2: "two", 3: "three"}.get(len(_bf), str(len(_bf)))
else:
    TXT["bri_grid_fit_settings"] = "none"; TXT["bri_grid_fit_n"] = "no"
TXT["pct"] = f"{PCT:g}"; TXT["gap"] = str(GAP); TXT["nboot"] = str(NBOOT); TXT["release"] = "v1.1.0"
TXT["boot500_max_diff"] = fmt(N["boot500_max_diff"], 2)
# run time of the one-click demonstration, measured separately (seconds); set by environment variables
TXT["demo_run_local"] = os.environ.get("DEMO_RUN_LOCAL", "n/a"); TXT["demo_run_cloud"] = os.environ.get("DEMO_RUN_CLOUD", "n/a")

with open(os.path.join(TAB, "rows.tex"), "w") as f:
    f.write("% Generated by make_figures.py. Do not edit by hand.\n")
    for k, v in ROWS.items():
        v = _re.sub(r"\[([^\]]*?)\]", r"\\mbox{[\1]}", v)
        v = _re.sub(r"(?<![\w$\\-])-(?=\d)", "$-$", v)          # true minus signs for negative numbers
        f.write("\\newcommand{\\Rows" + k + "}{%\n" + v + "}\n")
with open(os.path.join(HERE, "numbers.tex"), "w") as f:
    f.write("% Generated by make_figures.py. Do not edit by hand. Use as \\N{key} in the text.\n")
    f.write("\\newcommand{\\N}[1]{\\csname numval:#1\\endcsname}\n")
    for k, v in TXT.items():
        f.write("\\expandafter\\newcommand\\csname numval:" + k + "\\endcsname{" + v + "}\n")
with open(os.path.join(HERE, "numbers.json"), "w") as f:
    json.dump(N, f, indent=1, default=lambda o: o if not isinstance(o, (np.integer, np.floating, np.bool_)) else o.item())
print("wrote tables/rows.tex, numbers.tex, numbers.json")
