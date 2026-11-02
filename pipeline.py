"""
One-call analysis pipeline for the Wave Climate Trend-Extreme Tool.

run(records, targets, options) does everything the report needs:

1. Optional bias correction of every model-future target against a
   reference record (delta change, quantile delta mapping, or both), with a
   split-sample validation on the overlap years.
2. Trends in the annual means of wave height and wave period.
3. Stationary peaks-over-threshold analysis of the chosen variable.
4. Optional transformed-stationary (non-stationary) analysis.
5. Optional threshold diagnostics and a threshold / storm-gap sensitivity grid.

No interface code here; the app and the paper's figure script both call it.
"""

import time

import numpy as np
import pandas as pd

import analysis as A
import projection as P

DEFAULTS = {
    "variable": "swh",            # variable for the extreme-value analysis
    "percentile": 99.5,
    "gap": 72,
    "n_boot": 500,
    "return_periods": (2, 5, 10, 25, 50, 100, 200),
    "nonstationary": True,
    "window": 30,
    "correction": "off",          # off | delta | qdm | compare
    "kind": "multiplicative",
    "split": "alternate",
    "diagnostics": True,
    "sensitivity": True,
    "synthetic_reference": False,
    "seed": 0,
}
KEY_PERIODS = (10, 25, 50, 100)
METHOD_LABEL = {"raw": "raw model", "delta": "delta change", "qdm": "QDM"}


def options(**kw):
    o = dict(DEFAULTS)
    o.update({k: v for k, v in kw.items() if v is not None})
    return o


def _series(rec, var):
    df = rec["df"]
    return A.to_series(df, var) if var in df.columns else None


def find_bias_inputs(records, var, synthetic=False):
    """Pick a reference and a model-historical record that overlap by at
    least two years. With synthetic=True and no real pair, build a labelled
    synthetic pair from the first twenty years of the longest model-future
    record (reference = model x 1.2 + 0.1). Returns dict or None."""
    refs = [n for n, r in records.items() if r["role"] in ("observed", "reference") and var in r["df"].columns]
    hists = [n for n, r in records.items() if r["role"] == "model historical" and var in r["df"].columns]
    for h in hists:
        hs = _series(records[h], var)
        for rname in refs:
            rs = _series(records[rname], var)
            overlap = sorted(set(rs.index.year) & set(hs.index.year))
            if len(overlap) >= 2:
                return {"ref_name": rname, "hist_name": h, "overlap": overlap, "synthetic": False,
                        "ref": rs[np.isin(rs.index.year, overlap)], "hist": hs[np.isin(hs.index.year, overlap)]}
    if not synthetic:
        return None
    futs = [n for n, r in records.items() if r["role"] == "model future" and var in r["df"].columns]
    if not futs:
        return None
    longest = max(futs, key=lambda n: records[n]["info"]["span_years"])
    base = _series(records[longest], var)
    yrs = sorted(set(base.index.year))[:20]
    h = base[np.isin(base.index.year, yrs)]
    rng = np.random.default_rng(0)
    ref = h * 1.2 + 0.1 + rng.normal(0, 0.02, len(h))
    return {"ref_name": "synthetic reference (model x 1.2 + 0.1)", "hist_name": f"first {len(yrs)} years of {longest}",
            "overlap": yrs, "synthetic": True, "ref": ref, "hist": h}


def bias_available(records, var):
    return find_bias_inputs(records, var, synthetic=False) is not None


def run(records, targets, opts=None, progress=None):
    """records: {name: {"df", "info", "role", "source"}}; targets: names to analyse."""
    o = options(**(opts or {}))
    t0 = time.time()
    var = o["variable"]
    say = progress or (lambda msg: None)
    R = {"opts": o, "targets": list(targets), "series": {}, "role": {}, "trends": {"swh": {}, "tm": {}},
         "pot": {}, "ts": {}, "diag": {}, "sens": {}, "skipped": {}, "bc": None, "notes": []}

    # 1. Bias correction --------------------------------------------------
    primary_method = {"off": "raw", "delta": "delta", "qdm": "qdm", "compare": "qdm"}[o["correction"]]
    bc_inputs = None
    if o["correction"] != "off":
        bc_inputs = find_bias_inputs(records, var, synthetic=o["synthetic_reference"])
        if bc_inputs is None:
            R["notes"].append("Bias correction was requested but no reference record overlaps a model historical run; "
                              "the raw model series were analysed.")
            primary_method = "raw"
        else:
            say("Validating the bias correction on held-out years")
            vtab, vser, vinfo = P.validate_corrections(bc_inputs["ref"], bc_inputs["hist"], kind=o["kind"], mode=o["split"])
            R["bc"] = {"inputs": {k: bc_inputs[k] for k in ("ref_name", "hist_name", "overlap", "synthetic")},
                       "validation": vtab, "val_series": vser, "val_info": vinfo, "projected": {}, "compare_pot": {}}

    # analysed series: projected (if a correction applies) or as recorded
    for name in targets:
        rec = records[name]
        s = _series(rec, var)
        if s is None:
            R["skipped"][name] = f"no {var} column"
            continue
        label = name
        if R["bc"] is not None and rec["role"] == "model future":
            methods = ["raw", "delta", "qdm"] if o["correction"] == "compare" else [primary_method]
            proj = {}
            for m in methods:
                proj[m] = P.project(bc_inputs["ref"], bc_inputs["hist"], s, method={"raw": "raw", "delta": "delta", "qdm": "QDM"}[m], kind=o["kind"])
            R["bc"]["projected"][name] = proj
            s = proj[primary_method]
            label = f"{name} ({METHOD_LABEL[primary_method]})" if primary_method != "raw" else name
        R["series"][label] = s
        R["role"][label] = rec["role"]

    # 2. Trends (both variables, on the series as recorded) -----------------
    say("Testing the annual means for trends")
    for name in targets:
        for v in ("swh", "tm"):
            sv = _series(records[name], v)
            if sv is not None:
                R["trends"][v][name] = {"trend": A.trend_analysis(sv), "annual": A.annual_means(sv)}
    if R["bc"] is not None:
        for name, proj in R["bc"]["projected"].items():
            for m, sp in proj.items():
                if m == "qdm":  # delta change keeps the reference sequence, so it has no projected trend of its own
                    R["trends"][var][f"{name} ({METHOD_LABEL[m]})"] = {"trend": A.trend_analysis(sp), "annual": A.annual_means(sp)}
        if any("delta" in proj for proj in R["bc"]["projected"].values()):
            R["notes"].append("Delta change keeps the reference record's own sequence and dates, so the delta-change series carry no "
                              "projected trend; their trend test and non-stationary fit are skipped.")

    # 3-5. Extremes --------------------------------------------------------
    for label, s in R["series"].items():
        span = A.record_length_years(pd.Series(s.index))
        say(f"Fitting extremes: {label}")
        try:
            R["pot"][label] = A.pot_analysis(s, o["percentile"], o["gap"], return_periods=o["return_periods"], n_boot=o["n_boot"], seed=o["seed"])
        except ValueError as e:
            R["skipped"][label] = str(e)
            continue
        if span < 10:
            R["notes"].append(f"{label} spans {span:.1f} years; its return levels are not design values.")
        is_delta = label.endswith(f"({METHOD_LABEL['delta']})")
        if o["nonstationary"] and is_delta:
            pass
        elif o["nonstationary"] and span >= 1.5 * o["window"]:
            say(f"Non-stationary fit: {label}")
            try:
                R["ts"][label] = P.ts_eva(s, window_years=o["window"], percentile=o["percentile"], min_gap_hours=o["gap"],
                                          return_periods=o["return_periods"], n_boot=o["n_boot"], seed=o["seed"])
            except ValueError as e:
                R["notes"].append(f"Non-stationary fit skipped for {label}: {e}")
        elif o["nonstationary"]:
            R["notes"].append(f"Non-stationary fit skipped for {label}: record shorter than 1.5 windows ({span:.0f} years).")
        if o["diagnostics"]:
            say(f"Threshold diagnostics: {label}")
            R["diag"][label] = (A.mean_residual_life(s, np.arange(90, 99.91, 0.25)),
                                A.parameter_stability(s, np.arange(97.0, 99.91, 0.2), o["gap"]))
        if o["sensitivity"] and span >= 10:
            say(f"Sensitivity grid: {label}")
            R["sens"][label] = A.sensitivity_grid(s, (99.0, 99.5, 99.8), (24, 48, 72), T=100)

    if R["bc"] is not None and o["correction"] == "compare":
        say("Comparing correction methods")
        for name, proj in R["bc"]["projected"].items():
            for m, sp in proj.items():
                if m == primary_method:
                    R["bc"]["compare_pot"][(name, m)] = R["pot"].get(f"{name} ({METHOD_LABEL[m]})")
                else:
                    R["bc"]["compare_pot"][(name, m)] = A.pot_analysis(sp, o["percentile"], o["gap"], return_periods=(100,), n_boot=o["n_boot"], seed=o["seed"])

    R["seconds"] = time.time() - t0
    R["finished"] = pd.Timestamp.now(tz="UTC")
    return R


# ---------------------------------------------------------------------------
# Summaries used by the report
# ---------------------------------------------------------------------------

def key_levels(res, periods=KEY_PERIODS):
    """{T: (level, lo, hi)} for the requested periods, computed from the fit
    (the bootstrap band is interpolated in log T when T is not in the run)."""
    out = {}
    T = res["return_periods"]
    for p in periods:
        lvl = float(A.return_level(res["threshold"], res["xi"], res["beta"], res["rate_per_year"], p))
        if p in T:
            i = list(T).index(p)
            lo, hi = float(res["rl_ci"][0][i]), float(res["rl_ci"][1][i])
        else:
            lo = float(np.interp(np.log(p), np.log(T), res["rl_ci"][0]))
            hi = float(np.interp(np.log(p), np.log(T), res["rl_ci"][1]))
        out[p] = (lvl, lo, hi)
    return out
