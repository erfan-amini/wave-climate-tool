"""
Projection methods for the Wave Climate Trend-Extreme Tool.

Three things live here:

1. Bias correction of a climate-model wave series against a reference record
   at the same site, by the delta-change method and by quantile delta mapping
   (QDM, Cannon et al. 2015), both applied by calendar month.
2. A split-sample validation of those corrections on held-out years.
3. Non-stationary extreme-value analysis by the transformed-stationary
   approach (TS-EVA, Mentaschi et al. 2016), which returns time-varying
   return levels on top of the stationary POT-GPD machinery in analysis.py.

References
    Cannon, A. J., Sobie, S. R., Murdock, T. Q. (2015). Bias correction of GCM
    precipitation by quantile mapping: how well do methods preserve changes
    in quantiles and extremes? Journal of Climate 28, 6938-6959.
    Mentaschi, L., Vousdoukas, M., Voukouvalas, E., Sartini, L., Feyen, L.,
    Besio, G., Alfieri, L. (2016). The transformed-stationary approach: a
    generic and simplified methodology for non-stationary extreme value
    analysis. Hydrology and Earth System Sciences 20, 3527-3547.
"""

import numpy as np
import pandas as pd

import analysis as A

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _monthly_groups(s):
    """Yield (month, values) for a time-indexed Series."""
    for m in range(1, 13):
        yield m, s[s.index.month == m]


def _ecdf_quantile(sorted_vals, tau):
    """Quantile of an empirical distribution at probability tau, with linear
    interpolation between order statistics and flat extrapolation beyond."""
    n = len(sorted_vals)
    pos = tau * (n - 1)
    lo = np.clip(np.floor(pos).astype(int), 0, n - 1)
    hi = np.clip(lo + 1, 0, n - 1)
    w = pos - lo
    return sorted_vals[lo] * (1 - w) + sorted_vals[hi] * w


def _ecdf_prob(sorted_vals, x):
    """Non-exceedance probability of x in an empirical distribution
    (Weibull plotting position, kept inside (0, 1))."""
    n = len(sorted_vals)
    r = np.searchsorted(sorted_vals, x, side="right")
    return np.clip(r / (n + 1.0), 0.5 / (n + 1.0), 1 - 0.5 / (n + 1.0))


# ---------------------------------------------------------------------------
# 1. Bias correction
# ---------------------------------------------------------------------------

def delta_change(ref, hist, fut, kind="multiplicative"):
    """Delta-change projection.

    The model's change in the monthly mean between its historical run and
    its future run is applied to the reference record, month by month. The
    result keeps the reference record's own variability and length and
    shifts (additive) or scales (multiplicative) its level.

    Returns (projected Series over the reference period, DataFrame of monthly
    factors).
    """
    rows, out = [], ref.copy()
    for m, r in _monthly_groups(ref):
        h = hist[hist.index.month == m].mean()
        f = fut[fut.index.month == m].mean()
        if kind == "multiplicative":
            fac = f / h
            out[out.index.month == m] = r * fac
        else:
            fac = f - h
            out[out.index.month == m] = r + fac
        rows.append({"month": m, "hist_mean": h, "fut_mean": f, "factor": fac})
    return out, pd.DataFrame(rows)


def quantile_delta_mapping(ref, hist, fut, kind="multiplicative", n_quantiles=None):
    """Quantile delta mapping (Cannon et al. 2015), by calendar month.

    For every value in the model's future series: find its non-exceedance
    probability tau in the model future distribution; take the model's
    historical quantile at tau and the reference quantile at tau; the
    corrected value is the reference quantile scaled (multiplicative) or
    shifted (additive) by the model's own change at that quantile. Changes
    projected by the model are preserved at every quantile, including the
    tail, and the distribution is matched to the reference at the same time.

    ref, hist: Series over the same calibration period (any overlap is
    allowed; both are used as whole distributions per month).
    fut: model series to correct (the future run, or a held-out part of the
    historical run for validation).

    Returns the corrected Series over the period of fut.
    """
    out = fut.copy()
    for m in range(1, 13):
        r = np.sort(ref[ref.index.month == m].dropna().values)
        h = np.sort(hist[hist.index.month == m].dropna().values)
        sel = fut.index.month == m
        f = fut[sel].dropna()
        if len(r) < 10 or len(h) < 10 or len(f) == 0:
            continue
        fs = np.sort(f.values)
        tau = _ecdf_prob(fs, f.values)
        q_hist = _ecdf_quantile(h, tau)
        q_ref = _ecdf_quantile(r, tau)
        if kind == "multiplicative":
            q_hist = np.maximum(q_hist, 1e-6)
            corrected = q_ref * (f.values / q_hist)
        else:
            corrected = q_ref + (f.values - q_hist)
        out.loc[f.index] = corrected
    return out


def split_years(index, mode="alternate"):
    """Split the years of a time index into calibration and validation sets.
    'alternate' puts odd years in calibration and even years in validation;
    'halves' puts the first half in calibration."""
    years = np.array(sorted(set(index.year)))
    if mode == "halves":
        cut = len(years) // 2
        return years[:cut], years[cut:]
    return years[years % 2 == 1], years[years % 2 == 0]


def _sel_years(s, years):
    return s[np.isin(s.index.year, years)]


def validation_metrics(ref, corrected):
    """Skill of a corrected model series against the reference over the
    same years. All quantities in the units of the variable."""
    r, c = ref.dropna(), corrected.dropna()
    common = r.index.intersection(c.index)
    if len(common) > 100:
        r, c = r.loc[common], c.loc[common]
    qs = np.concatenate([np.linspace(0.01, 0.99, 99), [0.995, 0.999]])
    qr, qc = np.quantile(r.values, qs), np.quantile(c.values, qs)
    mr, mc = A.monthly_climatology(r), A.monthly_climatology(c)
    amr, amc = r.groupby(r.index.year).max(), c.groupby(c.index.year).max()
    return {
        "mean_bias": float(c.mean() - r.mean()),
        "std_ratio": float(c.std() / r.std()),
        "seasonal_rmse": float(np.sqrt(np.mean((mc.values - mr.values) ** 2))),
        "quantile_rmse": float(np.sqrt(np.mean((qc - qr) ** 2))),
        "p99_bias": float(np.quantile(c.values, 0.99) - np.quantile(r.values, 0.99)),
        "p995_bias": float(np.quantile(c.values, 0.995) - np.quantile(r.values, 0.995)),
        "annual_max_bias": float(amc.mean() - amr.mean()),
        "n": int(len(r)),
    }


def validate_corrections(ref, hist, kind="multiplicative", mode="alternate"):
    """Split-sample test of the bias corrections.

    Calibration years are used to fit the corrections (reference and model
    historical). Validation years of the model historical run are then
    corrected as if they were a future period and compared with the
    reference in those years. Three series are scored: the raw model, the
    delta-change projection and the QDM projection.

    Returns (DataFrame of metrics by method, dict of the validation-period
    series for plotting).
    """
    cal, val = split_years(ref.index, mode)
    ref_cal, ref_val = _sel_years(ref, cal), _sel_years(ref, val)
    hist_cal, hist_val = _sel_years(hist, cal), _sel_years(hist, val)
    raw = hist_val
    delta, _ = delta_change(ref_cal, hist_cal, hist_val, kind=kind)
    qdm = quantile_delta_mapping(ref_cal, hist_cal, hist_val, kind=kind)
    series = {"reference": ref_val, "raw model": raw, "delta change": delta, "QDM": qdm}
    rows = []
    for name in ("raw model", "delta change", "QDM"):
        m = validation_metrics(ref_val, series[name])
        m["method"] = name
        rows.append(m)
    table = pd.DataFrame(rows).set_index("method")
    return table, series, {"calibration_years": cal.tolist(), "validation_years": val.tolist()}


def project(ref, hist, fut, method="QDM", kind="multiplicative"):
    """Produce a projected series by the chosen method: 'raw' (model future
    as is), 'delta' (reference shifted by the model change) or 'QDM'."""
    if method == "raw":
        return fut.copy()
    if method == "delta":
        return delta_change(ref, hist, fut, kind=kind)[0]
    if method == "QDM":
        return quantile_delta_mapping(ref, hist, fut, kind=kind)
    raise ValueError(f"unknown method {method}")


# ---------------------------------------------------------------------------
# 2. Non-stationary extremes: transformed-stationary approach
# ---------------------------------------------------------------------------

def ts_transform(s, window_years=30):
    """Running mean and running standard deviation of a series over a
    centred window, and the standardised series y = (x - mu) / sigma.
    The window shrinks at the record ends (min_periods = half window)."""
    step_h = float(pd.Series(s.index).diff().dt.total_seconds().mode().iloc[0]) / 3600.0
    w = int(round(window_years * 365.25 * 24.0 / step_h))
    mu = s.rolling(w, center=True, min_periods=w // 2).mean()
    sd = s.rolling(w, center=True, min_periods=w // 2).std()
    y = (s - mu) / sd
    return y.dropna(), mu, sd


def ts_eva(s, window_years=30, percentile=99.5, min_gap_hours=72, return_periods=(10, 50, 100), n_boot=500, seed=0):
    """Transformed-stationary extreme-value analysis.

    1. Transform the series to y = (x - mu(t)) / sigma(t) with running
       statistics over `window_years`.
    2. Run the stationary POT-GPD analysis of analysis.py on y.
    3. Map the return levels of y back to each year:
       RL_x(t, T) = mu(t) + sigma(t) * RL_y(T).

    Returns a dict with the stationary result on y, the annual series of mu
    and sigma, and a DataFrame of return levels by year with bootstrap bands.
    """
    y, mu, sd = ts_transform(s, window_years)
    res_y = A.pot_analysis(y, percentile=percentile, min_gap_hours=min_gap_hours,
                           return_periods=return_periods, n_boot=n_boot, seed=seed)
    mu_y = mu.groupby(mu.index.year).mean()
    sd_y = sd.groupby(sd.index.year).mean()
    years = mu_y.dropna().index
    rows = []
    for yr in years:
        row = {"year": int(yr), "mu": float(mu_y[yr]), "sigma": float(sd_y[yr])}
        for T, rl, lo, hi in zip(res_y["return_periods"], res_y["return_levels"], *res_y["rl_ci"]):
            row[f"RL_{int(T)}"] = mu_y[yr] + sd_y[yr] * rl
            row[f"RL_{int(T)}_lo"] = mu_y[yr] + sd_y[yr] * lo
            row[f"RL_{int(T)}_hi"] = mu_y[yr] + sd_y[yr] * hi
        rows.append(row)
    table = pd.DataFrame(rows).set_index("year")
    return {"result_y": res_y, "mu": mu, "sigma": sd, "table": table, "window_years": window_years,
            "peaks_time": res_y["peaks"].index}


def ts_eva_summary(ts, stationary_rl, T=100):
    """Compare the time-varying T-year level with the stationary one:
    first-year, last-year, change over the record, and the range."""
    col = f"RL_{T}"
    t = ts["table"]
    return {
        "first_year": int(t.index[0]), "last_year": int(t.index[-1]),
        "rl_first": float(t[col].iloc[0]), "rl_last": float(t[col].iloc[-1]),
        "change": float(t[col].iloc[-1] - t[col].iloc[0]),
        "rl_min": float(t[col].min()), "rl_max": float(t[col].max()),
        "stationary": float(stationary_rl),
        "mean_ts": float(t[col].mean()),
    }
