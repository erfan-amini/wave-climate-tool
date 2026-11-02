"""
Core analysis for the Wave Climate Trend-Extreme Tool.

Pure functions, no GUI dependency. Three parts:

1. Data loading: flexible CSV reader for hourly (or sub-hourly) wave records.
2. Trend analysis: annual means, OLS trend with confidence interval,
   Mann-Kendall test and Sen's slope.
3. Extreme-value analysis: peaks over threshold with runs declustering,
   Generalized Pareto fit (location fixed at the threshold), return levels
   with bootstrap confidence intervals, and threshold diagnostics.

References
    Coles, S. (2001). An Introduction to Statistical Modeling of Extreme
    Values. Springer. Chapter 4 (threshold models) and Section 5.3
    (declustering).
    Gilbert, R. O. (1987). Statistical Methods for Environmental Pollution
    Monitoring. Van Nostrand Reinhold. Chapter 16 (Mann-Kendall, Sen's slope).
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import genpareto

# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

TIME_NAMES = ["time", "date/time (aest)", "date/time", "datetime", "date", "timestamp"]
SWH_NAMES = ["swh", "hs", "hs (m)", "h(s)", "significant_wave_height", "significant wave height", "vhm0"]
TM_NAMES = ["tm", "mp1", "tm (s)", "tm(s)", "mean_wave_period", "mean wave period", "vtm02", "tz", "tz (s)"]
TP_NAMES = ["tp", "tp (s)", "peak period", "peak_period", "vtpk"]
SENTINELS = [-99.9, -99.90, -99.0, -999.0, -9999.0]


def _match(columns, names):
    """Return the first column whose lower-cased, stripped name is in names."""
    for c in columns:
        if str(c).strip().lower() in names:
            return c
    return None


def load_series(path_or_buffer):
    """Read a wave record from CSV.

    Accepts any file with a time column and at least one of a significant
    wave height column and a wave period column. Column names are matched
    case-insensitively against common variants (swh, Hs, Hs (m); tm, mp1,
    Tz; tp, Tp (s)). Missing-value sentinels (-99.9 and the like) become NaN.

    Returns (df, info). df has columns 'time' plus 'swh' and/or 'tm' (and 'tp'
    when present), sorted by time with duplicate timestamps removed. info is a
    dict describing what was found.
    """
    name = getattr(path_or_buffer, "name", str(path_or_buffer))
    compression = "gzip" if str(name).lower().endswith(".gz") else None
    raw = pd.read_csv(path_or_buffer, compression=compression)
    tcol = _match(raw.columns, TIME_NAMES)
    if tcol is None:
        raise ValueError(f"No time column found. Columns: {list(raw.columns)}")

    out = pd.DataFrame({"time": pd.to_datetime(raw[tcol], format="mixed", errors="coerce")})
    found = {}
    for key, names in (("swh", SWH_NAMES), ("tm", TM_NAMES), ("tp", TP_NAMES)):
        col = _match(raw.columns, names)
        if col is not None:
            vals = pd.to_numeric(raw[col], errors="coerce").replace(SENTINELS, np.nan)
            out[key] = vals.values
            found[key] = str(col)
    if "swh" not in out.columns and "tm" not in out.columns:
        raise ValueError(f"No wave height or period column found. Columns: {list(raw.columns)}")

    out = out.dropna(subset=["time"]).sort_values("time")
    n_dup = int(out["time"].duplicated().sum())
    out = out.drop_duplicates(subset="time", keep="first").reset_index(drop=True)

    step = out["time"].diff().dt.total_seconds().mode()
    step_h = float(step.iloc[0]) / 3600.0 if len(step) else np.nan
    info = {
        "columns": found,
        "n_rows": int(len(out)),
        "start": out["time"].iloc[0],
        "end": out["time"].iloc[-1],
        "step_hours": step_h,
        "duplicates_removed": n_dup,
        "span_years": record_length_years(out["time"]),
    }
    return out, info


def record_length_years(times):
    """Length of the record in years, from first to last timestamp plus one step."""
    t = pd.to_datetime(times)
    if len(t) < 2:
        return 0.0
    step = (t.iloc[1] - t.iloc[0]) if hasattr(t, "iloc") else (t[1] - t[0])
    span = (t.max() - t.min()) + step
    return float(span.total_seconds() / (365.25 * 86400.0))


def to_series(df, variable):
    """Return a pandas Series of `variable` indexed by time, NaNs dropped."""
    s = pd.Series(df[variable].values, index=pd.DatetimeIndex(df["time"]), name=variable)
    return s.dropna()


# ---------------------------------------------------------------------------
# 2. Climatology and trends
# ---------------------------------------------------------------------------

def summary_stats(s):
    """Mean, standard deviation, percentiles and maximum of a series."""
    return {
        "n": int(s.size),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "p50": float(s.quantile(0.50)),
        "p90": float(s.quantile(0.90)),
        "p99": float(s.quantile(0.99)),
        "max": float(s.max()),
    }


def annual_means(s, min_coverage=0.8):
    """Annual means of a series. Years with less than min_coverage of the
    expected number of samples are dropped so that a partial year does not
    bias the trend."""
    step_h = float(pd.Series(s.index).diff().dt.total_seconds().mode().iloc[0]) / 3600.0
    counts = s.groupby(s.index.year).size()
    expected = 365.25 * 24.0 / step_h
    keep = counts[counts >= min_coverage * expected].index
    am = s.groupby(s.index.year).mean()
    return am.loc[keep]


def monthly_climatology(s):
    """Mean by calendar month (1 to 12)."""
    return s.groupby(s.index.month).mean()


def ols_trend(years, values, alpha=0.05):
    """Ordinary least squares trend of annual values on year."""
    x = np.asarray(years, dtype=float)
    y = np.asarray(values, dtype=float)
    res = stats.linregress(x, y)
    n = len(x)
    tcrit = stats.t.ppf(1 - alpha / 2, n - 2)
    return {
        "slope": float(res.slope),
        "slope_ci": (float(res.slope - tcrit * res.stderr), float(res.slope + tcrit * res.stderr)),
        "intercept": float(res.intercept),
        "p_value": float(res.pvalue),
        "r2": float(res.rvalue ** 2),
        "n": int(n),
    }


def mann_kendall(values):
    """Mann-Kendall test for a monotonic trend. Returns S, the test statistic
    Z (with continuity correction and tie correction) and the two-sided p."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    s = 0
    for i in range(n - 1):
        s += np.sign(x[i + 1:] - x[i]).sum()
    # variance with tie correction
    _, counts = np.unique(x, return_counts=True)
    tie_term = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"S": float(s), "Z": float(z), "p_value": float(p), "var_S": float(var_s)}


def sens_slope(years, values, alpha=0.05):
    """Sen's slope with a confidence interval (Gilbert 1987, Section 16.5)."""
    x = np.asarray(years, dtype=float)
    y = np.asarray(values, dtype=float)
    n = len(x)
    slopes = []
    for i in range(n - 1):
        dx = x[i + 1:] - x[i]
        dy = y[i + 1:] - y[i]
        slopes.extend((dy / dx).tolist())
    slopes = np.sort(np.asarray(slopes))
    N = len(slopes)
    mk = mann_kendall(y)
    c = stats.norm.ppf(1 - alpha / 2) * np.sqrt(mk["var_S"])
    m1 = int(np.floor((N - c) / 2))
    m2 = int(np.ceil((N + c) / 2))
    lo = slopes[max(m1 - 1, 0)] if N else np.nan
    hi = slopes[min(m2, N - 1)] if N else np.nan
    return {"slope": float(np.median(slopes)), "slope_ci": (float(lo), float(hi))}


def trend_analysis(s, min_years=10):
    """Trend in annual means of a series. Returns None if the record is too
    short for a trend to mean anything."""
    am = annual_means(s)
    if len(am) < min_years:
        return None
    years = am.index.values
    vals = am.values
    out = {"annual_means": am, "years": years}
    out.update({"ols": ols_trend(years, vals), "mk": mann_kendall(vals), "sen": sens_slope(years, vals)})
    horizon = float(years[-1] - years[0])
    out["change_over_record"] = {
        "ols": out["ols"]["slope"] * horizon,
        "sen": out["sen"]["slope"] * horizon,
        "years": horizon,
    }
    return out


# ---------------------------------------------------------------------------
# 3. Extreme-value analysis
# ---------------------------------------------------------------------------

def decluster(s, threshold, min_gap_hours):
    """Runs declustering (Coles 2001, Section 5.3.4).

    Exceedances of `threshold` separated by a gap longer than `min_gap_hours`
    belong to different storms. Each storm contributes its single highest
    value. Returns a Series of storm peaks indexed by the time of the peak.
    """
    exc = s[s > threshold]
    if exc.empty:
        return exc
    if min_gap_hours <= 0:
        return exc
    gaps = np.diff(exc.index.values).astype("timedelta64[s]").astype(float) / 3600.0
    new_cluster = np.concatenate([[True], gaps > min_gap_hours])
    cluster_id = np.cumsum(new_cluster)
    peaks = exc.groupby(cluster_id).agg(["idxmax", "max"])
    return pd.Series(peaks["max"].values, index=pd.DatetimeIndex(peaks["idxmax"].values), name=s.name)


def fit_gpd(exceedances):
    """Maximum-likelihood GPD fit to exceedances over the threshold, with the
    location parameter fixed at zero. Returns (xi, beta, nll)."""
    y = np.asarray(exceedances, dtype=float)
    xi, _, beta = genpareto.fit(y, floc=0)
    nll = -np.sum(genpareto.logpdf(y, xi, 0, beta))
    return float(xi), float(beta), float(nll)


def return_level(u, xi, beta, lam, T):
    """Return level for return period T (years) from the GPD fit, with
    exceedance rate lam (storm peaks per year). Coles (2001), eq. 4.13."""
    T = np.asarray(T, dtype=float)
    m = lam * T
    if abs(xi) < 1e-8:
        return u + beta * np.log(m)
    return u + (beta / xi) * (m ** xi - 1.0)


def upper_bound(u, xi, beta):
    """Finite upper end point of the fitted GPD when xi < 0, else inf."""
    return u - beta / xi if xi < 0 else np.inf


def bootstrap_return_levels(exceedances, u, lam, T, n_boot=500, seed=0):
    """Nonparametric bootstrap of the storm peaks. Resamples the exceedances
    with replacement, refits the GPD, and recomputes the return levels.
    Returns dict with 2.5 and 97.5 percentile bounds for each T and for the
    shape parameter."""
    rng = np.random.default_rng(seed)
    y = np.asarray(exceedances, dtype=float)
    T = np.atleast_1d(np.asarray(T, dtype=float))
    rl = np.empty((n_boot, len(T)))
    xis = np.empty(n_boot)
    for b in range(n_boot):
        yb = rng.choice(y, size=len(y), replace=True)
        xi_b, _, beta_b = genpareto.fit(yb, floc=0)
        xis[b] = xi_b
        rl[b] = return_level(u, xi_b, beta_b, lam, T)
    return {
        "rl_lo": np.percentile(rl, 2.5, axis=0),
        "rl_hi": np.percentile(rl, 97.5, axis=0),
        "xi_lo": float(np.percentile(xis, 2.5)),
        "xi_hi": float(np.percentile(xis, 97.5)),
        "n_boot": int(n_boot),
    }


def empirical_return_periods(peaks, lam):
    """Plotting positions for the storm peaks: Weibull exceedance probability
    k/(n+1) for the k-th largest peak, converted to a return period in years."""
    x = np.sort(np.asarray(peaks, dtype=float))[::-1]
    n = len(x)
    k = np.arange(1, n + 1)
    p = k / (n + 1.0)
    return x, 1.0 / (lam * p)


def pot_analysis(s, percentile=99.5, min_gap_hours=72, return_periods=(2, 5, 10, 25, 50, 100, 200),
                 n_boot=500, seed=0):
    """Full peaks-over-threshold analysis of a series.

    Steps: threshold at the given percentile of the whole record; runs
    declustering with the given minimum storm gap; GPD fit to the storm-peak
    exceedances; return levels at the requested return periods with bootstrap
    95% intervals.
    """
    s = s.dropna()
    u = float(np.percentile(s.values, percentile))
    peaks = decluster(s, u, min_gap_hours)
    n_years = record_length_years(pd.Series(s.index))
    lam = len(peaks) / n_years if n_years > 0 else np.nan
    exc = peaks.values - u
    if len(exc) < 10:
        raise ValueError(f"Only {len(exc)} storm peaks above the threshold; lower the percentile or the gap.")
    xi, beta, nll = fit_gpd(exc)
    T = np.asarray(return_periods, dtype=float)
    rl = return_level(u, xi, beta, lam, T)
    boot = bootstrap_return_levels(exc, u, lam, T, n_boot=n_boot, seed=seed)
    emp_x, emp_T = empirical_return_periods(peaks.values, lam)
    return {
        "threshold": u,
        "percentile": float(percentile),
        "min_gap_hours": float(min_gap_hours),
        "n_raw_exceedances": int((s > u).sum()),
        "n_peaks": int(len(peaks)),
        "record_years": float(n_years),
        "rate_per_year": float(lam),
        "xi": xi,
        "xi_ci": (boot["xi_lo"], boot["xi_hi"]),
        "beta": beta,
        "nll": nll,
        "upper_bound": float(upper_bound(u, xi, beta)),
        "return_periods": T,
        "return_levels": rl,
        "rl_ci": (boot["rl_lo"], boot["rl_hi"]),
        "peaks": peaks,
        "exceedances": exc,
        "empirical": (emp_x, emp_T),
    }


def return_level_table(res):
    """Return levels and 95% bootstrap intervals as a DataFrame."""
    return pd.DataFrame({
        "return_period_years": res["return_periods"].astype(int),
        "return_level": np.round(res["return_levels"], 3),
        "ci_low": np.round(res["rl_ci"][0], 3),
        "ci_high": np.round(res["rl_ci"][1], 3),
    })


def mean_residual_life(s, percentiles=np.arange(90, 99.91, 0.1)):
    """Mean excess over a range of thresholds, with a 95% interval. A region
    where the plot is linear in the threshold supports the GPD."""
    s = s.dropna()
    rows = []
    for p in percentiles:
        u = np.percentile(s.values, p)
        y = s.values[s.values > u] - u
        if len(y) < 5:
            continue
        rows.append({"percentile": p, "threshold": u, "n": len(y), "mean_excess": y.mean(),
                     "se": y.std(ddof=1) / np.sqrt(len(y))})
    return pd.DataFrame(rows)


def parameter_stability(s, percentiles=np.arange(97.0, 99.91, 0.1), min_gap_hours=72):
    """Shape and modified scale (beta - xi*u) of the declustered GPD fit over a
    range of thresholds. Both should be about constant above a valid threshold."""
    s = s.dropna()
    rows = []
    for p in percentiles:
        u = np.percentile(s.values, p)
        peaks = decluster(s, u, min_gap_hours)
        if len(peaks) < 10:
            continue
        xi, beta, _ = fit_gpd(peaks.values - u)
        rows.append({"percentile": p, "threshold": u, "n_peaks": len(peaks), "xi": xi,
                     "modified_scale": beta - xi * u})
    return pd.DataFrame(rows)


def sensitivity_grid(s, percentiles=(99.0, 99.5, 99.8), gaps_hours=(24, 48, 72), T=100, n_boot=0):
    """100-year (or other) return level over a grid of threshold percentiles
    and declustering gaps. Shows how much the headline number depends on the
    two analysis choices."""
    rows = []
    for p in percentiles:
        for g in gaps_hours:
            try:
                r = pot_analysis(s, percentile=p, min_gap_hours=g, return_periods=(T,), n_boot=max(n_boot, 1))
            except ValueError:
                continue
            rows.append({"percentile": p, "gap_hours": g, "threshold": r["threshold"], "n_peaks": r["n_peaks"],
                         "rate_per_year": r["rate_per_year"], "xi": r["xi"], "beta": r["beta"],
                         f"RL_{T}": float(r["return_levels"][0])})
    return pd.DataFrame(rows)


def qq_points(res):
    """Empirical vs fitted quantiles of the exceedances for a QQ plot."""
    y = np.sort(res["exceedances"])
    n = len(y)
    p = (np.arange(1, n + 1) - 0.5) / n
    fitted = genpareto.ppf(p, res["xi"], 0, res["beta"])
    return fitted, y
