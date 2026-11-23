"""Tests of the statistics behind the Wave Climate Trend-Extreme Tool.

Run with:  python -m pytest -q
Each test checks a property with a known answer: a synthetic record built from
a known distribution, or a closed-form result.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.stats import genpareto

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import analysis as A  # noqa: E402
import projection as P  # noqa: E402


def hourly(values, start="2000-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="h"))


def test_return_level_formula_and_limit():
    u, beta, lam, T = 3.0, 1.0, 3.0, 100.0
    # xi -> 0 gives the exponential (Gumbel-type) limit
    assert A.return_level(u, 1e-9, beta, lam, T) == pytest.approx(u + beta * np.log(lam * T), rel=1e-6)
    # negative shape is bounded by u - beta/xi
    assert A.return_level(u, -0.2, beta, lam, 1e12) < u - beta / -0.2


def test_gpd_fit_recovers_parameters():
    rng = np.random.default_rng(1)
    y = genpareto.rvs(-0.1, 0, 1.2, size=20000, random_state=rng)
    xi, beta = A.fit_gpd(y)[:2]
    assert xi == pytest.approx(-0.1, abs=0.02)
    assert beta == pytest.approx(1.2, rel=0.03)


def test_decluster_separates_storms_by_gap():
    s = hourly(np.zeros(500))
    s.iloc[[10, 11, 12, 100, 101, 300]] = [5, 6, 5, 7, 6, 8]
    peaks = A.decluster(s, threshold=4.0, min_gap_hours=48)
    assert list(peaks.values) == [6, 7, 8]


def test_pot_refuses_fewer_than_ten_peaks():
    s = hourly(np.r_[np.zeros(2000), 5 + np.arange(5)])
    with pytest.raises(ValueError):
        A.pot_analysis(s, percentile=99.5, min_gap_hours=1, n_boot=0)


def test_mann_kendall_detects_trend_and_not_noise():
    rng = np.random.default_rng(2)
    assert A.mann_kendall(np.arange(40) + rng.normal(0, 3, 40))["p_value"] < 0.01
    assert A.mann_kendall(rng.normal(0, 1, 40))["p_value"] > 0.01


def test_sen_slope_of_a_line():
    years = np.arange(2000, 2030)
    assert A.sens_slope(years, 0.05 * years + 1)["slope"] == pytest.approx(0.05)


def test_qdm_maps_reference_and_keeps_model_change():
    rng = np.random.default_rng(3)
    idx_h = pd.date_range("1980-01-01", "2009-12-31 23:00", freq="h")
    idx_f = pd.date_range("2050-01-01", "2079-12-31 23:00", freq="h")
    hist = pd.Series(rng.gamma(2.0, 0.5, len(idx_h)), index=idx_h)
    ref = 1.2 * pd.Series(rng.gamma(2.0, 0.5, len(idx_h)), index=idx_h)      # reference = model x 1.2
    fut = 1.1 * pd.Series(rng.gamma(2.0, 0.5, len(idx_f)), index=idx_f)      # model change = x 1.1
    q = P.quantile_delta_mapping(ref, hist, fut, kind="multiplicative")
    for p in (0.5, 0.9, 0.99):
        assert np.quantile(q, p) / np.quantile(ref, p) == pytest.approx(1.1, rel=0.03)


def test_storm_directions_dominant_sector():
    rng = np.random.default_rng(4)
    n = 24 * 365 * 12
    hs = hourly(rng.gamma(2, 0.6, n))
    d = hourly((300 + rng.normal(0, 15, n)) % 360)
    res = A.pot_analysis(hs, 99.5, 72, n_boot=0)
    out = A.storm_directions(d, res)
    assert out["dominant"] in ("WNW", "W", "NW")
    assert out["storm_share"].sum() == pytest.approx(100.0)


def test_loader_reads_direction_and_sentinels(tmp_path):
    f = tmp_path / "rec.csv"
    pd.DataFrame({"Date/Time (AEST)": pd.date_range("2024-01-01", periods=4, freq="30min"),
                  "Hs (m)": [1.0, -99.9, 1.2, 1.1], "Dir (deg)": [90, 95, 100, 110]}).to_csv(f, index=False)
    df, info = A.load_series(str(f))
    assert set(info["columns"]) == {"swh", "dir"}
    assert df["swh"].isna().sum() == 1
