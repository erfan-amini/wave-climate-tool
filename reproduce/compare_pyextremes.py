"""Cross-check of the tool's stationary POT-GPD fit against pyextremes.

Both fits use the same series, the same threshold, the same 72-hour runs
declustering, maximum likelihood and the GPD location fixed at the threshold.
Usage: python reproduce/compare_pyextremes.py data/<record>.csv.gz [percentile] [gap_hours]
Prints a table of storm counts, shape, scale and 10- to 100-year levels.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import analysis as A  # noqa: E402

warnings.filterwarnings("ignore")


def compare(series, percentile=99.5, gap_hours=72, periods=(10, 25, 50, 100)):
    from pyextremes import EVA
    ours = A.pot_analysis(series, percentile, gap_hours, return_periods=periods, n_boot=0)
    u = ours["threshold"]
    model = EVA(series)
    model.get_extremes(method="POT", extremes_type="high", threshold=u, r=f"{gap_hours}h")
    model.fit_model(model="MLE", distribution="genpareto", distribution_kwargs={"floc": u})
    p = model.distribution.mle_parameters
    rl = model.get_return_value(return_period=list(periods), return_period_size="365.2425D", alpha=None)[0]
    rl = np.atleast_1d(rl)
    rows = {"storm peaks": (ours["n_peaks"], len(model.extremes)),
            "shape xi": (ours["xi"], p["c"]), "scale beta (m)": (ours["beta"], p["scale"])}
    for T, a, b in zip(periods, ours["return_levels"], rl):
        rows[f"{T}-yr level (m)"] = (float(a), float(b))
    return pd.DataFrame(rows, index=["this tool", "pyextremes"]).T, u


if __name__ == "__main__":
    path = sys.argv[1]
    pct = float(sys.argv[2]) if len(sys.argv) > 2 else 99.5
    gap = float(sys.argv[3]) if len(sys.argv) > 3 else 72
    df, _ = A.load_series(path)
    tab, u = compare(A.to_series(df, "swh"), pct, gap)
    tab["difference"] = tab["this tool"] - tab["pyextremes"]
    print(f"threshold {u:.3f} m ({pct}th percentile), gap {gap:.0f} h")
    print(tab.round(4).to_string())
