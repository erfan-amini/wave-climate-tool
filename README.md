# Wave Climate Trend-Extreme Tool

An open Streamlit application for wave climate projections: bias correction against a reference record, trend analysis of the annual means, stationary and non-stationary extreme-value analysis, and the directions of the storms, in one guided workflow. It accompanies the manuscript *A Unified Trend-Extreme Analysis Tool for Wave Climate Projection under Climate Change* (Shahbazbegian and Amini).

Live app: https://wave-climate-tool.streamlit.app/ · Source: https://github.com/erfan-amini/wave-climate-tool · Archived release: see the DOI badge on the repository page (Zenodo)

## Workflow

Three screens: **Setup**, **Results**, **Team**. The landing page has a one-click demo that loads the built-in records, runs the default analysis (with quantile delta mapping against ERA5) and opens the report.

| Screen | What it does |
|---|---|
| **Setup** | *Data*: the built-in catalogue, CSV uploads, NOAA NDBC wave buoys (no key), or the Copernicus Climate Data Store with the user's own key (ERA5 at a point; any station of the C3S coastal wave product). Each record has an editable role: observed, reference, model historical, model future, projected. *Options*: the records to analyse, bias correction (off, delta change, quantile delta mapping, or all three compared), and stationary or stationary plus non-stationary extremes. Threshold, storm gap, bootstrap size, return periods, non-stationary window, correction type, validation split, diagnostics and the extremes variable sit behind an **Advanced options** switch. One **Run analysis** button runs everything; identical runs are served from a cache. |
| **Results** | A report. First the 10, 25, 50 and 100-year levels with 95% intervals for each record and a one-sentence summary. Then numbered sections, each with an explanation on the left built from the numbers and the figure on the right: design levels at a glance, return levels, change through time (transformed-stationary EVA), where the storms come from (when a direction column is present), trends in the annual means, bias correction and its validation, correction methods compared, checks on the fit (storm peaks, QQ plot, threshold diagnostics, sensitivity grid), and the data used. One HTML report and one ZIP with every table, series and figure. |

Methods: peaks over threshold with runs declustering, GPD by maximum likelihood with the location fixed at the threshold, return levels from the storm rate with bootstrap intervals (Coles 2001); the fit is refused when fewer than ten storm peaks remain; transformed-stationary EVA (Mentaschi et al. 2016); delta change and quantile delta mapping by calendar month (Cannon et al. 2015) with a split-sample test on held-out years; OLS with its 95% interval, Mann-Kendall and Sen's slope for the annual means; direction of each storm peak and the share of storms by 22.5-degree sector.

## Install and run

With pip (the versions the online app uses):

```bash
pip install -r requirements.txt
streamlit run app.py
```

With conda, the exact environment used for the paper:

```bash
conda env create -f environment.yml
conda activate wave-climate-tool
streamlit run app.py
```

## Tests

```bash
python -m pytest -q tests
```

The tests check the statistics against known answers: the return-level formula and its limits, recovery of GPD parameters from simulated data, declustering, the ten-peak refusal rule, the trend tests, QDM's preservation of the model change, the storm-direction summary and the loader. They run on every push (GitHub Actions).

## Reproduce the paper

```bash
python reproduce/make_figures.py .            # every figure, table and number in the manuscript
python reproduce/compare_pyextremes.py data/rcp85_2041_2100.csv.gz   # one fit repeated with pyextremes
```

`make_figures.py` writes `images/`, `tables/rows.tex` and `numbers.tex` next to itself; the manuscript reads its numbers from `numbers.tex`. It takes about four minutes with the paper's 2000 bootstrap resamples (set `NBOOT=200` for a quick check).

The interface figure needs screenshots of the running app. With the app running (`streamlit run app.py`) and Playwright installed:

```bash
python reproduce/shot_paper.py http://localhost:8501/   # writes reproduce/app_*.png; rerun make_figures.py to assemble them
```

## Files

```
app.py            Streamlit interface: landing page, Setup, Results report, Team
pipeline.py       one-call analysis run used by the app (bias correction, trends, extremes, checks, directions)
analysis.py       loading, climatology, trends, POT-GPD, diagnostics, storm directions
projection.py     delta change, quantile delta mapping, split-sample validation, transformed-stationary EVA
fetch.py          NDBC, Copernicus CDS (ERA5 point, C3S station), catalogue; also a command-line exporter
charts.py         Plotly figures
data/             demonstration records, station.json and catalogue.json
tests/            automated tests
reproduce/        the paper's figure script and the pyextremes cross-check
environment.yml   pinned environment; requirements.txt for the online app
```

`analysis.py`, `projection.py` and `pipeline.py` have no interface dependency.

## Built-in data

| File | Content |
|---|---|
| `data/reference_era5_1976_2005.csv.gz` | ERA5 reanalysis at 38.0°N, 12.0°E, 1976 to 2005, hourly `swh` and `dir` (mean wave direction) |
| `data/historical_1976_2005.csv.gz` | C3S station 3851 (38.0°N, 12.3°E), historical run, 1976 to 2005, hourly `swh`, `tm`, `dir` where available |
| `data/rcp45_2041_2100.csv.gz` | C3S station 3851, RCP4.5, 2041 to 2100 |
| `data/rcp85_2041_2100.csv.gz` | C3S station 3851, RCP8.5, 2041 to 2100 |
| `data/brisbane_2024.csv.gz` | 30-minute wave-buoy export, 2024, near Brisbane, Australia (as exported) |
| `data/station.json` | station and ERA5 point coordinates |

The C3S station is on the 20 m depth contour off western Sicily, next to the Egadi Islands, from the Copernicus Climate Change Service product *Ocean surface wave time series for the European coast from 1976 to 2100 derived from climate projections* (dataset `sis-ocean-wave-timeseries`). The reference is ERA5 (Hersbach et al. 2020) from the point time-series product `reanalysis-era5-single-levels-timeseries`. Generated using Copernicus Climate Change Service information 2026. Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains.

To add a station, place its files in `data/` (or at a URL) and add an entry to `data/catalogue.json`.

## Exporting records from the command line

```bash
python fetch.py ndbc --station 44025 --years 2015-2024 --out data/ndbc_44025.csv.gz
python fetch.py era5 --lat 38.0 --lon 12.0 --years 1976-2005 --out data/reference_era5_1976_2005.csv.gz
python fetch.py c3s  --lat 38.0 --lon 12.3 --experiment historical --years 1976-2005 --out data/historical_1976_2005.csv.gz
```

The CDS key can be passed with `--key` or kept in `~/.cdsapirc`; accept each dataset's licence once on the CDS website. The C3S product comes as one file per variable and year for the whole European coast (about 0.5 GB each), so a station's 30 or 60 years take hours; `fetch.py` keeps the one station and deletes each file after use.

## Input format

A CSV (optionally gzip) with a time column and a wave height and/or period column, and optionally a mean wave direction. Column names are matched case-insensitively against common variants (`swh`, `Hs`, `Hs (m)`, `VHM0`; `tm`, `mp1`, `Tz`, `VTM02`; `tp`, `Tp (s)`; `dir`, `mwd`, `Dir (deg)`, `VMDR`). Missing-value codes such as `-99.9` become NaN; duplicate timestamps are dropped.

## License

MIT. See `LICENSE`.
