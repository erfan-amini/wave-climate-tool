# Wave Climate Trend-Extreme Tool

An open Streamlit application for wave climate projections: bias correction against a reference record, trend analysis of the annual means, and stationary and non-stationary extreme-value analysis, in one guided workflow. It accompanies the manuscript *A Unified Trend-Extreme Analysis Tool for Wave Climate Projection under Climate Change* (Shahbazbegian and Amini).

Live app: https://wave-climate-tool.streamlit.app/ · Source: https://github.com/erfan-amini/wave-climate-tool

## Workflow

Three screens: **Setup**, **Results**, **Team**. The landing page has a one-click demo that loads the built-in records, runs the default analysis and opens the report.

| Screen | What it does |
|---|---|
| **Setup** | *Data*: the built-in catalogue, CSV uploads, NOAA NDBC wave buoys (no key), or the Copernicus Climate Data Store with the user's own key (ERA5 at a point; any station of the C3S coastal wave product). Each record has an editable role: observed, reference, model historical, model future, projected. *Options*: the records to analyse, bias correction (off, delta change, quantile delta mapping, or all three compared), and stationary or stationary plus non-stationary extremes. Threshold, storm gap, bootstrap size, return periods, non-stationary window, correction type, validation split, diagnostics and the extremes variable sit behind an **Advanced options** switch. One **Run analysis** button runs everything; identical runs are served from a cache. |
| **Results** | A report. First the 10, 25, 50 and 100-year levels with 95% intervals for each record and a one-sentence summary. Then numbered sections, each with an explanation on the left built from the numbers and the figure on the right: design levels at a glance, return levels, change through time (transformed-stationary EVA), trends in the annual means, bias correction and its validation, correction methods compared, checks on the fit (storm peaks, QQ plot, threshold diagnostics, sensitivity grid), and the data used. One HTML report and one ZIP with every table, series and figure. |

Methods: peaks over threshold with runs declustering, GPD by maximum likelihood with the location fixed at the threshold, return levels from the storm rate with bootstrap intervals (Coles 2001); transformed-stationary EVA (Mentaschi et al. 2016); delta change and quantile delta mapping by calendar month (Cannon et al. 2015) with a split-sample test on held-out years; OLS with its 95% interval, Mann-Kendall and Sen's slope for the annual means.

## Files

```
app.py            Streamlit interface: landing page, Setup, Results report, Team
pipeline.py       one-call analysis run used by the app (bias correction, trends, extremes, checks)
analysis.py       loading, climatology, trends, POT-GPD, diagnostics
projection.py     delta change, quantile delta mapping, split-sample validation, transformed-stationary EVA
fetch.py          NDBC, Copernicus CDS (ERA5 point, C3S station), catalogue; also a command-line exporter
charts.py         Plotly figures
data/             demonstration records and catalogue.json
.streamlit/       theme
```

`analysis.py`, `projection.py` and `pipeline.py` have no interface dependency. The manuscript's figure script imports them.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Built-in data

| File | Content |
|---|---|
| `data/historical_2005.csv.gz` | C3S station, historical run, 2005, hourly `swh` and `mp1` |
| `data/rcp45_2041_2100.csv.gz` | C3S station, RCP4.5, 2041 to 2100 |
| `data/rcp85_2041_2100.csv.gz` | C3S station, RCP8.5, 2041 to 2100 |
| `data/brisbane_2024.csv.gz` | 30-minute wave-buoy export, 2024, near Brisbane, Australia (as exported) |

The C3S station is at 65.6°N, 37.9°W (southeast Greenland, 20 m depth contour) from the Copernicus Climate Change Service product *Ocean surface wave time series for the European coast from 1976 to 2100 derived from climate projections*. Generated using Copernicus Climate Change Service information 2020. Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains.

To add a station, place its files in `data/` (or at a URL) and add an entry to `data/catalogue.json`.

## Exporting records from the command line

```bash
python fetch.py ndbc --station 44025 --years 2015-2024 --out data/ndbc_44025.csv.gz
python fetch.py era5 --lat 40.5 --lon 2.0 --years 1979-2005 --out data/reference_era5_1979_2005.csv.gz --key "<your CDS key>"
python fetch.py c3s  --lat 40.5 --lon 2.0 --experiment historical --out data/historical_1976_2005.csv.gz --key "<your CDS key>"
```

The CDS key can also sit in `~/.cdsapirc`. The request keys for the C3S dataset (`C3S_REQUEST` in `fetch.py`) should be checked once against the "Show API request" button on the dataset's CDS page.

## Input format

A CSV (optionally gzip) with a time column and a wave height and/or period column. Column names are matched case-insensitively against common variants (`swh`, `Hs`, `Hs (m)`, `VHM0`; `tm`, `mp1`, `Tz`, `VTM02`; `tp`, `Tp (s)`). Missing-value codes such as `-99.9` become NaN; duplicate timestamps are dropped.

## License

MIT. See `LICENSE`.
