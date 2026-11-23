"""
Data access for the Wave Climate Trend-Extreme Tool.

Three sources, all written to the tool's CSV format (time, swh[, tm, dir]):

1. NOAA NDBC wave buoys (no key): historical standard meteorological files by
   station and year.
2. Copernicus Climate Data Store (user's own API key):
   - ERA5 hourly significant wave height and mean wave direction at a point,
     from the point time-series product (one request; about a minute).
   - C3S "Ocean surface wave time series for the European coast" (dataset
     sis-ocean-wave-timeseries): the station nearest to a point, for one
     experiment and a range of years. The product is served as one file per
     variable and year for the whole domain (about 0.5 GB each), so long
     periods take hours; the function keeps one station and deletes each file.
3. A catalogue of ready-made records (data/catalogue.json) that can point at
   files in the repository or at URLs.

Command line (for one-off exports):
    python fetch.py ndbc --station 44025 --years 2015-2020 --out data/ndbc_44025.csv.gz
    python fetch.py era5 --lat 38.0 --lon 12.0 --years 1976-2005 --out data/reference_era5_1976_2005.csv.gz
    python fetch.py c3s  --lat 38.0 --lon 12.3 --experiment historical --years 1976-2005 --out data/historical_1976_2005.csv.gz

The CDS commands need a key: pass --key "<key>" or put it in ~/.cdsapirc, and
accept each dataset's licence once on the CDS website.
"""

import argparse
import gzip
import io
import json
import os
import tempfile
import zipfile

import numpy as np
import pandas as pd

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CDS_URL = "https://cds.climate.copernicus.eu/api"
NDBC_URL = "https://www.ndbc.noaa.gov/data/historical/stdmet/{station}h{year}.txt.gz"
NDBC_MISSING = {"WVHT": (99.0, 99.00, 999.0), "DPD": (99.0, 999.0), "APD": (99.0, 999.0)}

# ---------------------------------------------------------------------------
# Output format
# ---------------------------------------------------------------------------

def write_record(df, path):
    """Write a record (time, swh, tm[, tp]) as gzip CSV in the tool's format."""
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"]).dt.strftime("%Y-%m-%d %H:%M")
    cols = ["time"] + [c for c in ("swh", "tm", "tp") if c in out.columns]
    out[cols].round(4).to_csv(path, index=False, compression="gzip" if str(path).endswith(".gz") else None)
    return path


def years_from_spec(spec):
    """'1979-2005' -> [1979, ..., 2005]; '2015,2017' -> [2015, 2017]."""
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


# ---------------------------------------------------------------------------
# 1. NOAA NDBC
# ---------------------------------------------------------------------------

def parse_ndbc_stdmet(text):
    """Parse an NDBC standard meteorological file (any vintage) into a
    DataFrame with time, swh (WVHT), tp (DPD) and tm (APD)."""
    lines = [l for l in text.splitlines() if l.strip()]
    header = lines[0].lstrip("#").split()
    body_start = 2 if lines[1].lstrip().startswith("#") else 1
    rows = [l.split() for l in lines[body_start:] if not l.startswith("#")]
    df = pd.DataFrame(rows, columns=header[:len(rows[0])] if rows else header)
    df = df.apply(pd.to_numeric, errors="coerce")
    ycol = next(c for c in df.columns if c.upper() in ("YY", "YYYY", "#YY"))
    year = df[ycol].astype(int)
    if year.max() < 100:
        year = year + 1900
    minute = df["mm"] if "mm" in df.columns else 0
    time = pd.to_datetime(dict(year=year, month=df["MM"], day=df["DD"], hour=df["hh"], minute=minute), errors="coerce")
    out = pd.DataFrame({"time": time})
    for col, name in (("WVHT", "swh"), ("DPD", "tp"), ("APD", "tm")):
        if col in df.columns:
            v = df[col].astype(float)
            v[v.isin(NDBC_MISSING[col]) | (v >= 99.0)] = np.nan
            out[name] = v.values
    out = out.dropna(subset=["time"]).sort_values("time").drop_duplicates("time")
    return out.reset_index(drop=True)


def fetch_ndbc(station, years, session=None):
    """Download and parse NDBC historical files for a station over years.
    Years with no file are skipped and reported. Returns (DataFrame, notes)."""
    import requests
    s = session or requests.Session()
    frames, notes = [], []
    for y in years:
        url = NDBC_URL.format(station=str(station).lower(), year=int(y))
        r = s.get(url, timeout=60)
        if r.status_code != 200:
            notes.append(f"{y}: no file ({r.status_code})")
            continue
        text = gzip.decompress(r.content).decode("utf-8", errors="replace")
        frames.append(parse_ndbc_stdmet(text))
        notes.append(f"{y}: {len(frames[-1]):,} rows")
    if not frames:
        raise ValueError("No NDBC data found for that station and those years. " + "; ".join(notes))
    df = pd.concat(frames, ignore_index=True).sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return df, notes


# ---------------------------------------------------------------------------
# 2. Copernicus Climate Data Store
# ---------------------------------------------------------------------------

def cds_client(key=None):
    """A cdsapi client from an explicit key ('<uid>:<key>' or the new
    personal access token) or from ~/.cdsapirc."""
    import cdsapi
    if key:
        return cdsapi.Client(url=CDS_URL, key=key, quiet=True)
    return cdsapi.Client(quiet=True)


def _nearest_sea_point(ds, var, lat, lon):
    """Index of the nearest grid point of `ds` with valid data for `var`."""
    latn = "latitude" if "latitude" in ds.coords else "lat"
    lonn = "longitude" if "longitude" in ds.coords else "lon"
    valid = ds[var].isel({[d for d in ds[var].dims if d not in (latn, lonn)][0]: 0}).notnull()
    la, lo = np.meshgrid(ds[latn].values, ds[lonn].values, indexing="ij")
    dist = np.hypot(la - lat, (lo - lon) * np.cos(np.radians(lat)))
    dist = np.where(valid.values, dist, np.inf)
    i, j = np.unravel_index(np.argmin(dist), dist.shape)
    return {latn: ds[latn].values[i], lonn: ds[lonn].values[j]}


def fetch_era5_point(lat, lon, years, key=None, progress=None):
    """ERA5 hourly significant wave height and mean wave direction at the
    nearest grid point, from the CDS point time-series product (one request,
    typically about a minute for thirty years). Returns (DataFrame, point).
    The ERA5 mean wave period is defined from a different spectral moment than
    most model periods, so it is not returned as 'tm'."""
    c = cds_client(key)
    years = list(years)
    with tempfile.TemporaryDirectory() as tmpdir:
        target = os.path.join(tmpdir, "era5.zip")
        if progress:
            progress(0.1, "request submitted to CDS")
        c.retrieve("reanalysis-era5-single-levels-timeseries", {
            "variable": ["significant_height_of_combined_wind_waves_and_swell", "mean_wave_direction"],
            "location": {"latitude": float(lat), "longitude": float(lon)},
            "date": [f"{years[0]}-01-01/{years[-1]}-12-31"],
            "data_format": "csv",
        }, target)
        if progress:
            progress(0.8, "download finished; reading")
        if zipfile.is_zipfile(target):
            with zipfile.ZipFile(target) as zf:
                name = [n for n in zf.namelist() if n.endswith(".csv")][0]
                zf.extract(name, tmpdir)
                raw = pd.read_csv(os.path.join(tmpdir, name))
        else:
            raw = pd.read_csv(target)
    df = pd.DataFrame({"time": pd.to_datetime(raw["valid_time"]), "swh": raw["swh"], "dir": raw["mwd"]})
    point = (float(raw["latitude"].iloc[0]), float(raw["longitude"].iloc[0]))
    if df["swh"].isna().all():
        raise ValueError(f"ERA5 has no wave data at {point}; choose a point further offshore")
    return df.sort_values("time").reset_index(drop=True), point


# The C3S coastal wave time series: one file per variable and year for the
# whole domain (7,708 stations, about 0.5 GB per variable and year).
C3S_DATASET = "sis-ocean-wave-timeseries"
C3S_REQUEST = {
    "variable": {"swh": "Significant wave height", "tm": "Mean wave period", "dir": "Mean wave direction"},
    "experiment": {"historical": "Historical", "reanalysis": "ERA5 reanalysis", "rcp45": "RCP4.5", "rcp85": "RCP8.5"},
    "years": {"historical": (1976, 2005), "reanalysis": (2001, 2017), "rcp45": (2041, 2100), "rcp85": (2041, 2100)},
}


def _nearest_station(ds, lat, lon):
    """Index of the nearest station in a C3S coastal time-series file."""
    x = ds["station_x_coordinate"].values
    y = ds["station_y_coordinate"].values
    x = np.where(x > 180, x - 360, x)
    dist = np.hypot(y - lat, (x - lon) * np.cos(np.radians(lat)))
    i = int(np.argmin(dist))
    return i, float(x[i]), float(y[i])


def _station_series(nc_path, lat, lon):
    import xarray as xr
    ds = xr.open_dataset(nc_path)
    i, sx, sy = _nearest_station(ds, lat, lon)
    sdim = ds["station_x_coordinate"].dims[0]
    v = [k for k in ds.data_vars if k != "missing_value" and sdim in ds[k].dims][0]
    s = pd.Series(ds[v].isel({sdim: i}).values, index=pd.to_datetime(ds["time"].values))
    ds.close()
    return s, (sx, sy)


def fetch_c3s_station(lat, lon, experiment="historical", years=None, variables=("swh", "tm", "dir"), key=None, progress=None):
    """Download the C3S coastal wave product year by year, keep the station
    nearest to (lat, lon) and delete each whole-domain file after use.
    Returns (DataFrame with time and the requested variables, (lon, lat))."""
    c = cds_client(key)
    y0, y1 = C3S_REQUEST["years"][experiment]
    years = [y for y in (years or range(y0, y1 + 1)) if y0 <= y <= y1]
    if not years:
        raise ValueError(f"{experiment} covers {y0} to {y1}")
    pieces = {v: [] for v in variables}
    where = None
    jobs = [(y, v) for y in years for v in variables]
    for k, (y, v) in enumerate(jobs):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = os.path.join(tmpdir, "c3s.zip")
            c.retrieve(C3S_DATASET, {"variable": [C3S_REQUEST["variable"][v]], "experiment": C3S_REQUEST["experiment"][experiment],
                                     "year": [str(y)]}, target)
            with zipfile.ZipFile(target) as zf:
                name = [n for n in zf.namelist() if n.endswith(".nc")][0]
                zf.extract(name, tmpdir)
            os.remove(target)
            s, where = _station_series(os.path.join(tmpdir, name), lat, lon)
            pieces[v].append(s)
        if progress:
            progress((k + 1) / len(jobs), f"{experiment} {y} {v} done")
    out = pd.DataFrame({v: pd.concat(p).sort_index() for v, p in pieces.items() if p})
    out = out[~out.index.duplicated(keep="first")]
    out.index.name = "time"
    return out.reset_index(), where


# ---------------------------------------------------------------------------
# 3. Catalogue
# ---------------------------------------------------------------------------

def load_catalogue(path=None):
    path = path or os.path.join(APP_DIR, "data", "catalogue.json")
    with open(path) as f:
        return json.load(f)


def resolve_catalogue_file(entry_file):
    """A catalogue file is a path relative to the app folder or a URL.
    URLs are downloaded to a local cache once."""
    if entry_file.startswith("http://") or entry_file.startswith("https://"):
        import requests
        cache = os.path.join(tempfile.gettempdir(), "wave_climate_cache")
        os.makedirs(cache, exist_ok=True)
        local = os.path.join(cache, os.path.basename(entry_file))
        if not os.path.exists(local):
            r = requests.get(entry_file, timeout=120)
            r.raise_for_status()
            with open(local, "wb") as f:
                f.write(r.content)
        return local
    return os.path.join(APP_DIR, entry_file)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ndbc", help="NOAA NDBC buoy, historical files")
    a.add_argument("--station", required=True); a.add_argument("--years", required=True); a.add_argument("--out", required=True)
    b = sub.add_parser("era5", help="ERA5 wave height and direction at a point (CDS key needed)")
    b.add_argument("--lat", type=float, required=True); b.add_argument("--lon", type=float, required=True)
    b.add_argument("--years", required=True); b.add_argument("--out", required=True); b.add_argument("--key")
    c = sub.add_parser("c3s", help="C3S coastal wave time series, nearest station (CDS key needed)")
    c.add_argument("--lat", type=float, required=True); c.add_argument("--lon", type=float, required=True)
    c.add_argument("--experiment", choices=list(C3S_REQUEST["experiment"]), default="historical")
    c.add_argument("--years", help="e.g. 1976-2005; default: the whole experiment"); c.add_argument("--out", required=True); c.add_argument("--key")
    args = p.parse_args()
    if args.cmd == "ndbc":
        df, notes = fetch_ndbc(args.station, years_from_spec(args.years))
        print("\n".join(notes))
    elif args.cmd == "era5":
        df, point = fetch_era5_point(args.lat, args.lon, years_from_spec(args.years), key=args.key,
                                     progress=lambda f, m: print(f"{f:5.0%} {m}"))
        print("nearest ERA5 sea point:", point)
    else:
        df, where = fetch_c3s_station(args.lat, args.lon, args.experiment, years_from_spec(args.years) if args.years else None, key=args.key,
                                      progress=lambda f, m: print(f"{f:5.0%} {m}"))
        print("nearest C3S station (lon, lat):", where)
    write_record(df, args.out)
    print(f"wrote {args.out}: {len(df):,} rows, {df['time'].iloc[0]} to {df['time'].iloc[-1]}")


if __name__ == "__main__":
    main()
