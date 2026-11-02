"""
Data access for the Wave Climate Trend-Extreme Tool.

Three sources, all written to the tool's CSV format (time, swh, tm[, tp]):

1. NOAA NDBC wave buoys (no key): historical standard meteorological files by
   station and year.
2. Copernicus Climate Data Store (user's own API key):
   - ERA5 hourly single levels at a point (significant wave height and mean
     wave period), year by year.
   - C3S "Ocean surface wave time series for the European coast" (dataset
     sis-ocean-wave-timeseries), nearest station to a point, for one
     experiment and period.
3. A catalogue of ready-made records (data/catalogue.json) that can point at
   files in the repository or at URLs.

Command line (for one-off exports):
    python fetch.py ndbc --station 44025 --years 2015-2020 --out data/ndbc_44025.csv.gz
    python fetch.py era5 --lat 40.5 --lon 2.0 --years 1979-2005 --out data/reference_era5_1979_2005.csv.gz
    python fetch.py c3s  --lat 40.5 --lon 2.0 --experiment historical --period 1976_2005 --out data/historical_1976_2005.csv.gz

The CDS commands need a key: pass --key "<uid>:<key>" or put it in ~/.cdsapirc.
The request keys for the C3S dataset should be checked once against the
"Show API request" button on its CDS page; they are set in C3S_REQUEST below.
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
    """ERA5 hourly significant wave height and mean wave period at the
    nearest sea point, one CDS request per year. Returns (DataFrame, point)."""
    import xarray as xr
    c = cds_client(key)
    frames, point = [], None
    for k, y in enumerate(years):
        req = {
            "product_type": ["reanalysis"],
            "variable": ["significant_height_of_combined_wind_waves_and_swell", "mean_wave_period"],
            "year": [str(y)], "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)], "time": [f"{h:02d}:00" for h in range(24)],
            "area": [lat + 0.75, lon - 0.75, lat - 0.75, lon + 0.75],
            "data_format": "netcdf", "download_format": "unarchived",
        }
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            target = tmp.name
        c.retrieve("reanalysis-era5-single-levels", req, target)
        ds = xr.open_dataset(target)
        if point is None:
            point = _nearest_sea_point(ds, "swh", lat, lon)
        sub = ds.sel(point, method="nearest")
        tname = "valid_time" if "valid_time" in sub.coords else "time"
        frames.append(pd.DataFrame({"time": pd.to_datetime(sub[tname].values), "swh": sub["swh"].values.ravel(),
                                    "tm": sub["mwp"].values.ravel()}))
        ds.close()
        os.remove(target)
        if progress:
            progress((k + 1) / len(years), f"ERA5 {y} done")
    df = pd.concat(frames, ignore_index=True).sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return df, point


# Request keys for the C3S coastal wave time series. Confirm once against
# the "Show API request" button on the dataset's CDS page.
C3S_DATASET = "sis-ocean-wave-timeseries"
C3S_REQUEST = {
    "variable": ["significant_wave_height", "mean_wave_period"],
    "experiment": {"historical": "historical", "reanalysis": "reanalysis", "rcp45": "rcp4_5", "rcp85": "rcp8_5"},
    "period": {"historical": "1976_2005", "reanalysis": "2001_2017", "rcp45": "2041_2100", "rcp85": "2041_2100"},
    "format": "zip",
}
C3S_VARNAMES = {"swh": ("swh", "VHM0", "significant_wave_height"), "tm": ("mp1", "mwp", "VTM02", "mean_wave_period")}


def _pick_var(ds, names):
    for n in names:
        if n in ds.data_vars:
            return n
    return None


def _nearest_station(ds, lat, lon):
    """Index of the nearest station in a C3S coastal time-series file."""
    x = ds["station_x_coordinate"].values
    y = ds["station_y_coordinate"].values
    x = np.where(x > 180, x - 360, x)
    dist = np.hypot(y - lat, (x - lon) * np.cos(np.radians(lat)))
    i = int(np.argmin(dist))
    return i, float(x[i]), float(y[i])


def extract_c3s_station(zip_path, lat, lon):
    """Read every NetCDF inside a C3S zip and return the nearest station's
    time series as a DataFrame (time, swh, tm) plus the station location."""
    import xarray as xr
    pieces = {"swh": [], "tm": []}
    where = None
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".nc"):
                continue
            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
                tmp.write(zf.read(name))
                target = tmp.name
            ds = xr.open_dataset(target)
            i, sx, sy = _nearest_station(ds, lat, lon)
            where = (sx, sy)
            sdim = [d for d in ds.dims if d.startswith("station")][0]
            for key, names in C3S_VARNAMES.items():
                v = _pick_var(ds, names)
                if v is not None:
                    sub = ds[v].isel({sdim: i})
                    pieces[key].append(pd.Series(sub.values, index=pd.to_datetime(ds["time"].values)))
            ds.close()
            os.remove(target)
    out = pd.DataFrame({k: pd.concat(v).sort_index() for k, v in pieces.items() if v})
    out = out[~out.index.duplicated(keep="first")]
    out.index.name = "time"
    return out.reset_index(), where


def fetch_c3s_station(lat, lon, experiment="historical", period=None, key=None, progress=None):
    """Request one experiment of the C3S coastal wave product and extract the
    nearest station. The whole-domain files are large; expect a long wait."""
    c = cds_client(key)
    req = {
        "variable": C3S_REQUEST["variable"],
        "experiment": [C3S_REQUEST["experiment"][experiment]],
        "period": [period or C3S_REQUEST["period"][experiment]],
        "format": C3S_REQUEST["format"],
    }
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        target = tmp.name
    if progress:
        progress(0.1, "request submitted to CDS; waiting in the queue")
    c.retrieve(C3S_DATASET, req, target)
    if progress:
        progress(0.7, "download finished; extracting the nearest station")
    df, where = extract_c3s_station(target, lat, lon)
    os.remove(target)
    return df, where


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
    b = sub.add_parser("era5", help="ERA5 wave height and period at a point (CDS key needed)")
    b.add_argument("--lat", type=float, required=True); b.add_argument("--lon", type=float, required=True)
    b.add_argument("--years", required=True); b.add_argument("--out", required=True); b.add_argument("--key")
    c = sub.add_parser("c3s", help="C3S coastal wave time series, nearest station (CDS key needed)")
    c.add_argument("--lat", type=float, required=True); c.add_argument("--lon", type=float, required=True)
    c.add_argument("--experiment", choices=list(C3S_REQUEST["experiment"]), default="historical")
    c.add_argument("--period"); c.add_argument("--out", required=True); c.add_argument("--key")
    args = p.parse_args()
    if args.cmd == "ndbc":
        df, notes = fetch_ndbc(args.station, years_from_spec(args.years))
        print("\n".join(notes))
    elif args.cmd == "era5":
        df, point = fetch_era5_point(args.lat, args.lon, years_from_spec(args.years), key=args.key,
                                     progress=lambda f, m: print(f"{f:5.0%} {m}"))
        print("nearest ERA5 sea point:", point)
    else:
        df, where = fetch_c3s_station(args.lat, args.lon, args.experiment, args.period, key=args.key,
                                      progress=lambda f, m: print(f"{f:5.0%} {m}"))
        print("nearest C3S station (lon, lat):", where)
    write_record(df, args.out)
    print(f"wrote {args.out}: {len(df):,} rows, {df['time'].iloc[0]} to {df['time'].iloc[-1]}")


if __name__ == "__main__":
    main()
