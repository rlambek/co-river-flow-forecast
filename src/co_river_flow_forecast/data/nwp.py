"""Numerical Weather Prediction (NWP) forecast retrieval via Herbie.

Currently supports GFS (0.25 deg, 16-day horizon). HRRR and GEFS hook in
the same way; add when needed.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Iterable

import pandas as pd

HERBIE_SAVE_DIR = os.path.join("data", "cache", "herbie")

# Map of output column -> herbie GRIB search string. Search strings are
# regexes matched against the GRIB index lines.
#
# GFS APCP is reported as an accumulation over some interval ending at fxx:
# the interval is "0-N hour acc fcst" at the start of each 6-hour cycle and
# "(N-6)-N hour acc fcst" between resets. We match either form and let
# downstream code sum / interpret based on `lead_hour` spacing.
GFS_SEARCH = {
    "t2m_c": ":TMP:2 m above ground:",
    "apcp_mm_interval": r":APCP:surface:\d+-\d+ hour acc fcst:",
}


def fetch_gfs_forecast_at_point(
    lat: float,
    lon: float,
    init_time: datetime | str | None = None,
    lead_hours: Iterable[int] = (0, 24, 48, 72, 96, 120, 144, 168),
) -> pd.DataFrame:
    """Pull GFS surface temperature and accumulated precipitation at a point.

    Returns a DataFrame with columns:
        init_time, valid_time, lead_hour, t2m_c, apcp_mm_interval
    where `apcp_mm_interval` is total precipitation accumulated from init to valid time.
    """
    from herbie import Herbie  # heavy import, deferred

    init_dt = _resolve_init_time(init_time)
    os.makedirs(HERBIE_SAVE_DIR, exist_ok=True)

    rows: list[dict] = []
    for fxx in lead_hours:
        fxx = int(fxx)
        record: dict = {
            "init_time": init_dt,
            "lead_hour": fxx,
            "valid_time": init_dt + timedelta(hours=fxx),
            "t2m_c": None,
            "apcp_mm_interval": None,
        }
        for col, search in GFS_SEARCH.items():
            if col == "apcp_mm_interval" and fxx == 0:
                record[col] = 0.0  # no accumulation at f000
                continue
            try:
                H = Herbie(
                    init_dt.strftime("%Y-%m-%d %H:%M"),
                    model="gfs",
                    fxx=fxx,
                    save_dir=HERBIE_SAVE_DIR,
                    verbose=False,
                )
                ds = H.xarray(search, remove_grib=False)
            except Exception as exc:
                print(f"  [warn] {col} fxx={fxx}: {exc}")
                continue
            record[col] = _extract_at_point(ds, lat, lon, col)
        rows.append(record)

    df = pd.DataFrame(rows)
    if "t2m_c" in df.columns:
        # Herbie/cfgrib returns 2-m temperature in kelvin
        df["t2m_c"] = pd.to_numeric(df["t2m_c"], errors="coerce") - 273.15
    if "apcp_mm_interval" in df.columns:
        df["apcp_mm_interval"] = pd.to_numeric(df["apcp_mm_interval"], errors="coerce")
    return df


def _resolve_init_time(init_time: datetime | str | None) -> datetime:
    """Pick a recent GFS init time that is likely already published."""
    if init_time is None:
        # Step back ~6h from now and round to nearest 6-hourly cycle (00/06/12/18 UTC).
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        candidate = now - timedelta(hours=6)
        candidate = candidate.replace(
            hour=(candidate.hour // 6) * 6, minute=0, second=0, microsecond=0
        )
        return candidate
    if isinstance(init_time, str):
        return pd.Timestamp(init_time).to_pydatetime()
    return init_time


def _extract_at_point(ds, lat: float, lon: float, col: str) -> float | None:
    """Nearest-neighbor sample from an xarray Dataset returned by Herbie.xarray()."""
    # GFS is on a 0..360 longitude grid.
    norm_lon = lon + 360.0 if lon < 0 else lon
    if not hasattr(ds, "data_vars"):  # could be a list of datasets on some configurations
        return None
    data_var = _pick_data_var(ds, col)
    if data_var is None:
        return None
    da = ds[data_var]
    try:
        sample = da.sel(latitude=lat, longitude=norm_lon, method="nearest")
    except Exception:
        return None
    val = sample.values
    if val.size == 0:
        return None
    return float(val.flatten()[0])


def _pick_data_var(ds, col: str) -> str | None:
    """Choose the variable name in the cfgrib-decoded dataset that matches our column."""
    preferred = {
        "t2m_c": ("t2m", "2t", "TMP"),
        "apcp_mm_interval": ("tp", "APCP", "unknown"),  # cfgrib often names total precip "tp"
    }.get(col, ())
    for name in preferred:
        if name in ds.data_vars:
            return name
    # Fall back to first data variable that is not a coordinate-ish name.
    return next(iter(ds.data_vars), None)
