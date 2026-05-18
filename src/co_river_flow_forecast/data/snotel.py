"""SNOTEL daily SWE retrieval and basin-polygon-based station discovery via metloom."""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Iterable

import pandas as pd
from metloom.pointdata import SnotelPointData
from metloom.variables import SnotelVariables

from co_river_flow_forecast.basins import Basin


def discover_snotel_sites(basin: Basin) -> list[dict[str, Any]]:
    """Find SNOTEL stations whose location falls within the basin polygon.

    Pulls the upstream-of-gauge polygon from NLDI (cached), then asks metloom
    for SNOTEL stations within that polygon that report SWE. Returns a list of
    station-info dicts with keys: triplet, name, elevation_ft, lat, lon.
    """
    # Local imports keep geopandas + shapely off the module-import path for
    # callers that only need fetch_daily_swe().
    import geopandas as gpd

    from co_river_flow_forecast.data.basin_geometry import get_basin_polygon

    polygon = get_basin_polygon(basin)
    gdf = gpd.GeoDataFrame({"name": [basin.short_name]}, geometry=[polygon], crs="EPSG:4326")

    collection = SnotelPointData.points_from_geometry(
        gdf,
        [SnotelVariables.SWE],
        snow_courses=False,
        within_geometry=True,
    )

    out: list[dict[str, Any]] = []
    for pt in collection:
        info = _station_info(pt)
        out.append(info)
    # Sort by elevation descending so the highest-altitude (most snow-relevant)
    # stations come first.
    out.sort(key=lambda s: -(s.get("elevation_ft") or 0))
    return out


def _station_info(pt: SnotelPointData) -> dict[str, Any]:
    triplet = getattr(pt, "id", None) or getattr(pt, "station_id", None) or ""
    name = getattr(pt, "name", "") or ""
    elevation = None
    lat = None
    lon = None
    try:
        meta = pt.metadata  # property; may issue a network call
    except Exception:
        meta = None
    if meta is not None:
        # metloom returns a shapely Point with x=lon, y=lat, z=elevation in many cases.
        try:
            lon, lat = float(meta.x), float(meta.y)
            elevation = float(meta.z) if meta.has_z else None
        except Exception:
            pass
        # Or a dict-like metadata
        if isinstance(meta, dict):
            name = meta.get("name", name) or name
            elevation = meta.get("elevation", elevation)
            lat = meta.get("latitude", lat)
            lon = meta.get("longitude", lon)
    return {
        "triplet": triplet,
        "name": name,
        "elevation_ft": elevation,
        "lat": lat,
        "lon": lon,
    }


def fetch_daily_swe(
    triplets: Iterable[str],
    start: str | date,
    end: str | date | None = None,
) -> pd.DataFrame:
    """Daily SWE (inches) for one or more SNOTEL stations.

    Returns a long-format DataFrame with columns:
      date          : observation date (tz-naive)
      triplet       : SNOTEL triplet, e.g. "482:CO:SNTL"
      station_name  : metloom-reported station name (falls back to triplet)
      swe_in        : snow water equivalent in inches
    """
    start_dt = _as_datetime(start)
    end_dt = _as_datetime(end) if end is not None else datetime.combine(
        date.today(), datetime.min.time()
    )

    frames: list[pd.DataFrame] = []
    for triplet in triplets:
        pt = SnotelPointData(triplet, triplet)
        raw = pt.get_daily_data(start_dt, end_dt, [SnotelVariables.SWE])
        if raw is None or len(raw) == 0:
            continue
        frames.append(_normalize(raw, triplet, _station_name(pt, triplet)))

    if not frames:
        return pd.DataFrame(columns=["date", "triplet", "station_name", "swe_in"])
    return pd.concat(frames, ignore_index=True)


def _normalize(df: pd.DataFrame, triplet: str, station_name: str) -> pd.DataFrame:
    df = df.reset_index()
    # metloom returns the variable column named after the variable's `name`
    # attribute. SWE.name == "SWE".
    swe_col = next((c for c in df.columns if str(c).upper() == "SWE"), None)
    date_col = next((c for c in df.columns if str(c).lower() in ("datetime", "date")), None)
    if swe_col is None or date_col is None:
        return pd.DataFrame(columns=["date", "triplet", "station_name", "swe_in"])
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col]).dt.tz_localize(None).dt.normalize(),
            "triplet": triplet,
            "station_name": station_name,
            "swe_in": pd.to_numeric(df[swe_col], errors="coerce"),
        }
    )
    return out.dropna(subset=["swe_in"]).reset_index(drop=True)


def _station_name(pt: SnotelPointData, fallback: str) -> str:
    try:
        meta = pt.metadata
    except Exception:
        return fallback
    if meta is None:
        return fallback
    if isinstance(meta, dict):
        return meta.get("name") or fallback
    return getattr(meta, "name", fallback) or fallback


def _as_datetime(d: str | date) -> datetime:
    if isinstance(d, str):
        d = date.fromisoformat(d)
    if isinstance(d, datetime):
        return d
    return datetime.combine(d, datetime.min.time())
