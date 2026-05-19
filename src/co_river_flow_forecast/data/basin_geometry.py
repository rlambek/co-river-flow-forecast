"""Basin polygon retrieval via USGS NLDI.

For any USGS gauge, NLDI returns the upstream-of-gauge drainage polygon as
GeoJSON. This is more accurate than the basin's HUC8 (which can include
downstream area) and is the right geometry for siting SNOTEL stations,
basin-averaging gridded products, etc.
"""
from __future__ import annotations

import json
import os
from typing import Any

import requests
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from co_river_flow_forecast.basins import Basin

NLDI_BASE = "https://api.water.usgs.gov/nldi/linked-data"
CACHE_DIR = os.path.join("data", "cache", "basin_geometry")
REQUEST_TIMEOUT_SEC = 60


def get_basin_polygon(basin: Basin, refresh: bool = False) -> BaseGeometry:
    """Return the upstream-of-gauge basin polygon as a shapely geometry (EPSG:4326).

    Caches the raw NLDI GeoJSON to `data/cache/basin_geometry/<short_name>.geojson`.
    Pass `refresh=True` to bypass the cache.
    """
    geojson = _load_basin_geojson(basin, refresh=refresh)
    features = geojson.get("features") or []
    if not features:
        raise ValueError(f"NLDI returned no basin polygon for {basin.usgs_id}.")
    return shape(features[0]["geometry"])


def get_basin_centroid(basin: Basin) -> tuple[float, float]:
    """Return the basin polygon centroid as (lat, lon) in EPSG:4326."""
    polygon = get_basin_polygon(basin)
    c = polygon.centroid
    return float(c.y), float(c.x)


def polygon_grid_mean(
    values: "Any",  # 2D numpy array indexed [lat_i, lon_j]
    lats: "Any",   # 1D array of latitudes (EPSG:4326)
    lons: "Any",   # 1D array of longitudes (could be 0..360 or -180..180)
    polygon: BaseGeometry,
) -> tuple[float, int] | tuple[None, int]:
    """Mean of grid-cell values whose centroid falls inside `polygon`.

    Returns (mean_value, n_cells). If no cells' centroids are inside the
    polygon, returns (None, 0); callers should fall back to nearest-cell.
    """
    import numpy as np
    from shapely.geometry import Point
    from shapely.prepared import prep

    minlon, minlat, maxlon, maxlat = polygon.bounds

    # Normalize the grid's longitude convention to match the polygon (-180..180).
    grid_lons = np.asarray(lons, dtype=float)
    grid_lats = np.asarray(lats, dtype=float)
    grid_lons_180 = np.where(grid_lons > 180.0, grid_lons - 360.0, grid_lons)

    lat_mask = (grid_lats >= minlat) & (grid_lats <= maxlat)
    lon_mask = (grid_lons_180 >= minlon) & (grid_lons_180 <= maxlon)
    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]
    if lat_idx.size == 0 or lon_idx.size == 0:
        return None, 0

    prepared = prep(polygon)
    samples: list[float] = []
    arr = np.asarray(values)
    for i in lat_idx:
        for j in lon_idx:
            if prepared.contains(Point(float(grid_lons_180[j]), float(grid_lats[i]))):
                samples.append(float(arr[i, j]))
    if not samples:
        return None, 0
    return float(sum(samples) / len(samples)), len(samples)


def _load_basin_geojson(basin: Basin, refresh: bool = False) -> dict[str, Any]:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{basin.short_name}.geojson")
    if not refresh and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    url = f"{NLDI_BASE}/nwissite/USGS-{basin.usgs_id}/basin"
    resp = requests.get(url, timeout=REQUEST_TIMEOUT_SEC)
    resp.raise_for_status()
    geojson = resp.json()
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f)
    return geojson
