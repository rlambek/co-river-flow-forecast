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
