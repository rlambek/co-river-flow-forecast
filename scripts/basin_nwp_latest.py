"""Pull the latest GFS forecast at a basin's centroid as a smoke test.

Prints a small forecast table out to D+7 (lead hours 0, 24, 48, ..., 168)
and writes it to data/processed/<basin_slug>/gfs_latest.csv.

Usage:
    python scripts/basin_nwp_latest.py --basin yampa_steamboat
"""
from __future__ import annotations

import argparse
import os

from co_river_flow_forecast.basins import get_basin
from co_river_flow_forecast.data import fetch_gfs_forecast_at_point
from co_river_flow_forecast.data.basin_geometry import get_basin_centroid


def main(basin_short: str) -> None:
    basin = get_basin(basin_short)
    lat, lon = get_basin_centroid(basin)
    print(f"Basin: {basin.name}")
    print(f"  centroid: ({lat:.3f}, {lon:.3f})")

    print("Fetching latest GFS forecast (this downloads ~10 GRIB messages from AWS)...")
    df = fetch_gfs_forecast_at_point(lat, lon)
    init = df["init_time"].iloc[0]
    print(f"  init time (UTC): {init}")
    print()
    print(f"  {'lead (h)':>8}  {'valid (UTC)':<19}  {'T2m (C)':>8}  {'APCP (mm)':>10}")
    print(f"  {'-'*8}  {'-'*19}  {'-'*8}  {'-'*10}")
    for _, r in df.iterrows():
        valid = r["valid_time"].strftime("%Y-%m-%d %H:%M")
        t2m = f"{r['t2m_c']:>8.1f}" if r["t2m_c"] is not None and not _isnan(r["t2m_c"]) else "       ?"
        apcp = f"{r['apcp_mm_interval']:>10.1f}" if r["apcp_mm_interval"] is not None and not _isnan(r["apcp_mm_interval"]) else "         ?"
        print(f"  {int(r['lead_hour']):>8}  {valid:<19}  {t2m}  {apcp}")

    out_dir = os.path.join("data", "processed", basin.short_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "gfs_latest.csv")
    df.to_csv(out_path, index=False)
    print()
    print(f"Wrote {out_path}")


def _isnan(x) -> bool:
    try:
        return x != x  # NaN check that works for python floats
    except Exception:
        return False


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--basin", default="yampa_steamboat", help="Basin short_name")
    args = p.parse_args()
    main(args.basin)
