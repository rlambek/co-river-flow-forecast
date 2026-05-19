"""Pull the latest GFS forecast over a basin polygon as a smoke test.

By default averages GFS grid cells whose centroids fall inside the upstream-
of-gauge basin polygon. Pass --centroid to extract at a single point instead.
Writes the result to data/processed/<basin_slug>/gfs_latest.csv.

Usage:
    python scripts/basin_nwp_latest.py --basin yampa_steamboat
    python scripts/basin_nwp_latest.py --basin clear_creek_golden --centroid
"""
from __future__ import annotations

import argparse
import math
import os

from co_river_flow_forecast.basins import get_basin
from co_river_flow_forecast.data import (
    fetch_gfs_forecast_at_point,
    fetch_gfs_forecast_for_basin,
)
from co_river_flow_forecast.data.basin_geometry import get_basin_centroid


def main(basin_short: str, mode: str) -> None:
    basin = get_basin(basin_short)
    print(f"Basin: {basin.name}")

    if mode == "basin_mean":
        print("Mode: basin-polygon mean")
        df = fetch_gfs_forecast_for_basin(basin)
        n_cells = int(df["n_cells"].max()) if len(df) else 0
        print(f"  GFS cells covering basin: {n_cells}")
    else:
        lat, lon = get_basin_centroid(basin)
        print(f"Mode: single point at centroid ({lat:.3f}, {lon:.3f})")
        df = fetch_gfs_forecast_at_point(lat, lon)

    init = df["init_time"].iloc[0]
    print(f"  init time (UTC): {init}")
    print()
    print(f"  {'lead (h)':>8}  {'valid (UTC)':<19}  {'T2m (C)':>8}  {'APCP (mm)':>10}")
    print(f"  {'-'*8}  {'-'*19}  {'-'*8}  {'-'*10}")
    for _, r in df.iterrows():
        valid = r["valid_time"].strftime("%Y-%m-%d %H:%M")
        t2m = f"{r['t2m_c']:>8.1f}" if _is_real(r["t2m_c"]) else "       ?"
        apcp = f"{r['apcp_mm_interval']:>10.1f}" if _is_real(r["apcp_mm_interval"]) else "         ?"
        print(f"  {int(r['lead_hour']):>8}  {valid:<19}  {t2m}  {apcp}")

    out_dir = os.path.join("data", "processed", basin.short_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "gfs_latest.csv")
    df.to_csv(out_path, index=False)
    print()
    print(f"Wrote {out_path}")


def _is_real(x) -> bool:
    try:
        return x is not None and not math.isnan(float(x))
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--basin", default="yampa_steamboat", help="Basin short_name")
    p.add_argument(
        "--centroid",
        action="store_true",
        help="Extract at the basin centroid instead of polygon mean",
    )
    args = p.parse_args()
    main(args.basin, mode="centroid" if args.centroid else "basin_mean")
