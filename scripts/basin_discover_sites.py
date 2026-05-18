"""Discover SNOTEL stations within a basin's upstream-of-gauge polygon.

Pulls the polygon from USGS NLDI (cached) and asks NRCS via metloom for
SWE-reporting SNOTEL stations within it. Prints the discovered triplets and
a paste-ready Python snippet for `basins.py`.

Usage:
    python scripts/basin_discover_sites.py --basin yampa_steamboat
"""
from __future__ import annotations

import argparse

from co_river_flow_forecast.basins import get_basin
from co_river_flow_forecast.data import discover_snotel_sites


def main(basin_short: str) -> None:
    basin = get_basin(basin_short)
    print(f"Discovering SNOTEL sites for {basin.name} (USGS {basin.usgs_id})...")
    sites = discover_snotel_sites(basin)
    print(f"Found {len(sites)} SWE-reporting SNOTEL stations in upstream basin:")
    print()
    print(f"  {'triplet':>14}  {'name':<28}  {'elev (ft)':>9}  {'lat':>8}  {'lon':>9}")
    print(f"  {'-'*14}  {'-'*28}  {'-'*9}  {'-'*8}  {'-'*9}")
    for s in sites:
        elev = f"{s['elevation_ft']:>9,.0f}" if s.get("elevation_ft") else "        ?"
        lat = f"{s['lat']:>8.3f}" if s.get("lat") is not None else "       ?"
        lon = f"{s['lon']:>9.3f}" if s.get("lon") is not None else "        ?"
        print(f"  {s['triplet']:>14}  {s['name']:<28}  {elev}  {lat}  {lon}")

    print()
    print("Paste into basins.py to replace the basin's snotel_triplets:")
    print()
    print("    snotel_triplets=(")
    for s in sites:
        print(f"        \"{s['triplet']}\",  # {s['name']}")
    print("    ),")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--basin", default="yampa_steamboat", help="Basin short_name")
    args = p.parse_args()
    main(args.basin)
