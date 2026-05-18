"""SNOTEL SWE summary for a registered basin.

Pulls daily SWE for each of the basin's SNOTEL stations, computes per-site
peak-SWE day and peak-SWE magnitude per water year, and joins those against
the annual-peak-flow CSV produced by basin_flow_climatology.py so we can see
how SWE-derived predictors covary with peak flow.

Writes:
  data/processed/<basin_slug>/snotel_daily.csv     (raw daily SWE, long format)
  data/processed/<basin_slug>/snotel_annual.csv    (per-site water-year peak SWE)
  data/processed/<basin_slug>/swe_vs_flow.csv      (joined, if annual_peaks.csv exists)

Usage:
    python scripts/basin_snotel_summary.py --basin yampa_steamboat --start 1990-10-01
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from co_river_flow_forecast.basins import get_basin
from co_river_flow_forecast.data import discover_snotel_sites, fetch_daily_swe
from co_river_flow_forecast.water_year import doy_to_calendar, water_year, water_year_doy


def main(basin_short: str, start: str, discover: bool = False) -> None:
    basin = get_basin(basin_short)
    triplets: tuple[str, ...] = basin.snotel_triplets
    if discover or not triplets:
        print(f"Discovering SNOTEL stations in upstream basin of {basin.usgs_id}...")
        triplets = tuple(s["triplet"] for s in discover_snotel_sites(basin))
        print(f"  discovered {len(triplets)} stations.")

    if not triplets:
        raise SystemExit(f"No SNOTEL stations available for {basin.short_name}.")

    print(f"Fetching daily SWE for {len(triplets)} SNOTEL stations in {basin.name}")
    print(f"  triplets: {', '.join(triplets)}")
    print(f"  start:    {start}")
    daily = fetch_daily_swe(triplets, start=start)
    if daily.empty:
        raise SystemExit("metloom returned no data.")
    print(f"  rows:     {len(daily):,}")
    print(f"  stations: {sorted(daily['triplet'].unique())}")
    print(f"  span:     {daily['date'].min().date()} -> {daily['date'].max().date()}")

    daily["water_year"] = daily["date"].map(water_year)
    daily["wy_doy"] = daily["date"].map(water_year_doy)

    # Per-station, per-water-year peak SWE
    annual = (
        daily.sort_values("swe_in", ascending=False)
        .drop_duplicates(["triplet", "water_year"])
        .sort_values(["triplet", "water_year"])
        .rename(columns={"date": "peak_swe_date", "swe_in": "peak_swe_in", "wy_doy": "peak_swe_wy_doy"})
        [["triplet", "station_name", "water_year", "peak_swe_date", "peak_swe_wy_doy", "peak_swe_in"]]
        .reset_index(drop=True)
    )

    print()
    print("Per-station peak SWE climatology:")
    for triplet, sub in annual.groupby("triplet"):
        name = sub["station_name"].iloc[0]
        mean_doy = sub["peak_swe_wy_doy"].mean()
        mean_swe = sub["peak_swe_in"].mean()
        print(
            f"  {triplet:>14}  {name:<24}  n={len(sub):3d}  "
            f"mean peak SWE = {mean_swe:5.1f} in   "
            f"mean peak day = {mean_doy:5.1f}  ({doy_to_calendar(mean_doy)})"
        )

    out_dir = os.path.join("data", "processed", basin.short_name)
    os.makedirs(out_dir, exist_ok=True)
    daily.to_csv(os.path.join(out_dir, "snotel_daily.csv"), index=False)
    annual.to_csv(os.path.join(out_dir, "snotel_annual.csv"), index=False)
    print()
    print(f"Wrote {out_dir}/snotel_daily.csv")
    print(f"Wrote {out_dir}/snotel_annual.csv")

    # If annual_peaks.csv exists (from basin_flow_climatology.py), join and correlate.
    flow_path = os.path.join(out_dir, "annual_peaks.csv")
    if not os.path.exists(flow_path):
        print()
        print(f"Skip SWE-vs-flow join: run basin_flow_climatology.py first.")
        return

    flow = pd.read_csv(flow_path)
    # Basin-aggregate SWE: mean across stations within each water year.
    basin_swe = (
        annual.groupby("water_year")
        .agg(basin_peak_swe_in=("peak_swe_in", "mean"),
             basin_peak_swe_wy_doy=("peak_swe_wy_doy", "mean"),
             n_stations=("triplet", "nunique"))
        .reset_index()
    )
    joined = flow.merge(basin_swe, on="water_year", how="inner")
    joined.to_csv(os.path.join(out_dir, "swe_vs_flow.csv"), index=False)
    print(f"Wrote {out_dir}/swe_vs_flow.csv  ({len(joined)} overlapping water years)")

    print()
    print("Pearson correlations across overlapping water years:")
    pairs = [
        ("basin_peak_swe_in", "discharge_cfs",  "peak SWE depth     vs peak discharge"),
        ("basin_peak_swe_wy_doy", "wy_doy",     "peak SWE day       vs peak flow day"),
        ("basin_peak_swe_in", "wy_doy",         "peak SWE depth     vs peak flow day"),
    ]
    for x, y, label in pairs:
        r = joined[[x, y]].corr().iloc[0, 1]
        print(f"  {label:<50}  r = {r:+.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--basin", default="yampa_steamboat", help="Basin short_name (see basins.py)")
    p.add_argument("--start", default="1990-10-01", help="Earliest date to pull (ISO)")
    p.add_argument(
        "--discover",
        action="store_true",
        help="Discover SNOTEL stations from NLDI polygon, ignoring any curated list in basins.py",
    )
    args = p.parse_args()
    main(args.basin, args.start, discover=args.discover)
