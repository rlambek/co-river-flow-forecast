"""Annual peak-flow climatology for a registered basin.

Pulls USGS daily discharge for the basin's outlet gauge and reports peak-day
and peak-magnitude climatology. Writes annual peaks to
`data/processed/<basin_slug>/annual_peaks.csv`.

Usage:
    python scripts/basin_flow_climatology.py --basin yampa_steamboat
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from co_river_flow_forecast.basins import get_basin
from co_river_flow_forecast.data import fetch_daily_streamflow
from co_river_flow_forecast.water_year import (
    doy_to_calendar,
    water_year,
    water_year_doy,
)

MIN_DAYS_FOR_COMPLETE_WY = 360


def main(basin_short: str) -> None:
    basin = get_basin(basin_short)
    print(f"Fetching daily discharge for USGS {basin.usgs_id} ({basin.name})...")
    df = fetch_daily_streamflow(basin.usgs_id)
    df = df.dropna(subset=["discharge_cfs"])
    if df.empty:
        raise SystemExit("No data returned.")
    print(f"  rows:       {len(df):,}")
    print(f"  date span:  {df.index.min().date()} -> {df.index.max().date()}")

    df = df.copy()
    df["water_year"] = df.index.map(water_year)
    df["wy_doy"] = df.index.map(water_year_doy)

    peak_idx = df.groupby("water_year")["discharge_cfs"].idxmax()
    peaks = (
        df.loc[peak_idx, ["water_year", "wy_doy", "discharge_cfs"]]
        .reset_index()
        .rename(columns={"datetime": "peak_date"})
    )

    counts = df.groupby("water_year").size()
    complete = counts[counts >= MIN_DAYS_FOR_COMPLETE_WY].index
    peaks = peaks[peaks["water_year"].isin(complete)].reset_index(drop=True)
    print(f"  complete water years (>= {MIN_DAYS_FOR_COMPLETE_WY} days): {len(peaks)}")
    if peaks.empty:
        raise SystemExit("No complete water years; nothing to summarize.")

    print()
    print(f"Long-term peak climatology ({len(peaks)} water years):")
    _summarize(peaks)

    most_recent = peaks["water_year"].max()
    recent = peaks[peaks["water_year"] >= most_recent - 9]
    print()
    print(f"Most recent {len(recent)} water years ({recent['water_year'].min()}-{most_recent}):")
    _summarize(recent)

    out_dir = os.path.join("data", "processed", basin.short_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "annual_peaks.csv")
    peaks.to_csv(out_path, index=False)
    print()
    print(f"Wrote {out_path}")


def _summarize(peaks: pd.DataFrame) -> None:
    mean_doy = peaks["wy_doy"].mean()
    median_doy = peaks["wy_doy"].median()
    print(f"  mean peak DOY (Oct-1 = 1):    {mean_doy:5.1f}   ({doy_to_calendar(mean_doy)})")
    print(f"  median peak DOY:              {median_doy:5.0f}   ({doy_to_calendar(median_doy)})")
    print(f"  std of peak DOY:              {peaks['wy_doy'].std():5.1f} days")
    print(f"  mean peak discharge:          {peaks['discharge_cfs'].mean():>7,.0f} cfs")
    print(f"  median peak discharge:        {peaks['discharge_cfs'].median():>7,.0f} cfs")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--basin", default="yampa_steamboat", help="Basin short_name (see basins.py)")
    args = p.parse_args()
    main(args.basin)
