"""Summarize upstream reservoir inflow/outflow for a basin's registered pairs.

Pulls daily inflow and outflow from CDSS for each reservoir pair in the basin
and writes per-pair CSVs to data/processed/<basin_slug>/. Prints recent
summary statistics.

Usage:
    python scripts/basin_reservoir_summary.py --basin yampa_steamboat --start 2020-10-01
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from co_river_flow_forecast.basins import get_basin
from co_river_flow_forecast.data.reservoirs import fetch_reservoir_pair_daily


def main(basin_short: str, start: str, end: str | None) -> None:
    basin = get_basin(basin_short)
    if not basin.reservoir_pairs:
        raise SystemExit(f"No reservoir pairs registered for {basin.short_name}.")

    out_dir = os.path.join("data", "processed", basin.short_name)
    os.makedirs(out_dir, exist_ok=True)

    for pair in basin.reservoir_pairs:
        print(f"\nReservoir: {pair.name}")
        print(f"  inflow:  {pair.inflow_abbrev}")
        print(f"  outflow: {pair.outflow_abbrev}")
        df = fetch_reservoir_pair_daily(pair.inflow_abbrev, pair.outflow_abbrev, start=start, end=end)
        if df.empty:
            print("  (no data returned)")
            continue
        print(f"  rows:    {len(df):,}")
        print(f"  span:    {df.index.min().date()} -> {df.index.max().date()}")
        print(f"  recent ({df.index.max().date()}): "
              f"inflow={df['inflow_cfs'].iloc[-1]:.0f} cfs, "
              f"outflow={df['outflow_cfs'].iloc[-1]:.0f} cfs, "
              f"net={df['net_cfs'].iloc[-1]:+.0f} cfs")
        peak_inflow = df["inflow_cfs"].max()
        peak_outflow = df["outflow_cfs"].max()
        print(f"  peak inflow over span:  {peak_inflow:,.0f} cfs on {df['inflow_cfs'].idxmax().date()}")
        print(f"  peak outflow over span: {peak_outflow:,.0f} cfs on {df['outflow_cfs'].idxmax().date()}")
        # Approximate cumulative net storage change in acre-feet
        cf_to_af_day = 1.9835  # 1 cfs for 1 day ~ 1.9835 acre-feet
        cum_af = (df["net_cfs"].fillna(0).cumsum() * cf_to_af_day)
        if len(cum_af):
            print(f"  cumulative net storage change over span: {cum_af.iloc[-1]:+,.0f} acre-feet")

        out_path = os.path.join(out_dir, f"reservoir_{pair.name.lower().replace(' ', '_')}.csv")
        df.to_csv(out_path)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--basin", default="yampa_steamboat", help="Basin short_name")
    p.add_argument("--start", default="2020-10-01", help="Earliest date (ISO)")
    p.add_argument("--end", default=None, help="Latest date (ISO), default today")
    args = p.parse_args()
    main(args.basin, args.start, args.end)
