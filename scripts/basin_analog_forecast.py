"""Analog-year forecast for a basin/window combination.

Finds the K historical water years whose flow state on the issue date most
closely resembles this year, then uses their actual target-window flows to
forecast. Compare to climatology and the scaled-climatology baselines.

Usage:
    python scripts/basin_analog_forecast.py \
        --basin green_green_river_ut \
        --target-start 2026-06-16 --target-end 2026-06-21 \
        --k 5 --lookback 7
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime

import pandas as pd

from co_river_flow_forecast.basins import get_basin
from co_river_flow_forecast.modeling.analog_forecast import forecast_with_analogs


def main(
    basin_short: str,
    target_start: str,
    target_end: str,
    as_of: str | None,
    k: int,
    lookback_days: int,
    use_swe: bool = False,
) -> None:
    basin = get_basin(basin_short)
    as_of_ts = pd.Timestamp(as_of) if as_of else pd.Timestamp(datetime.utcnow().date())

    print(f"Basin: {basin.name}")
    print(f"Target window: {target_start} -> {target_end}")
    print(f"As of: {as_of_ts.date()}     K = {k}     lookback = {lookback_days} days     SWE = {use_swe}")

    fc = forecast_with_analogs(
        basin,
        target_start=target_start,
        target_end=target_end,
        as_of=as_of_ts,
        k=k,
        lookback_days=lookback_days,
        use_swe=use_swe,
    )

    print()
    print(f"Current state vector (as of {fc.as_of.date()}):")
    for name, val in fc.current_state.items():
        if val == val:  # not NaN
            print(f"  {name:>30}: {val:>10,.0f}")
        else:
            print(f"  {name:>30}: NaN")

    print()
    print(f"Top-{k} analog years (closest by z-scored state distance):")
    print(f"  {'year':>6}  {'distance':>9}  {'window mean cfs':>16}")
    for year, dist in fc.analogs.items():
        flow = fc.analog_window_flows[year]
        print(f"  {year:>6}  {dist:>9.3f}  {flow:>16,.0f}")

    print()
    print(f"** Analog-year forecast for {fc.target_start.date()} .. {fc.target_end.date()}:")
    print(f"   median:    {fc.forecast_median:>10,.0f} cfs")
    print(f"   P25..P75:  {fc.forecast_p25:>10,.0f} .. {fc.forecast_p75:,.0f} cfs")
    print(f"   P10..P90:  {fc.forecast_p10:>10,.0f} .. {fc.forecast_p90:,.0f} cfs")
    print(f"   pool size: {fc.historical_states_used} historical years with complete data")

    out_dir = os.path.join("data", "processed", basin.short_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir,
        f"analog_forecast_k{k}_lb{lookback_days}_{target_start}_to_{target_end}.csv",
    )
    pd.DataFrame(
        [
            {
                "as_of": fc.as_of.date().isoformat(),
                "target_start": fc.target_start.date().isoformat(),
                "target_end": fc.target_end.date().isoformat(),
                "k": k,
                "lookback_days": lookback_days,
                "forecast_median": fc.forecast_median,
                "forecast_p10": fc.forecast_p10,
                "forecast_p25": fc.forecast_p25,
                "forecast_p75": fc.forecast_p75,
                "forecast_p90": fc.forecast_p90,
                "analog_years": ",".join(str(y) for y in fc.analogs.index),
                "analog_distances": ",".join(f"{d:.3f}" for d in fc.analogs.values),
                "analog_window_flows": ",".join(f"{v:.0f}" for v in fc.analog_window_flows.values()),
            }
        ]
    ).to_csv(out_path, index=False)
    print()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--basin", required=True)
    p.add_argument("--target-start", required=True)
    p.add_argument("--target-end", required=True)
    p.add_argument("--as-of", default=None, help="YYYY-MM-DD (default: today)")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--lookback", type=int, default=7)
    p.add_argument("--with-swe", action="store_true",
                   help="Include basin-aggregate SWE features in the state vector")
    args = p.parse_args()
    main(
        args.basin,
        args.target_start,
        args.target_end,
        args.as_of,
        args.k,
        args.lookback,
        use_swe=args.with_swe,
    )
