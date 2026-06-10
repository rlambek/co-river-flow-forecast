"""Daily 7-day flow forecast for a basin's outlet, from current conditions.

Uses the analog-year ensemble (ESP-style trace scaling) to produce a daily
P10/P25/P50/P75/P90 forecast for the next `horizon` days, saves a chart PNG,
and prints the table.

Usage:
    python scripts/basin_daily_forecast.py --basin clear_creek_golden
    python scripts/basin_daily_forecast.py --basin clear_creek_golden --as-of 2026-05-29 --horizon 7
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from co_river_flow_forecast.basins import get_basin
from co_river_flow_forecast.modeling.daily_trajectory import analog_daily_trajectory


def main(basin_short: str, as_of: str | None, horizon: int, k: int, lookback: int) -> None:
    basin = get_basin(basin_short)
    print(f"Basin: {basin.name}  (USGS {basin.usgs_id})")

    traj = analog_daily_trajectory(
        basin, as_of=as_of, horizon_days=horizon, k=k, lookback_days=lookback,
    )

    print(f"As of (current conditions): {traj.as_of.date()}")
    print(f"Recent {lookback}-day mean discharge: {traj.current_recent_cfs:,.0f} cfs")
    print(f"Analog years (distance, trace scale):")
    for y, d in traj.analogs.items():
        print(f"    {y}   dist={d:.3f}   scale={traj.analog_scales[y]:.2f}x")

    print()
    fc = traj.forecast
    print("Daily forecast (cfs):")
    print(f"  {'date':<12} {'P10':>8} {'P25':>8} {'P50':>8} {'P75':>8} {'P90':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for d, r in fc.iterrows():
        print(f"  {d.date().isoformat():<12} {r['p10']:>8,.0f} {r['p25']:>8,.0f} "
              f"{r['p50']:>8,.0f} {r['p75']:>8,.0f} {r['p90']:>8,.0f}")

    # --- chart ---
    out_dir = os.path.join("data", "processed", basin.short_name)
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, f"daily_forecast_{traj.as_of.date()}_{horizon}d.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = fc.index

    # Thin analog traces in the background.
    for y in traj.analog_traces.columns:
        ax.plot(x, traj.analog_traces[y].values, color="0.8", lw=0.8, zorder=1)

    ax.fill_between(x, fc["p10"], fc["p90"], color="#4C9BE8", alpha=0.25, label="P10-P90", zorder=2)
    ax.fill_between(x, fc["p25"], fc["p75"], color="#4C9BE8", alpha=0.40, label="P25-P75", zorder=3)
    ax.plot(x, fc["p50"], color="#0B3D91", lw=2.5, marker="o", label="P50 (median)", zorder=4)

    ax.set_title(
        f"{basin.name}\nDaily flow forecast, {horizon} days from {traj.as_of.date()} "
        f"(analog ensemble, K={k})"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Discharge (cfs)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)

    csv_path = os.path.join(out_dir, f"daily_forecast_{traj.as_of.date()}_{horizon}d.csv")
    fc.to_csv(csv_path)
    print()
    print(f"Wrote chart: {png_path}")
    print(f"Wrote table: {csv_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--basin", required=True)
    p.add_argument("--as-of", default=None, help="YYYY-MM-DD (default: latest observation)")
    p.add_argument("--horizon", type=int, default=7)
    p.add_argument("--k", type=int, default=7)
    p.add_argument("--lookback", type=int, default=7)
    args = p.parse_args()
    main(args.basin, args.as_of, args.horizon, args.k, args.lookback)
