"""Forecast mean flow in a target date window for a basin.

For lead times beyond NWP range, runs a scaled-climatology forecast at the
basin outlet. If the basin has registered contributor gauges, they get the
same forecast individually to give context.

Always prints a 1/3/7/14-day trend table so a sharp recession is visible.
Use `--ratio-window` to control which window drives the primary forecast
(default 3 days, which handles recessions correctly).

Usage:
    python scripts/basin_window_forecast.py --basin green_green_river_ut \
        --start 2026-06-16 --end 2026-06-21
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from co_river_flow_forecast.basins import get_basin
from co_river_flow_forecast.modeling.window_forecast import forecast_flow_window


def _print_forecast(label: str, site_id: str, out: dict, ratio_window: int) -> None:
    print(f"\n{label} (USGS {site_id}):")
    print(f"  data through:                {out['data_through']}    most recent: {out['most_recent_cfs']:>8,.0f} cfs")
    print(f"  climatology n years:         {out['n_years_climo']}")
    print(f"  target-window climo median:  {out['climo_median_target_cfs']:>10,.0f} cfs   (P10..P90 = {out['climo_p10_target_cfs']:,.0f} .. {out['climo_p90_target_cfs']:,.0f})")

    print(f"  trend (recent / same-date climo / ratio):")
    print(f"    {'window':>7}  {'recent':>10}  {'climo':>10}  {'ratio':>8}  {'forecast':>10}")
    for w_str, t in out["trends"].items():
        marker = "  <- primary" if w_str == f"{ratio_window}d" else ""
        forecast_w = out["climo_median_target_cfs"] * t["ratio"] if t["ratio"] == t["ratio"] else float("nan")
        print(f"    {w_str:>7}  {t['mean']:>10,.0f}  {t['climo']:>10,.0f}  {t['ratio']:>8.2f}x  {forecast_w:>10,.0f}{marker}")

    print()
    print(f"  ** Scaled forecast for {out['target_start']} .. {out['target_end']}:")
    print(f"     median:    {out['forecast_median_cfs']:>10,.0f} cfs")
    print(f"     P10..P90:  {out['forecast_p10_cfs']:>10,.0f} .. {out['forecast_p90_cfs']:,.0f} cfs")


def main(basin_short: str, target_start: str, target_end: str, ratio_window: int) -> None:
    basin = get_basin(basin_short)
    print(f"Basin: {basin.name}")
    print(f"Target window: {target_start} to {target_end}")
    print(f"Ratio window for primary forecast: {ratio_window} days")

    rows = []

    outlet = forecast_flow_window(basin.usgs_id, target_start, target_end, ratio_window_days=ratio_window)
    _print_forecast(f"Outlet: {basin.short_name}", basin.usgs_id, outlet, ratio_window)
    rows.append({"role": "outlet", "label": basin.short_name, **{k: v for k, v in outlet.items() if k != "trends"}})

    for label, sid in basin.contributor_usgs_ids:
        try:
            c = forecast_flow_window(sid, target_start, target_end, ratio_window_days=ratio_window)
            _print_forecast(label, sid, c, ratio_window)
            rows.append({"role": "contributor", "label": label, **{k: v for k, v in c.items() if k != "trends"}})
        except Exception as exc:  # noqa: BLE001
            print(f"\n{label} (USGS {sid}): ERROR -- {exc}")

    if rows[1:]:
        sum_recent = sum(r["recent_mean_cfs"] for r in rows[1:])
        sum_forecast = sum(r["forecast_median_cfs"] for r in rows[1:])
        print()
        print(f"Contributor sum vs outlet (sanity, no routing lag):")
        print(f"  recent mean:    contributors={sum_recent:>10,.0f}   outlet={outlet['recent_mean_cfs']:>10,.0f}   ratio={sum_recent/outlet['recent_mean_cfs']:.2f}")
        print(f"  forecast mean:  contributors={sum_forecast:>10,.0f}   outlet={outlet['forecast_median_cfs']:>10,.0f}   ratio={sum_forecast/outlet['forecast_median_cfs']:.2f}")

    out_dir = os.path.join("data", "processed", basin.short_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"window_forecast_{target_start}_to_{target_end}.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--basin", required=True, help="Basin short_name")
    p.add_argument("--start", required=True, help="Target window start, ISO (YYYY-MM-DD)")
    p.add_argument("--end", required=True, help="Target window end, ISO (YYYY-MM-DD)")
    p.add_argument(
        "--ratio-window", type=int, default=3,
        help="Window (days) for the primary scaling ratio. Short (1-3) during a sharp recession, "
             "longer (7-14) during steady state. Default 3.",
    )
    args = p.parse_args()
    main(args.basin, args.start, args.end, args.ratio_window)
