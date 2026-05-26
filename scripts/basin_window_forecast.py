"""Forecast mean flow in a target date window for a basin.

For lead times beyond NWP range, runs a scaled-climatology forecast at the
basin outlet. If the basin has registered contributor gauges (upstream
tributaries / reservoir releases), they get the same forecast individually
to give context.

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


def _print_forecast(label: str, site_id: str, out: dict) -> None:
    print(f"\n{label} (USGS {site_id}):")
    print(f"  data through:               {out['data_through']}")
    print(f"  climatology n years:        {out['n_years_climo']}")
    print(f"  target-window climo (median): {out['climo_median_target_cfs']:>10,.0f} cfs")
    print(f"                P10..P90:       {out['climo_p10_target_cfs']:>10,.0f} .. {out['climo_p90_target_cfs']:,.0f} cfs")
    print(f"  recent {out['recent_window_days']}-day mean:     {out['recent_mean_cfs']:>10,.0f} cfs")
    print(f"  same-window climo (recent): {out['same_window_climo_mean_cfs']:>10,.0f} cfs")
    print(f"  scaling ratio:              {out['scaling_ratio']:>10.2f}x")
    print()
    print(f"  ** Scaled forecast for {out['target_start']} .. {out['target_end']}:")
    print(f"     median:    {out['forecast_median_cfs']:>10,.0f} cfs")
    print(f"     P10..P90:  {out['forecast_p10_cfs']:>10,.0f} .. {out['forecast_p90_cfs']:,.0f} cfs")


def main(basin_short: str, target_start: str, target_end: str) -> None:
    basin = get_basin(basin_short)
    print(f"Basin: {basin.name}")
    print(f"Target window: {target_start} to {target_end}")

    rows = []

    outlet = forecast_flow_window(basin.usgs_id, target_start, target_end)
    _print_forecast(f"Outlet: {basin.short_name}", basin.usgs_id, outlet)
    rows.append({"role": "outlet", "label": basin.short_name, **outlet})

    for label, sid in basin.contributor_usgs_ids:
        try:
            c = forecast_flow_window(sid, target_start, target_end)
            _print_forecast(label, sid, c)
            rows.append({"role": "contributor", "label": label, **c})
        except Exception as exc:  # noqa: BLE001
            print(f"\n{label} (USGS {sid}): ERROR -- {exc}")

    # Sanity check: do the contributors approximately sum to the outlet?
    if rows[1:]:
        sum_recent = sum(r["recent_mean_cfs"] for r in rows[1:])
        sum_forecast = sum(r["forecast_median_cfs"] for r in rows[1:])
        print()
        print(f"Contributor sum vs outlet (sanity, no lag adjustment):")
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
    args = p.parse_args()
    main(args.basin, args.start, args.end)
