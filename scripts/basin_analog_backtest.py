"""Backtest the analog-year forecaster on a basin/window combination.

Sweeps a small grid of hyperparameters (K, lookback window) and reports
leave-one-year-out MAE and MAPE vs the climatology baseline. Use to pick
hyperparameters before running `basin_analog_forecast.py` for a real
forecast.

Usage:
    python scripts/basin_analog_backtest.py \
        --basin green_green_river_ut \
        --target-start 2026-06-16 --target-end 2026-06-21 \
        --as-of 05-25
"""
from __future__ import annotations

import argparse

import pandas as pd

from co_river_flow_forecast.basins import get_basin
from co_river_flow_forecast.modeling.analog_forecast import (
    backtest,
    backtest_summary,
)


def main(
    basin_short: str,
    target_start: str,
    target_end: str,
    as_of: str,
    ks: list[int],
    lookbacks: list[int],
) -> None:
    basin = get_basin(basin_short)
    month, day = (int(x) for x in as_of.split("-"))
    print(f"Backtest: {basin.name}")
    print(f"  target window: {target_start} -> {target_end}")
    print(f"  as-of (month-day): {as_of} of each year")
    print()
    print(f"  {'k':>3}  {'lookback':>8}  {'n':>3}  {'analog MAE':>11}  {'climo MAE':>10}  {'skill':>7}  {'analog MAPE':>12}  {'P10-P90 cov':>11}")
    print(f"  {'-'*3}  {'-'*8}  {'-'*3}  {'-'*11}  {'-'*10}  {'-'*7}  {'-'*12}  {'-'*11}")

    best = None
    for k in ks:
        for lb in lookbacks:
            df = backtest(
                basin,
                target_start=target_start,
                target_end=target_end,
                as_of_month=month,
                as_of_day=day,
                k=k,
                lookback_days=lb,
                trend_days=14,
            )
            s = backtest_summary(df)
            print(
                f"  {k:>3}  {lb:>8}  {s['n']:>3}  "
                f"{s['analog_mae']:>11,.0f}  {s['climo_mae']:>10,.0f}  "
                f"{s['skill_vs_climo_mae']*100:>6.0f}%  "
                f"{s['analog_mape']:>11.1f}%  "
                f"{s['p10_p90_coverage']*100:>10.0f}%"
            )
            if best is None or s["analog_mae"] < best[0]["analog_mae"]:
                best = (s, k, lb, df)

    s_best, k_best, lb_best, df_best = best
    print()
    print(f"Best by analog MAE: k={k_best}, lookback={lb_best}")
    print(f"  MAE  {s_best['analog_mae']:,.0f} cfs  vs  climo MAE  {s_best['climo_mae']:,.0f} cfs   "
          f"(skill {s_best['skill_vs_climo_mae']*100:.0f}%)")
    print(f"  MAPE {s_best['analog_mape']:.1f}%  vs  climo MAPE {s_best['climo_mape']:.1f}%")
    print(f"  P10-P90 coverage: {s_best['p10_p90_coverage']*100:.0f}% (ideal ~80%)")

    out_dir = f"data/processed/{basin.short_name}"
    df_best.to_csv(f"{out_dir}/analog_backtest_k{k_best}_lb{lb_best}_{target_start}_to_{target_end}.csv")
    print()
    print(f"Wrote {out_dir}/analog_backtest_k{k_best}_lb{lb_best}_{target_start}_to_{target_end}.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--basin", required=True)
    p.add_argument("--target-start", required=True, help="MM-DD or YYYY-MM-DD")
    p.add_argument("--target-end", required=True)
    p.add_argument("--as-of", required=True, help="MM-DD for the issue date")
    p.add_argument("--ks", default="3,5,7,10",
                   help="Comma-separated K values to sweep")
    p.add_argument("--lookbacks", default="3,7,14",
                   help="Comma-separated lookback-day values to sweep")
    args = p.parse_args()
    main(
        args.basin, args.target_start, args.target_end, args.as_of,
        ks=[int(x) for x in args.ks.split(",")],
        lookbacks=[int(x) for x in args.lookbacks.split(",")],
    )
