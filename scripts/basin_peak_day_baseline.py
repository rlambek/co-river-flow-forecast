"""Peak-flow-day baselines for a registered basin.

Builds a per-water-year feature frame from SNOTEL peak-SWE features and the
USGS peak-flow record, then evaluates several baselines via leave-one-
water-year-out CV. Reports MAE in days plus 7-day and 14-day window hit
rates.

Usage:
    python scripts/basin_peak_day_baseline.py --basin yampa_steamboat
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from co_river_flow_forecast.basins import get_basin
from co_river_flow_forecast.modeling.peak_day import (
    ClimatologyMean,
    build_feature_frame,
    evaluate_loo,
)
from co_river_flow_forecast.water_year import doy_to_calendar

MODELS = [
    ("climatology_mean",  lambda: ClimatologyMean()),
    ("ridge_alpha_1",     None),    # filled below after sklearn import
    ("gradient_boost",    None),
]


def main(basin_short: str, start: str) -> None:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    factories = [
        ("climatology_mean",  ClimatologyMean),
        ("ridge_alpha_1",     lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        ("gradient_boost",    lambda: GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0
        )),
    ]

    basin = get_basin(basin_short)
    print(f"Building feature frame for {basin.name}...")
    tf = build_feature_frame(basin, snotel_start=start)
    print(f"  training rows: {len(tf.X)}")
    print(f"  features:      {tf.feature_cols}")
    print(f"  target mean:   {tf.y.mean():.1f} ({doy_to_calendar(tf.y.mean())})")
    print(f"  target std:    {tf.y.std():.1f} days")

    rows = []
    for name, factory in factories:
        try:
            m = evaluate_loo(tf, factory)
        except Exception as exc:  # noqa: BLE001
            print(f"  {name}: FAILED ({exc})")
            continue
        rows.append({
            "model": name,
            "n": m["n"],
            "mae_days": m["mae_days"],
            "rmse_days": m["rmse_days"],
            "bias_days": m["bias_days"],
            "hit_rate_7d": m["hit_rate_7d_window"],
            "hit_rate_14d": m["hit_rate_14d_window"],
        })

    if not rows:
        raise SystemExit("All models failed.")
    out = pd.DataFrame(rows).set_index("model")

    print()
    print("Leave-one-water-year-out evaluation (peak flow DOY in days):")
    print()
    print(f"  {'model':<20}  {'n':>3}  {'MAE':>6}  {'RMSE':>6}  {'bias':>6}  {'hit 7d':>7}  {'hit 14d':>8}")
    print(f"  {'-'*20}  {'-'*3}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*8}")
    for model, row in out.iterrows():
        print(
            f"  {model:<20}  {int(row['n']):>3}  "
            f"{row['mae_days']:>6.2f}  {row['rmse_days']:>6.2f}  {row['bias_days']:>+6.2f}  "
            f"{row['hit_rate_7d']*100:>6.0f}%  {row['hit_rate_14d']*100:>7.0f}%"
        )

    out_dir = os.path.join("data", "processed", basin.short_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "peak_day_baselines.csv")
    out.to_csv(out_path)
    print()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--basin", default="yampa_steamboat", help="Basin short_name")
    p.add_argument("--start", default="1990-10-01", help="Earliest SNOTEL date to pull (ISO)")
    args = p.parse_args()
    main(args.basin, args.start)
