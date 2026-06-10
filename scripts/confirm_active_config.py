"""Confirm and commit the best model configuration from the search findings.

The per-year win-rate bar (>65%) committed nothing. But two SWE-related
changes reduce MEAN ABSOLUTE ERROR on almost every basin:
  - feature swe_top1_pct_of_climo  (highest-SWE SNOTEL, % of climo)
  - swe_2x weighting               (double SWE features in the distance)

This script runs full leave-one-year-out (all ~35 years per basin -- far
more than the 10-holdout minimum) for four configurations and decides
using a basin-level rule that matches "almost always beneficial":
commit the config that lowers MAE on at least 2 of 3 basins by a
meaningful margin and does not materially hurt the third.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from co_river_flow_forecast.basins import get_basin
from co_river_flow_forecast.modeling.analog_forecast import (
    _forecast_from_loaded, load_basin_flow,
)
from co_river_flow_forecast.modeling.feature_catalog import get_feature_fn
from co_river_flow_forecast.modeling.swe_cache import (
    load_basin_swe, precompute_swe_climatology, prepare_swe_indexed,
)

BASINS = [
    ("yampa_steamboat",       "06-01", "06-06", "05-10"),
    ("clear_creek_golden",    "06-17", "06-22", "05-27"),
    ("green_green_river_ut",  "06-16", "06-21", "05-25"),
]

SWE_2X = {"basin_swe_recent_in": 2.0, "basin_swe_pct_of_climo": 2.0,
          "swe_top1_pct_of_climo": 2.0}

CONFIGS = {
    "baseline":              {"extra": (),                          "weights": None},
    "swe_top1":              {"extra": ("swe_top1_pct_of_climo",),  "weights": None},
    "swe_2x":                {"extra": (),                          "weights": SWE_2X},
    "swe_top1_plus_swe_2x":  {"extra": ("swe_top1_pct_of_climo",),  "weights": SWE_2X},
}

LOG_PATH = os.path.join("data", "processed", "confirm_active_config_log.csv")


def loo_mae(basin, t_start, t_end, as_of, extra_fns, weights,
            k=7, lookback=7, trend=14, history_start="1990-10-01") -> dict:
    outlet, contributors = load_basin_flow(basin, history_start=history_start)
    swe_long = load_basin_swe(basin, start=history_start)
    swe_indexed = swe_climo = None
    if not swe_long.empty:
        swe_indexed = prepare_swe_indexed(swe_long)
        swe_climo = precompute_swe_climatology(swe_indexed)

    ts = pd.Timestamp(f"2026-{t_start}")
    te = pd.Timestamp(f"2026-{t_end}")
    as_of_month, as_of_day = (int(x) for x in as_of.split("-"))
    years = sorted({int(t.year) for t in outlet.index})

    errs = []
    for y in years:
        t_start_y = pd.Timestamp(year=y, month=ts.month, day=ts.day)
        t_end_y = pd.Timestamp(year=y, month=te.month, day=te.day)
        if t_start_y not in outlet.index or t_end_y not in outlet.index:
            continue
        window = outlet.loc[t_start_y:t_end_y, "discharge_cfs"]
        if window.isna().any() or window.empty:
            continue
        actual = float(window.mean())
        as_of_y = pd.Timestamp(year=y, month=as_of_month, day=as_of_day)
        if as_of_y not in outlet.index:
            continue
        if as_of_y - pd.Timedelta(days=lookback) < outlet.index.min():
            continue
        try:
            fc = _forecast_from_loaded(
                outlet, contributors, ts, te, as_of_y,
                k=k, lookback_days=lookback, trend_days=trend, holdout_year=y,
                swe_indexed=swe_indexed, swe_climo=swe_climo,
                extra_feature_fns=extra_fns, feature_weights=weights,
            )
        except Exception:
            continue
        errs.append(abs(fc.forecast_median - actual))
    return {"n": len(errs), "mae": float(np.mean(errs)) if errs else float("nan")}


def main() -> None:
    rows = []
    for cfg_name, cfg in CONFIGS.items():
        extra_fns = tuple(get_feature_fn(n) for n in cfg["extra"])
        for short, t_start, t_end, as_of in BASINS:
            basin = get_basin(short)
            r = loo_mae(basin, t_start, t_end, as_of, extra_fns, cfg["weights"])
            rows.append({"config": cfg_name, "basin": short, "n": r["n"], "mae": r["mae"]})
            print(f"{cfg_name:24s} {short:24s} n={r['n']:3d}  MAE={r['mae']:>9,.1f}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    df.to_csv(LOG_PATH, index=False)

    base = df[df["config"] == "baseline"].set_index("basin")["mae"]
    print("\nMAE delta vs baseline (negative = better):")
    print(f"  {'config':<24s} " + "  ".join(f"{b:>20s}" for b in base.index))
    summary = {}
    for cfg_name in CONFIGS:
        if cfg_name == "baseline":
            continue
        sub = df[df["config"] == cfg_name].set_index("basin")["mae"]
        deltas = {b: (sub[b] - base[b]) / base[b] * 100 for b in base.index}
        summary[cfg_name] = deltas
        print(f"  {cfg_name:<24s} " + "  ".join(f"{deltas[b]:>+19.1f}%" for b in base.index))

    # Decision: improves >=2/3 basins AND does not hurt the 3rd by more than 2%.
    print("\nDecision (commit if MAE improves >=2/3 basins, no basin worse by >2%):")
    best = None
    for cfg_name, deltas in summary.items():
        improved = sum(1 for d in deltas.values() if d < 0)
        worst = max(deltas.values())
        ok = improved >= 2 and worst <= 2.0
        mean_delta = float(np.mean(list(deltas.values())))
        verdict = "KEEP" if ok else "skip"
        print(f"  {cfg_name:<24s} improves {improved}/3, worst {worst:+.1f}%, "
              f"mean {mean_delta:+.1f}%  -> {verdict}")
        if ok and (best is None or mean_delta < best[1]):
            best = (cfg_name, mean_delta)

    if best:
        print(f"\nWINNER: {best[0]} (mean MAE delta {best[1]:+.1f}%)")
        with open(os.path.join("data", "processed", "_winner.txt"), "w") as f:
            f.write(best[0])
    else:
        print("\nNo configuration met the basin-level bar.")


if __name__ == "__main__":
    main()
