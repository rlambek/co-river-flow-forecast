"""Phase 3 (weighted distance) and Phase 4 (learned GBM ensemble) search.

Same paired-backtest, same threshold (>=65% win rate on at least 2 of 3
basins). Logs results to data/processed/algorithm_search_log.csv.

Usage:
    python scripts/algorithm_search.py
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd

from co_river_flow_forecast.basins import get_basin
from co_river_flow_forecast.modeling.learned_forecast import (
    paired_gbm_backtest, summarize_gbm,
)
from co_river_flow_forecast.modeling.weighted_eval import (
    WEIGHT_SCHEMES, paired_weighted_backtest, summarize_weighted,
)

BASINS = [
    ("yampa_steamboat",       "06-01", "06-06", "05-10"),
    ("clear_creek_golden",    "06-17", "06-22", "05-27"),
    ("green_green_river_ut",  "06-16", "06-21", "05-25"),
]
LOG_PATH = os.path.join("data", "processed", "algorithm_search_log.csv")
WIN_RATE_THRESHOLD = 0.65
MIN_BASINS_WINNING = 2


def _log(row: dict) -> None:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    pd.DataFrame([row]).to_csv(
        LOG_PATH, mode="a", index=False,
        header=not os.path.exists(LOG_PATH),
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    print(f"Algorithm search started {_now()}")
    overview = []

    # Phase 3: weighted-distance schemes
    print("\n=== PHASE 3: weighted-distance K-NN ===")
    for scheme_name, scheme_weights in WEIGHT_SCHEMES.items():
        print(f"\n--- Scheme: {scheme_name} ({scheme_weights}) ---")
        per_basin = []
        for short, t_start, t_end, as_of in BASINS:
            basin = get_basin(short)
            # contributors_half: build weights from this basin's contributor labels
            weights = dict(scheme_weights)
            if scheme_name == "contributors_half":
                for label, _sid in basin.contributor_usgs_ids:
                    weights[f"{label}_recent"] = 0.5
            df = paired_weighted_backtest(
                basin, weights,
                target_start_md=t_start, target_end_md=t_end, as_of_md=as_of,
            )
            s = summarize_weighted(df)
            if not s["ok"]:
                print(f"  {short:24s}  n={s['n']:3d}  insufficient data")
                per_basin.append({"basin": short, **s})
                continue
            print(
                f"  {short:24s}  n={s['n']:3d}  win {s['win_rate']:.0%}  "
                f"MAE base {s['mae_base']:>8,.0f} -> new {s['mae_new']:>8,.0f}  "
                f"({s['mae_delta_pct']:+.1f}%)"
            )
            per_basin.append({"basin": short, **s})
            _log({"ts": _now(), "phase": "weighted", "scheme": scheme_name, "basin": short,
                  **{k: v for k, v in s.items() if k != "per_year"}})

        passing = [b for b in per_basin
                   if b.get("ok") and b.get("win_rate", 0) >= WIN_RATE_THRESHOLD
                   and b.get("mean_improvement_cfs", -1) > 0]
        keep = len(passing) >= MIN_BASINS_WINNING
        decision = {"phase": "weighted", "name": scheme_name,
                    "winners": f"{len(passing)}/{len(per_basin)}",
                    "decision": "KEEP" if keep else "SKIP"}
        overview.append(decision)
        print(f"  -> {decision['decision']}: {decision['winners']} basins pass")

    # Phase 4: GBM vs K-NN
    print("\n\n=== PHASE 4: GradientBoostingRegressor vs K-NN ===")
    per_basin = []
    for short, t_start, t_end, as_of in BASINS:
        basin = get_basin(short)
        print(f"\n--- Basin: {short} ---")
        df = paired_gbm_backtest(
            basin,
            target_start_md=t_start, target_end_md=t_end, as_of_md=as_of,
        )
        s = summarize_gbm(df)
        if not s["ok"]:
            print(f"  {short:24s}  n={s['n']:3d}  insufficient data")
            per_basin.append({"basin": short, **s})
            continue
        print(
            f"  {short:24s}  n={s['n']:3d}  win {s['win_rate']:.0%}  "
            f"KNN MAE {s['knn_mae']:>8,.0f} -> GBM {s['gbm_mae']:>8,.0f}  "
            f"({s['mae_delta_pct']:+.1f}%)"
        )
        per_basin.append({"basin": short, **s})
        _log({"ts": _now(), "phase": "gbm", "scheme": "gbm_default",
              "basin": short, **{k: v for k, v in s.items() if k != "per_year"}})

    passing = [b for b in per_basin
               if b.get("ok") and b.get("win_rate", 0) >= WIN_RATE_THRESHOLD
               and b.get("mean_improvement_cfs", -1) > 0]
    keep = len(passing) >= MIN_BASINS_WINNING
    decision = {"phase": "gbm", "name": "gbm_default",
                "winners": f"{len(passing)}/{len(per_basin)}",
                "decision": "KEEP" if keep else "SKIP"}
    overview.append(decision)
    print(f"\nGBM -> {decision['decision']}: {decision['winners']} basins pass")

    print("\n\n=== FINAL OVERVIEW (Phases 3 + 4) ===")
    print(f"  {'phase':<10s}  {'name':<22s}  {'decision':<8s}  {'winners':>8s}")
    print(f"  {'-'*10}  {'-'*22}  {'-'*8}  {'-'*8}")
    for d in overview:
        print(f"  {d['phase']:<10s}  {d['name']:<22s}  {d['decision']:<8s}  {d['winners']}")


if __name__ == "__main__":
    main()
