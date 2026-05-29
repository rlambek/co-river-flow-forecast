"""Combined overview of the feature search (14 candidates) and the
algorithm search (Phase 3 weighted distance + Phase 4 GBM).
"""
from __future__ import annotations

import os

import pandas as pd

FEATURE_LOG = os.path.join("data", "processed", "feature_search_log.csv")
ALG_LOG = os.path.join("data", "processed", "algorithm_search_log.csv")
WIN_RATE_THRESHOLD = 0.65


def _section_features() -> None:
    if not os.path.exists(FEATURE_LOG):
        print(f"(no {FEATURE_LOG})")
        return
    df = pd.read_csv(FEATURE_LOG).sort_values("ts").drop_duplicates(
        ["feature", "basin"], keep="last"
    )
    win = df.pivot_table(index="feature", columns="basin", values="win_rate", aggfunc="last")
    mae = df.pivot_table(index="feature", columns="basin", values="mae_delta_pct", aggfunc="last")
    n_basins = df.groupby("feature")["basin"].nunique()

    decisions = []
    for feat, sub in df.groupby("feature"):
        winners = sub[
            (sub["win_rate"] >= WIN_RATE_THRESHOLD)
            & (sub["mean_improvement_cfs"] > 0)
        ]
        decisions.append({
            "feature": feat,
            "decision": "KEEP" if len(winners) >= 2 else "SKIP",
            "winners": f"{len(winners)}/{n_basins.loc[feat]}",
            "best_mae_delta": f"{sub['mae_delta_pct'].min():+.1f}%",
            "best_basin": sub.loc[sub["mae_delta_pct"].idxmin(), "basin"],
        })

    dec_df = pd.DataFrame(decisions).set_index("feature")
    print("Win-rate matrix (fraction of years where the new feature beat baseline):")
    print(win.map(lambda x: f"{x:.0%}" if pd.notna(x) else "").to_string())
    print()
    print("MAE delta matrix (% change in mean absolute error, negative = better):")
    print(mae.map(lambda x: f"{x:+.1f}%" if pd.notna(x) else "").to_string())
    print()
    print("Per-feature decision:")
    print(dec_df.to_string())


def _section_algorithm() -> None:
    if not os.path.exists(ALG_LOG):
        print(f"(no {ALG_LOG})")
        return
    df = pd.read_csv(ALG_LOG)
    df = df.sort_values("ts")

    weighted = df[df["phase"] == "weighted"].copy()
    if not weighted.empty:
        print("Phase 3 - weighted-distance schemes:")
        print()
        w_win = weighted.pivot_table(index="scheme", columns="basin", values="win_rate", aggfunc="last")
        w_mae = weighted.pivot_table(index="scheme", columns="basin", values="mae_delta_pct", aggfunc="last")
        print("Win rate:")
        print(w_win.map(lambda x: f"{x:.0%}" if pd.notna(x) else "").to_string())
        print()
        print("MAE delta:")
        print(w_mae.map(lambda x: f"{x:+.1f}%" if pd.notna(x) else "").to_string())
        print()

    gbm = df[df["phase"] == "gbm"].copy()
    if not gbm.empty:
        print("Phase 4 - GradientBoostingRegressor vs K-NN:")
        print()
        g_win = gbm.pivot_table(index="scheme", columns="basin", values="win_rate", aggfunc="last")
        g_mae = gbm.pivot_table(index="scheme", columns="basin", values="mae_delta_pct", aggfunc="last")
        print("Win rate:")
        print(g_win.map(lambda x: f"{x:.0%}" if pd.notna(x) else "").to_string())
        print()
        print("MAE delta:")
        print(g_mae.map(lambda x: f"{x:+.1f}%" if pd.notna(x) else "").to_string())
        print()


def main() -> None:
    print("=" * 72)
    print("FEATURE SEARCH: 14 candidate state-vector additions")
    print("=" * 72)
    _section_features()
    print()
    print("=" * 72)
    print("ALGORITHM SEARCH: weighted distance + learned ensemble")
    print("=" * 72)
    _section_algorithm()


if __name__ == "__main__":
    main()
