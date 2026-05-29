"""Summarize the autonomous feature search log into a readable overview."""
from __future__ import annotations

import os

import pandas as pd

LOG_PATH = os.path.join("data", "processed", "feature_search_log.csv")
WIN_RATE_THRESHOLD = 0.65


def main() -> None:
    if not os.path.exists(LOG_PATH):
        raise SystemExit(f"No log found at {LOG_PATH}")
    df = pd.read_csv(LOG_PATH)
    # Latest result per (feature, basin) pair.
    df = df.sort_values("ts").drop_duplicates(["feature", "basin"], keep="last")

    pivot_win = df.pivot_table(index="feature", columns="basin", values="win_rate", aggfunc="last")
    pivot_mae = df.pivot_table(index="feature", columns="basin", values="mae_delta_pct", aggfunc="last")

    print("Per-feature, per-basin win rate (fraction of years where the new feature beat baseline):")
    print()
    print(pivot_win.applymap(lambda x: f"{x:.0%}" if pd.notna(x) else "").to_string())
    print()
    print("Per-feature, per-basin MAE delta (% change in mean absolute error, lower is better):")
    print()
    print(pivot_mae.applymap(lambda x: f"{x:+.1f}%" if pd.notna(x) else "").to_string())
    print()

    # Decision per feature
    decisions = []
    for feat, sub in df.groupby("feature"):
        winners = sub[(sub["win_rate"] >= WIN_RATE_THRESHOLD) & (sub["mean_improvement_cfs"] > 0)]
        n_winners = len(winners)
        n_basins = sub["basin"].nunique()
        keep = n_winners >= 2
        avg_win_rate = sub["win_rate"].mean()
        avg_mae_delta = sub["mae_delta_pct"].mean()
        decisions.append({
            "feature": feat,
            "decision": "KEEP" if keep else "SKIP",
            "winners": f"{n_winners}/{n_basins}",
            "avg_win_rate": f"{avg_win_rate:.0%}",
            "avg_mae_delta_pct": f"{avg_mae_delta:+.1f}%",
        })

    print("Decisions:")
    print()
    dec_df = pd.DataFrame(decisions).set_index("feature")
    print(dec_df.to_string())


if __name__ == "__main__":
    main()
