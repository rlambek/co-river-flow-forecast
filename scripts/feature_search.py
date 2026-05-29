"""Iterate the candidate feature catalog and evaluate each across basins.

For every (basin, candidate feature) pair:
  - Run paired leave-one-year-out backtest (baseline vs baseline + feature)
  - Report per-basin win rate and MAE delta
  - Aggregate across basins
  - If the feature wins on at least 2/3 basins with win-rate >= 0.65 AND
    mean-improvement >= 0, commit it as a permanent baseline addition.
  - Otherwise log and skip.

Runs autonomously. Logs results to data/processed/feature_search_log.csv
and prints a summary table at the end.
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime

import pandas as pd

from co_river_flow_forecast.basins import GREEN_AT_GREEN_RIVER_UT, get_basin
from co_river_flow_forecast.modeling.feature_catalog import FEATURE_CATALOG
from co_river_flow_forecast.modeling.feature_eval import paired_backtest, summarize

BASINS = [
    ("yampa_steamboat", "06-01", "06-06", "05-10"),
    ("clear_creek_golden", "06-17", "06-22", "05-27"),
    ("green_green_river_ut", "06-16", "06-21", "05-25"),
]

WIN_RATE_THRESHOLD = 0.65         # how often the new feature must beat the baseline per basin
MIN_BASINS_WINNING = 2            # how many of the 3 basins must clear the threshold
LOG_DIR = os.path.join("data", "processed")
LOG_PATH = os.path.join(LOG_DIR, "feature_search_log.csv")
COMMITTED_FEATURES_PATH = os.path.join(
    "src", "co_river_flow_forecast", "modeling", "_active_features.py"
)


def _read_active_features() -> list[str]:
    """Return the list of currently committed extra features."""
    if not os.path.exists(COMMITTED_FEATURES_PATH):
        return []
    ns: dict = {}
    with open(COMMITTED_FEATURES_PATH, "r", encoding="utf-8") as f:
        exec(f.read(), ns)
    return ns.get("ACTIVE_EXTRA_FEATURES", [])


def _write_active_features(names: list[str]) -> None:
    os.makedirs(os.path.dirname(COMMITTED_FEATURES_PATH), exist_ok=True)
    body = '"""Auto-managed by scripts/feature_search.py - do not edit by hand."""\n\n'
    body += f"ACTIVE_EXTRA_FEATURES = {names!r}\n"
    with open(COMMITTED_FEATURES_PATH, "w", encoding="utf-8") as f:
        f.write(body)


def _git_commit(name: str, summary: list[dict]) -> bool:
    """Commit the active-features change. Returns True if the commit succeeded."""
    bullets = "\n".join(
        f"  {s['basin']}: win {s['win_rate']:.0%}  MAE base {s['mae_base']:,.0f} -> "
        f"new {s['mae_new']:,.0f} (delta {s['mae_delta_pct']:+.1f}%)"
        for s in summary
    )
    msg = (
        f"Add feature {name} to analog state vector\n"
        f"\nPaired leave-one-year-out evaluation across basins:\n{bullets}\n"
    )
    add = subprocess.run(
        ["git", "add", COMMITTED_FEATURES_PATH, LOG_PATH],
        capture_output=True, text=True,
    )
    if add.returncode != 0:
        print(f"git add failed: {add.stderr}")
        return False
    commit = subprocess.run(
        ["git", "commit", "-m", msg],
        capture_output=True, text=True,
    )
    if commit.returncode != 0:
        print(f"git commit failed: {commit.stderr}")
        return False
    push = subprocess.run(["git", "push", "origin", "main"], capture_output=True, text=True)
    if push.returncode != 0:
        print(f"git push failed: {push.stderr}")
        # commit landed locally even if push failed
    return True


def _log_row(row: dict) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    df = pd.DataFrame([row])
    header = not os.path.exists(LOG_PATH)
    df.to_csv(LOG_PATH, mode="a", header=header, index=False)


def main() -> None:
    print(f"Feature search started {datetime.utcnow().isoformat()}Z")
    print(f"  basins: {[b[0] for b in BASINS]}")
    print(f"  features under test ({len(FEATURE_CATALOG)}):")
    for name, _, desc in FEATURE_CATALOG:
        print(f"    {name:32s}  {desc}")
    print()

    active = _read_active_features()
    print(f"Active features at start: {active or '(none)'}")
    print()

    overview = []
    for name, fn, desc in FEATURE_CATALOG:
        print(f"--- Evaluating: {name} ---  {desc}")
        per_basin = []
        for short, t_start, t_end, as_of in BASINS:
            basin = get_basin(short)
            df = paired_backtest(
                basin, new_feature_fn=fn,
                target_start_md=t_start, target_end_md=t_end,
                as_of_md=as_of,
            )
            s = summarize(df)
            if not s["ok"]:
                print(f"  {short:24s}  n={s['n']:3d}  -- insufficient data")
                per_basin.append({"basin": short, **s})
                continue
            print(
                f"  {short:24s}  n={s['n']:3d}  win {s['win_rate']:.0%}  "
                f"MAE base {s['mae_base']:>8,.0f} -> new {s['mae_new']:>8,.0f}  "
                f"({s['mae_delta_pct']:+.1f}%)"
            )
            per_basin.append({"basin": short, **s})
            _log_row({
                "ts": datetime.utcnow().isoformat(),
                "feature": name, "basin": short,
                **{k: v for k, v in s.items() if k != "per_year"},
            })

        # Decide: keep or skip
        passing = [b for b in per_basin if b.get("ok") and b.get("win_rate", 0) >= WIN_RATE_THRESHOLD and b.get("mean_improvement_cfs", -1) > 0]
        winners = len(passing)
        keep = winners >= MIN_BASINS_WINNING

        decision = {
            "feature": name,
            "winners": winners,
            "basins_total": len(per_basin),
            "decision": "KEEP" if keep else "SKIP",
            "summary": per_basin,
        }
        overview.append(decision)

        if keep:
            print(f"  -> KEEP: {winners}/{len(per_basin)} basins meet threshold. Committing.")
            active.append(name)
            _write_active_features(active)
            _git_commit(name, per_basin)
        else:
            print(f"  -> SKIP: only {winners}/{len(per_basin)} basins meet threshold.")
        print()

    print()
    print("=== FINAL OVERVIEW ===")
    print(f"  {'feature':<32s}  {'decision':<8s}  {'winners':>8s}")
    print(f"  {'-'*32}  {'-'*8}  {'-'*8}")
    for d in overview:
        print(f"  {d['feature']:<32s}  {d['decision']:<8s}  {d['winners']}/{d['basins_total']}")
    print()
    print(f"Active features at end: {active or '(none)'}")


if __name__ == "__main__":
    main()
