"""Paired leave-one-year-out evaluation for candidate features.

For each held-out water year y, run two analog forecasts -- baseline and
baseline+new-feature -- and compare their absolute errors against the
observed actual flow. Report per-year results and aggregate stats.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

from co_river_flow_forecast.basins import Basin
from co_river_flow_forecast.modeling.analog_forecast import (
    _forecast_from_loaded,
    load_basin_flow,
)
from co_river_flow_forecast.modeling.swe_cache import (
    load_basin_swe,
    precompute_swe_climatology,
    prepare_swe_indexed,
)


def paired_backtest(
    basin: Basin,
    new_feature_fn: Callable | None,
    target_start_md: str,        # "MM-DD"
    target_end_md: str,          # "MM-DD"
    as_of_md: str,               # "MM-DD"
    target_year_for_dates: int = 2026,
    k: int = 7,
    lookback_days: int = 7,
    trend_days: int = 14,
    history_start: str = "1990-10-01",
    use_swe_baseline: bool = True,
) -> pd.DataFrame:
    """Run paired backtest. Returns per-year DataFrame.

    Pass `new_feature_fn=None` to run the baseline only.
    """
    outlet, contributors = load_basin_flow(basin, history_start=history_start)

    swe_indexed = None
    swe_climo = None
    if use_swe_baseline:
        swe_long = load_basin_swe(basin, start=history_start)
        if not swe_long.empty:
            swe_indexed = prepare_swe_indexed(swe_long)
            swe_climo = precompute_swe_climatology(swe_indexed)

    ts = pd.Timestamp(f"{target_year_for_dates}-{target_start_md}")
    te = pd.Timestamp(f"{target_year_for_dates}-{target_end_md}")
    as_of_month, as_of_day = (int(x) for x in as_of_md.split("-"))

    years = sorted({int(t.year) for t in outlet.index})

    rows = []
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
        if as_of_y - pd.Timedelta(days=lookback_days) < outlet.index.min():
            continue

        try:
            fc_base = _forecast_from_loaded(
                outlet, contributors, ts, te, as_of_y,
                k=k, lookback_days=lookback_days, trend_days=trend_days,
                holdout_year=y,
                swe_indexed=swe_indexed, swe_climo=swe_climo,
            )
        except Exception:
            continue

        if new_feature_fn is None:
            rows.append({
                "year": y, "actual": actual,
                "pred_base": fc_base.forecast_median,
                "err_base": abs(fc_base.forecast_median - actual),
            })
            continue

        try:
            fc_new = _forecast_from_loaded(
                outlet, contributors, ts, te, as_of_y,
                k=k, lookback_days=lookback_days, trend_days=trend_days,
                holdout_year=y,
                swe_indexed=swe_indexed, swe_climo=swe_climo,
                extra_feature_fns=(new_feature_fn,),
            )
        except Exception:
            continue

        err_base = abs(fc_base.forecast_median - actual)
        err_new = abs(fc_new.forecast_median - actual)
        rows.append({
            "year": y, "actual": actual,
            "pred_base": fc_base.forecast_median,
            "err_base": err_base,
            "pred_new": fc_new.forecast_median,
            "err_new": err_new,
            "improvement": err_base - err_new,
            "new_better": err_new < err_base,
            "new_p10": fc_new.forecast_p10,
            "new_p90": fc_new.forecast_p90,
        })

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    """Aggregate stats from paired_backtest output."""
    if df.empty or "err_new" not in df.columns:
        return {"n": int(len(df)), "ok": False}
    return {
        "n": int(len(df)),
        "win_rate": float(df["new_better"].mean()),
        "mean_improvement_cfs": float(df["improvement"].mean()),
        "mae_base": float(df["err_base"].mean()),
        "mae_new": float(df["err_new"].mean()),
        "mae_delta_pct": float(
            (df["err_new"].mean() - df["err_base"].mean()) / df["err_base"].mean() * 100
        ),
        "ok": True,
    }
