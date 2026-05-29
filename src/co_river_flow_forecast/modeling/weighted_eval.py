"""Leave-one-year-out comparison of analog forecasts with feature weights.

Compares baseline (uniform weights) against a candidate weight scheme using
paired per-year errors.
"""
from __future__ import annotations

import pandas as pd

from co_river_flow_forecast.basins import Basin
from co_river_flow_forecast.modeling.analog_forecast import (
    _forecast_from_loaded,
    load_basin_flow,
)
from co_river_flow_forecast.modeling.swe_cache import (
    load_basin_swe, precompute_swe_climatology, prepare_swe_indexed,
)


WEIGHT_SCHEMES: dict[str, dict[str, float]] = {
    # Each scheme is feature_name -> weight (unspecified = 1.0)
    "outlet_2x":         {"outlet_recent": 2.0, "outlet_trend": 2.0, "outlet_ratio_climo": 2.0},
    "outlet_3x":         {"outlet_recent": 3.0, "outlet_trend": 3.0, "outlet_ratio_climo": 3.0},
    "swe_2x":            {"basin_swe_recent_in": 2.0, "basin_swe_pct_of_climo": 2.0},
    "swe_3x":            {"basin_swe_recent_in": 3.0, "basin_swe_pct_of_climo": 3.0},
    "contributors_half": {  # downweight contributor recents - depends on basin contributor labels
        # Will be populated per-basin at runtime.
    },
    "downweight_n_stations": {"basin_swe_n_stations": 0.0},  # remove count from distance
}


def paired_weighted_backtest(
    basin: Basin,
    weights: dict[str, float],
    target_start_md: str,
    target_end_md: str,
    as_of_md: str,
    *,
    target_year_for_dates: int = 2026,
    k: int = 7,
    lookback_days: int = 7,
    trend_days: int = 14,
    history_start: str = "1990-10-01",
    use_swe: bool = True,
) -> pd.DataFrame:
    """For each held-out year, run baseline (uniform) and weighted forecasts.

    Returns per-year DataFrame with both predictions and absolute errors.
    """
    outlet, contributors = load_basin_flow(basin, history_start=history_start)
    swe_indexed = swe_climo = None
    if use_swe:
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

        try:
            fc_new = _forecast_from_loaded(
                outlet, contributors, ts, te, as_of_y,
                k=k, lookback_days=lookback_days, trend_days=trend_days,
                holdout_year=y,
                swe_indexed=swe_indexed, swe_climo=swe_climo,
                feature_weights=weights,
            )
        except Exception:
            continue

        eb = abs(fc_base.forecast_median - actual)
        en = abs(fc_new.forecast_median - actual)
        rows.append({
            "year": y, "actual": actual,
            "pred_base": fc_base.forecast_median, "err_base": eb,
            "pred_new": fc_new.forecast_median, "err_new": en,
            "improvement": eb - en,
            "new_better": en < eb,
        })

    return pd.DataFrame(rows)


def summarize_weighted(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"ok": False, "n": 0}
    return {
        "ok": True,
        "n": int(len(df)),
        "win_rate": float(df["new_better"].mean()),
        "mean_improvement_cfs": float(df["improvement"].mean()),
        "mae_base": float(df["err_base"].mean()),
        "mae_new": float(df["err_new"].mean()),
        "mae_delta_pct": float(
            (df["err_new"].mean() - df["err_base"].mean()) / df["err_base"].mean() * 100
        ),
    }
