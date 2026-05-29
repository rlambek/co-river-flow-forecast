"""Learned-ensemble forecasting alternative to K-NN analog matching.

Same state vector as `analog_forecast`, but instead of finding K analogs
and taking their median, train a GradientBoostingRegressor on the
(historical state vector, observed target-window mean) pairs and predict
for the current state.

Designed to be compared head-to-head with `forecast_with_analogs` via
`paired_gbm_backtest` below.
"""
from __future__ import annotations

from datetime import datetime, date
from typing import Iterable

import numpy as np
import pandas as pd

from co_river_flow_forecast.basins import Basin
from co_river_flow_forecast.modeling.analog_forecast import (
    _forecast_from_loaded,
    build_state_vector,
    load_basin_flow,
)
from co_river_flow_forecast.modeling.swe_cache import (
    load_basin_swe, precompute_swe_climatology, prepare_swe_indexed,
)


def _build_historical_table(
    outlet: pd.DataFrame,
    contributors: dict[str, pd.DataFrame],
    target_start: pd.Timestamp,
    target_end: pd.Timestamp,
    as_of_month: int,
    as_of_day: int,
    lookback_days: int,
    trend_days: int,
    swe_indexed: pd.DataFrame | None,
    swe_climo: pd.DataFrame | None,
    extra_feature_fns: tuple,
    exclude_years: Iterable[int] = (),
) -> tuple[pd.DataFrame, pd.Series]:
    """Build per-water-year feature matrix X and target vector y."""
    climo_by_doy = outlet.groupby(outlet.index.dayofyear)["discharge_cfs"].median()
    years = sorted({int(t.year) for t in outlet.index})
    exclude = set(exclude_years)

    rows: list[dict] = []
    for y in years:
        if y in exclude:
            continue
        as_of_y = pd.Timestamp(year=y, month=as_of_month, day=as_of_day)
        if as_of_y not in outlet.index:
            continue
        if as_of_y - pd.Timedelta(days=lookback_days) < outlet.index.min():
            continue
        t_start_y = pd.Timestamp(year=y, month=target_start.month, day=target_start.day)
        t_end_y = pd.Timestamp(year=y, month=target_end.month, day=target_end.day)
        if t_start_y not in outlet.index or t_end_y not in outlet.index:
            continue
        window = outlet.loc[t_start_y:t_end_y, "discharge_cfs"]
        if window.isna().any() or window.empty:
            continue
        try:
            state = build_state_vector(
                outlet, contributors, as_of_y,
                lookback_days=lookback_days, trend_days=trend_days,
                climo_by_doy=climo_by_doy,
                swe_indexed=swe_indexed, swe_climo=swe_climo,
                extra_feature_fns=extra_feature_fns,
            )
        except Exception:
            continue
        state["_year"] = y
        state["_target"] = float(window.mean())
        rows.append(state)

    if not rows:
        return pd.DataFrame(), pd.Series(dtype=float)

    df = pd.DataFrame(rows).set_index("_year")
    y_target = df.pop("_target")
    return df.dropna(), y_target.loc[df.dropna().index]


def gbm_forecast_for_year(
    basin: Basin,
    target_start: str,
    target_end: str,
    as_of_month: int,
    as_of_day: int,
    target_year: int,
    *,
    lookback_days: int = 7,
    trend_days: int = 14,
    history_start: str = "1990-10-01",
    use_swe: bool = True,
    extra_feature_fns: tuple = (),
    n_estimators: int = 200,
    max_depth: int = 3,
    learning_rate: float = 0.05,
    random_state: int = 0,
) -> dict:
    """Train GBM on all years EXCEPT `target_year`, predict for `target_year`."""
    from sklearn.ensemble import GradientBoostingRegressor

    outlet, contributors = load_basin_flow(basin, history_start=history_start)
    swe_indexed = swe_climo = None
    if use_swe:
        swe_long = load_basin_swe(basin, start=history_start)
        if not swe_long.empty:
            swe_indexed = prepare_swe_indexed(swe_long)
            swe_climo = precompute_swe_climatology(swe_indexed)

    ts = pd.Timestamp(f"{target_year}-{target_start}")
    te = pd.Timestamp(f"{target_year}-{target_end}")

    X, y = _build_historical_table(
        outlet, contributors, ts, te,
        as_of_month, as_of_day, lookback_days, trend_days,
        swe_indexed, swe_climo, extra_feature_fns,
        exclude_years=[target_year],
    )
    if X.empty:
        return {"ok": False, "reason": "no training data"}

    # Build prediction vector for target_year.
    climo_by_doy = outlet.groupby(outlet.index.dayofyear)["discharge_cfs"].median()
    as_of_t = pd.Timestamp(year=target_year, month=as_of_month, day=as_of_day)
    pred_state = build_state_vector(
        outlet, contributors, as_of_t,
        lookback_days=lookback_days, trend_days=trend_days,
        climo_by_doy=climo_by_doy,
        swe_indexed=swe_indexed, swe_climo=swe_climo,
        extra_feature_fns=extra_feature_fns,
    )
    pred_X = pd.DataFrame([pred_state])[X.columns].fillna(X.mean())

    model = GradientBoostingRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, random_state=random_state,
    )
    model.fit(X.values, y.values)
    pred = float(model.predict(pred_X.values)[0])
    return {
        "ok": True, "n_train": int(len(X)),
        "prediction": pred,
        "feature_columns": list(X.columns),
    }


def paired_gbm_backtest(
    basin: Basin,
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
    """Leave-one-year-out comparison: K-NN vs GBM, same state vector."""
    from sklearn.ensemble import GradientBoostingRegressor

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

        # KNN
        try:
            fc_knn = _forecast_from_loaded(
                outlet, contributors, ts, te, as_of_y,
                k=k, lookback_days=lookback_days, trend_days=trend_days,
                holdout_year=y,
                swe_indexed=swe_indexed, swe_climo=swe_climo,
            )
            knn_pred = fc_knn.forecast_median
        except Exception:
            continue

        # GBM
        try:
            X, y_target = _build_historical_table(
                outlet, contributors, ts, te,
                as_of_month, as_of_day, lookback_days, trend_days,
                swe_indexed, swe_climo, (),
                exclude_years=[y],
            )
            if X.empty:
                continue
            climo_by_doy = outlet.groupby(outlet.index.dayofyear)["discharge_cfs"].median()
            pred_state = build_state_vector(
                outlet, contributors, as_of_y,
                lookback_days=lookback_days, trend_days=trend_days,
                climo_by_doy=climo_by_doy,
                swe_indexed=swe_indexed, swe_climo=swe_climo,
            )
            pred_X = pd.DataFrame([pred_state])[X.columns].fillna(X.mean())
            model = GradientBoostingRegressor(
                n_estimators=200, max_depth=3,
                learning_rate=0.05, random_state=0,
            )
            model.fit(X.values, y_target.values)
            gbm_pred = float(model.predict(pred_X.values)[0])
        except Exception:
            continue

        rows.append({
            "year": y, "actual": actual,
            "knn_pred": knn_pred, "knn_err": abs(knn_pred - actual),
            "gbm_pred": gbm_pred, "gbm_err": abs(gbm_pred - actual),
            "improvement": abs(knn_pred - actual) - abs(gbm_pred - actual),
            "gbm_better": abs(gbm_pred - actual) < abs(knn_pred - actual),
        })

    return pd.DataFrame(rows)


def summarize_gbm(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"ok": False, "n": 0}
    return {
        "ok": True,
        "n": int(len(df)),
        "win_rate": float(df["gbm_better"].mean()),
        "mean_improvement_cfs": float(df["improvement"].mean()),
        "knn_mae": float(df["knn_err"].mean()),
        "gbm_mae": float(df["gbm_err"].mean()),
        "mae_delta_pct": float(
            (df["gbm_err"].mean() - df["knn_err"].mean()) / df["knn_err"].mean() * 100
        ),
    }
