"""Analog-year forecasting for a target date window.

Find historical water years whose "state vector" (recent flow at the outlet
and at each registered contributor, plus simple trend/ratio terms) most
closely matches this year's state vector, evaluated AS OF a fixed
issue-date day-of-year. Forecast the target window using the actual flow
those analog years recorded during that same calendar window.

This is the same family of methods CBRFC uses for ESP (Ensemble Streamflow
Prediction). It is honest about uncertainty by construction (the spread
across analogs IS the predictive interval), and it correctly handles
years that have already peaked vs years still rising, because both are
in the historical pool.

This module exposes:
  - build_state_vector(...)        feature vector for a (site, as_of_date)
  - find_analogs(...)              K-NN over historical years
  - forecast_with_analogs(...)     produce a forecast for one basin
  - backtest(...)                  leave-one-year-out evaluation
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Iterable

import numpy as np
import pandas as pd

from co_river_flow_forecast.basins import Basin
from co_river_flow_forecast.data import fetch_daily_streamflow

# Reasonable defaults; the backtest script sweeps these.
DEFAULT_K = 5
DEFAULT_LOOKBACK_DAYS = 7
DEFAULT_TREND_DAYS = 14


def _wy_doy_of(ts: pd.Timestamp) -> int:
    """Day of *water* year (Oct 1 = 1) for a tz-naive Timestamp."""
    wy = ts.year if ts.month < 10 else ts.year + 1
    return (ts.normalize() - pd.Timestamp(year=wy - 1, month=10, day=1)).days + 1


def _calendar_doy_of(ts: pd.Timestamp) -> int:
    return ts.dayofyear


def _as_of_in_year(as_of: pd.Timestamp, year: int) -> pd.Timestamp:
    """Same month/day as `as_of`, in calendar year `year`."""
    return pd.Timestamp(year=year, month=as_of.month, day=as_of.day)


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    return out.dropna(subset=["discharge_cfs"])


def build_state_vector(
    site_flow: pd.DataFrame,
    contributor_flows: dict[str, pd.DataFrame],
    as_of: pd.Timestamp,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    trend_days: int = DEFAULT_TREND_DAYS,
    climo_by_doy: pd.Series | None = None,
    swe_long: pd.DataFrame | None = None,
    swe_indexed: pd.DataFrame | None = None,
    swe_climo: pd.DataFrame | None = None,
    extra_feature_fns: tuple = (),
) -> dict[str, float]:
    """Compute the feature vector summarizing recent state at `as_of`.

    Features (all in cfs unless otherwise noted):
      outlet_recent           mean discharge over [as_of - lookback, as_of]
      outlet_trend            (recent half) - (early half) of `trend_days` window
      outlet_ratio_climo      outlet_recent / same-period climatology
      <label>_recent          mean discharge for each contributor (same window)

    When `swe_long` is provided (long-format daily SWE across basin SNOTELs):
      basin_swe_recent_in     mean SWE across stations, last `lookback_days`
      basin_swe_pct_of_climo  same, as % of same-DOY median SWE
    """
    state: dict[str, float] = {}
    window = site_flow.loc[as_of - pd.Timedelta(days=lookback_days - 1) : as_of, "discharge_cfs"]
    state["outlet_recent"] = float(window.mean()) if len(window) else float("nan")

    trend_window = site_flow.loc[as_of - pd.Timedelta(days=trend_days - 1) : as_of, "discharge_cfs"]
    if len(trend_window) >= 4:
        half = len(trend_window) // 2
        state["outlet_trend"] = float(trend_window.iloc[half:].mean() - trend_window.iloc[:half].mean())
    else:
        state["outlet_trend"] = float("nan")

    if climo_by_doy is not None:
        doys = window.index.dayofyear
        climo_window_mean = float(climo_by_doy.loc[doys].mean())
        state["outlet_ratio_climo"] = state["outlet_recent"] / climo_window_mean if climo_window_mean > 0 else float("nan")
    else:
        state["outlet_ratio_climo"] = float("nan")

    for label, trib in contributor_flows.items():
        w = trib.loc[as_of - pd.Timedelta(days=lookback_days - 1) : as_of, "discharge_cfs"]
        state[f"{label}_recent"] = float(w.mean()) if len(w) else float("nan")

    if swe_long is not None or swe_indexed is not None:
        # Lazy import to keep modeling/swe_cache out of the hot path for
        # streamflow-only state vectors.
        from co_river_flow_forecast.modeling.swe_cache import basin_swe_features
        state.update(basin_swe_features(
            swe_long if swe_long is not None else pd.DataFrame(),
            as_of,
            lookback_days=lookback_days,
            swe_indexed=swe_indexed,
            climo=swe_climo,
        ))

    for fn in extra_feature_fns:
        state.update(fn(site_flow, contributor_flows, swe_indexed, swe_climo,
                        as_of, lookback_days))

    return state


def find_analogs(
    current_state: dict[str, float],
    historical_states: pd.DataFrame,
    k: int,
    feature_weights: dict[str, float] | None = None,
) -> pd.Series:
    """K-NN over historical years using z-scored, optionally weighted Euclidean.

    `historical_states` is a DataFrame indexed by water_year with the same
    feature columns produced by `build_state_vector`.

    `feature_weights` is an optional dict mapping feature name to a relative
    weight. Default weight = 1.0 for unspecified features. The squared
    differences are multiplied by the weight before summing.
    """
    features = list(current_state.keys())
    historical = historical_states[features].dropna()
    if historical.empty:
        raise ValueError("No historical states with full feature vectors.")

    mean = historical.mean()
    std = historical.std().replace(0.0, 1.0)
    historical_z = (historical - mean) / std
    current_z = (pd.Series(current_state, index=features) - mean) / std

    sq_diff = (historical_z - current_z) ** 2
    if feature_weights:
        weights = pd.Series(
            {f: feature_weights.get(f, 1.0) for f in features},
            index=features,
        )
        sq_diff = sq_diff * weights

    dists = np.sqrt(sq_diff.sum(axis=1))
    return dists.nsmallest(k)


@dataclass
class AnalogForecast:
    as_of: pd.Timestamp
    target_start: pd.Timestamp
    target_end: pd.Timestamp
    analogs: pd.Series                       # year -> distance
    analog_window_flows: dict[int, float]    # year -> mean flow over target window
    forecast_median: float
    forecast_p10: float
    forecast_p25: float
    forecast_p75: float
    forecast_p90: float
    current_state: dict[str, float]
    historical_states_used: int


def load_basin_flow(basin: Basin, history_start: str = "1990-10-01") -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Fetch outlet + contributor streamflow once. Reuse across many forecasts."""
    outlet = _normalize_index(fetch_daily_streamflow(basin.usgs_id, start=history_start))
    contributors = {
        label: _normalize_index(fetch_daily_streamflow(sid, start=history_start))
        for label, sid in basin.contributor_usgs_ids
    }
    return outlet, contributors


def _forecast_from_loaded(
    outlet: pd.DataFrame,
    contributors: dict[str, pd.DataFrame],
    target_start: pd.Timestamp,
    target_end: pd.Timestamp,
    as_of: pd.Timestamp,
    k: int,
    lookback_days: int,
    trend_days: int,
    holdout_year: int | None,
    swe_long: pd.DataFrame | None = None,
    swe_indexed: pd.DataFrame | None = None,
    swe_climo: pd.DataFrame | None = None,
    extra_feature_fns: tuple = (),
    feature_weights: dict[str, float] | None = None,
) -> AnalogForecast:
    """Analog forecast using pre-loaded outlet + contributor DataFrames.

    Pass `swe_long` (long-format SNOTEL SWE) to include basin SWE features.
    For performance in loops, prefer passing pre-built `swe_indexed` and
    `swe_climo` (see `swe_cache.prepare_swe_indexed` and
    `swe_cache.precompute_swe_climatology`).
    """
    climo_by_doy = outlet.groupby(outlet.index.dayofyear)["discharge_cfs"].median()

    # Build SWE supports once per call if caller did not supply them.
    if swe_long is not None and not swe_long.empty and swe_indexed is None:
        from co_river_flow_forecast.modeling.swe_cache import (
            precompute_swe_climatology, prepare_swe_indexed,
        )
        swe_indexed = prepare_swe_indexed(swe_long)
        swe_climo = precompute_swe_climatology(swe_indexed)

    current_state = build_state_vector(
        outlet, contributors, as_of,
        lookback_days=lookback_days, trend_days=trend_days,
        climo_by_doy=climo_by_doy,
        swe_indexed=swe_indexed,
        swe_climo=swe_climo,
        extra_feature_fns=extra_feature_fns,
    )

    # Build state vector for each historical year, AS OF the same month/day.
    years = sorted({int(ts.year) for ts in outlet.index})
    historical_rows = []
    historical_targets = {}
    for y in years:
        if y == as_of.year or y == holdout_year:
            continue
        as_of_y = _as_of_in_year(as_of, y)
        if as_of_y not in outlet.index:
            continue
        # State vector requires lookback_days of prior data.
        early = as_of_y - pd.Timedelta(days=lookback_days)
        if early < outlet.index.min():
            continue
        state_y = build_state_vector(
            outlet, contributors, as_of_y,
            lookback_days=lookback_days, trend_days=trend_days,
            climo_by_doy=climo_by_doy,
            swe_indexed=swe_indexed,
            swe_climo=swe_climo,
            extra_feature_fns=extra_feature_fns,
        )

        # Target window for year y, calendar-aligned.
        t_start_y = pd.Timestamp(year=y, month=target_start.month, day=target_start.day)
        t_end_y = pd.Timestamp(year=y, month=target_end.month, day=target_end.day)
        if t_end_y not in outlet.index or t_start_y not in outlet.index:
            continue
        window = outlet.loc[t_start_y:t_end_y, "discharge_cfs"]
        if window.isna().any() or window.empty:
            continue
        state_y["_year"] = y
        state_y["_target_mean"] = float(window.mean())
        historical_rows.append(state_y)
        historical_targets[y] = float(window.mean())

    if not historical_rows:
        raise ValueError("No historical years with complete state vectors and target windows.")

    df = pd.DataFrame(historical_rows).set_index("_year")
    target_means = df.pop("_target_mean")

    analogs = find_analogs(current_state, df, k=k, feature_weights=feature_weights)
    analog_targets = target_means.loc[analogs.index]
    return AnalogForecast(
        as_of=as_of,
        target_start=target_start,
        target_end=target_end,
        analogs=analogs,
        analog_window_flows=analog_targets.to_dict(),
        forecast_median=float(analog_targets.median()),
        forecast_p10=float(analog_targets.quantile(0.10)),
        forecast_p25=float(analog_targets.quantile(0.25)),
        forecast_p75=float(analog_targets.quantile(0.75)),
        forecast_p90=float(analog_targets.quantile(0.90)),
        current_state=current_state,
        historical_states_used=len(df),
    )


def _resolve_active_extra_features(use_active_features: bool) -> tuple:
    """Look up the list of auto-committed extra features and resolve to callables."""
    if not use_active_features:
        return ()
    try:
        from co_river_flow_forecast.modeling._active_features import ACTIVE_EXTRA_FEATURES
        from co_river_flow_forecast.modeling.feature_catalog import get_feature_fn
    except Exception:
        return ()
    return tuple(get_feature_fn(name) for name in ACTIVE_EXTRA_FEATURES)


def forecast_with_analogs(
    basin: Basin,
    target_start: pd.Timestamp | str | date,
    target_end: pd.Timestamp | str | date,
    as_of: pd.Timestamp | str | date | None = None,
    k: int = DEFAULT_K,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    trend_days: int = DEFAULT_TREND_DAYS,
    history_start: str = "1990-10-01",
    holdout_year: int | None = None,
    use_swe: bool = False,
    use_active_features: bool = True,
) -> AnalogForecast:
    """Convenience wrapper: fetch data then forecast.

    By default includes whatever extra features have been auto-committed
    by `scripts/feature_search.py`. Pass `use_active_features=False` to
    fall back to the baseline state vector only.
    """
    outlet, contributors = load_basin_flow(basin, history_start=history_start)
    swe_indexed = None
    swe_climo = None
    if use_swe:
        from co_river_flow_forecast.modeling.swe_cache import (
            load_basin_swe, precompute_swe_climatology, prepare_swe_indexed,
        )
        swe_long = load_basin_swe(basin, start=history_start)
        swe_indexed = prepare_swe_indexed(swe_long)
        swe_climo = precompute_swe_climatology(swe_indexed)
    extra_feature_fns = _resolve_active_extra_features(use_active_features)
    target_start = pd.Timestamp(target_start)
    target_end = pd.Timestamp(target_end)
    as_of_ts = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(datetime.utcnow().date())
    return _forecast_from_loaded(
        outlet, contributors, target_start, target_end, as_of_ts,
        k=k, lookback_days=lookback_days, trend_days=trend_days,
        holdout_year=holdout_year,
        swe_indexed=swe_indexed,
        swe_climo=swe_climo,
        extra_feature_fns=extra_feature_fns,
    )


def backtest(
    basin: Basin,
    target_start: pd.Timestamp | str | date,
    target_end: pd.Timestamp | str | date,
    as_of_month: int,
    as_of_day: int,
    k: int = DEFAULT_K,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    trend_days: int = DEFAULT_TREND_DAYS,
    history_start: str = "1990-10-01",
    use_swe: bool = False,
) -> pd.DataFrame:
    """Leave-one-year-out backtest of the analog forecaster.

    For each historical year y, hold y out of the analog pool and forecast
    the [target_start, target_end] window for y using y's state as of
    (as_of_month, as_of_day). Compare predicted vs actual.

    Returns a DataFrame indexed by water year with columns including
    predicted_median, actual, abs_err, pct_err, and the climatology baseline
    for reference.
    """
    target_start = pd.Timestamp(target_start)
    target_end = pd.Timestamp(target_end)

    # Fetch USGS data ONCE; reuse across every analog-forecast invocation
    # below. The previous implementation refetched per iteration and was
    # unusable in practice.
    outlet, contributors = load_basin_flow(basin, history_start=history_start)
    swe_long: pd.DataFrame | None = None
    swe_indexed: pd.DataFrame | None = None
    swe_climo: pd.DataFrame | None = None
    if use_swe:
        from co_river_flow_forecast.modeling.swe_cache import (
            load_basin_swe, precompute_swe_climatology, prepare_swe_indexed,
        )
        swe_long = load_basin_swe(basin, start=history_start)
        swe_indexed = prepare_swe_indexed(swe_long)
        swe_climo = precompute_swe_climatology(swe_indexed)
    years = sorted({int(ts.year) for ts in outlet.index})

    actuals: dict[int, float] = {}
    for y in years:
        t_start_y = pd.Timestamp(year=y, month=target_start.month, day=target_start.day)
        t_end_y = pd.Timestamp(year=y, month=target_end.month, day=target_end.day)
        if t_start_y in outlet.index and t_end_y in outlet.index:
            window = outlet.loc[t_start_y:t_end_y, "discharge_cfs"]
            if not window.isna().any() and not window.empty:
                actuals[y] = float(window.mean())
    climo_target = float(np.median(list(actuals.values()))) if actuals else float("nan")

    rows = []
    for y, actual in actuals.items():
        as_of_y = pd.Timestamp(year=y, month=as_of_month, day=as_of_day)
        try:
            fc = _forecast_from_loaded(
                outlet, contributors,
                target_start=target_start, target_end=target_end,
                as_of=as_of_y, k=k,
                lookback_days=lookback_days, trend_days=trend_days,
                holdout_year=y,
                swe_indexed=swe_indexed,
                swe_climo=swe_climo,
            )
        except Exception:
            continue
        rows.append({
            "water_year": y,
            "actual": actual,
            "predicted_median": fc.forecast_median,
            "predicted_p10": fc.forecast_p10,
            "predicted_p90": fc.forecast_p90,
            "climo_baseline": climo_target,
            "n_historical": fc.historical_states_used,
        })

    df = pd.DataFrame(rows).set_index("water_year").sort_index()
    df["analog_abs_err"] = (df["predicted_median"] - df["actual"]).abs()
    df["climo_abs_err"] = (df["climo_baseline"] - df["actual"]).abs()
    df["analog_pct_err"] = df["analog_abs_err"] / df["actual"] * 100
    df["climo_pct_err"] = df["climo_abs_err"] / df["actual"] * 100
    df["in_p10_p90"] = ((df["actual"] >= df["predicted_p10"]) & (df["actual"] <= df["predicted_p90"])).astype(int)
    return df


def backtest_summary(df: pd.DataFrame) -> dict[str, float]:
    return {
        "n": int(len(df)),
        "analog_mae": float(df["analog_abs_err"].mean()),
        "analog_mape": float(df["analog_pct_err"].mean()),
        "climo_mae": float(df["climo_abs_err"].mean()),
        "climo_mape": float(df["climo_pct_err"].mean()),
        "skill_vs_climo_mae": 1.0 - float(df["analog_abs_err"].mean()) / float(df["climo_abs_err"].mean()),
        "p10_p90_coverage": float(df["in_p10_p90"].mean()),
    }
