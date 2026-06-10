"""Daily 7-day flow trajectory from analog years (ESP-style).

Where `analog_forecast` reduces each analog year to a single target-window
mean, this module keeps the full daily shape. We:

  1. Select the K analog years by matching the as-of state vector (the same
     machinery used by analog_forecast / validated in the feature search).
  2. For each analog year, take its ACTUAL observed daily discharge for the
     `horizon` days following the same calendar as-of date.
  3. Scale each analog trace by ratio = (this year's recent flow) /
     (that analog year's recent flow) over the lookback window, anchoring
     the historical shape to current conditions. This is exactly the trace-
     scaling CBRFC applies in operational ESP.
  4. Per forecast day, report P10/P25/P50/P75/P90 across the scaled traces.

The ensemble spread IS the predictive interval -- there is no distributional
assumption.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd

from co_river_flow_forecast.basins import Basin
from co_river_flow_forecast.modeling.analog_forecast import (
    _as_of_in_year,
    build_state_vector,
    find_analogs,
    load_basin_flow,
    resolve_active_weights,
)

DEFAULT_K = 7
DEFAULT_LOOKBACK_DAYS = 7
DEFAULT_TREND_DAYS = 14


@dataclass
class DailyTrajectory:
    basin_short: str
    as_of: pd.Timestamp
    horizon_days: int
    forecast: pd.DataFrame          # index=date, cols p10/p25/p50/p75/p90/mean
    analog_traces: pd.DataFrame     # index=lead day 1..H, cols = analog year (scaled cfs)
    analogs: pd.Series              # year -> distance
    analog_scales: dict[int, float] # year -> trace scaling factor
    current_recent_cfs: float


def analog_daily_trajectory(
    basin: Basin,
    as_of: pd.Timestamp | str | date | None = None,
    horizon_days: int = 7,
    k: int = DEFAULT_K,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    trend_days: int = DEFAULT_TREND_DAYS,
    history_start: str = "1990-10-01",
    use_swe: bool = True,
    scale_to_current: bool = True,
    use_active_features: bool = True,
) -> DailyTrajectory:
    outlet, contributors = load_basin_flow(basin, history_start=history_start)
    if outlet.empty:
        raise ValueError(f"No outlet flow for {basin.short_name}.")

    # Default as-of = most recent observation (forecast from current conditions).
    if as_of is None:
        as_of = outlet.index.max()
    else:
        as_of = pd.Timestamp(as_of)

    swe_indexed = swe_climo = None
    if use_swe:
        from co_river_flow_forecast.modeling.swe_cache import (
            load_basin_swe, precompute_swe_climatology, prepare_swe_indexed,
        )
        swe_long = load_basin_swe(basin, start=history_start)
        if not swe_long.empty:
            swe_indexed = prepare_swe_indexed(swe_long)
            swe_climo = precompute_swe_climatology(swe_indexed)

    climo_by_doy = outlet.groupby(outlet.index.dayofyear)["discharge_cfs"].median()

    def state_at(ts: pd.Timestamp) -> dict[str, float]:
        return build_state_vector(
            outlet, contributors, ts,
            lookback_days=lookback_days, trend_days=trend_days,
            climo_by_doy=climo_by_doy,
            swe_indexed=swe_indexed, swe_climo=swe_climo,
        )

    current_state = state_at(as_of)

    def recent_mean(ts: pd.Timestamp) -> float:
        w = outlet.loc[ts - pd.Timedelta(days=lookback_days - 1): ts, "discharge_cfs"]
        return float(w.mean()) if len(w) else float("nan")

    current_recent = recent_mean(as_of)

    # Build historical state table for candidate analog years that also have
    # a full daily trace over the horizon.
    years = sorted({int(t.year) for t in outlet.index})
    rows = []
    traces: dict[int, pd.Series] = {}
    analog_recent: dict[int, float] = {}
    for y in years:
        if y == as_of.year:
            continue
        as_of_y = _as_of_in_year(as_of, y)
        if as_of_y not in outlet.index:
            continue
        if as_of_y - pd.Timedelta(days=lookback_days) < outlet.index.min():
            continue
        # Need daily flow for the horizon following as_of_y.
        fut_idx = pd.date_range(as_of_y + pd.Timedelta(days=1), periods=horizon_days, freq="D")
        if not all(d in outlet.index for d in fut_idx):
            continue
        trace = outlet.loc[fut_idx, "discharge_cfs"]
        if trace.isna().any():
            continue
        st = state_at(as_of_y)
        st["_year"] = y
        rows.append(st)
        traces[y] = trace.reset_index(drop=True)  # index 0..H-1
        analog_recent[y] = recent_mean(as_of_y)

    if not rows:
        raise ValueError("No analog years with full state + horizon traces.")

    hist_df = pd.DataFrame(rows).set_index("_year")
    feature_weights = resolve_active_weights(use_active_features)
    analogs = find_analogs(current_state, hist_df, k=min(k, len(hist_df)),
                           feature_weights=feature_weights)

    # Scale each selected analog trace to current conditions.
    scales: dict[int, float] = {}
    scaled_traces: dict[int, pd.Series] = {}
    for y in analogs.index:
        if scale_to_current and analog_recent.get(y, 0) and analog_recent[y] > 0:
            scale = current_recent / analog_recent[y]
        else:
            scale = 1.0
        scales[y] = float(scale)
        scaled_traces[y] = traces[y] * scale

    trace_df = pd.DataFrame(scaled_traces)
    trace_df.index = range(1, horizon_days + 1)  # lead day 1..H
    trace_df.index.name = "lead_day"

    fc_dates = pd.date_range(as_of + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    forecast = pd.DataFrame({
        "date": fc_dates,
        "p10": trace_df.quantile(0.10, axis=1).values,
        "p25": trace_df.quantile(0.25, axis=1).values,
        "p50": trace_df.quantile(0.50, axis=1).values,
        "p75": trace_df.quantile(0.75, axis=1).values,
        "p90": trace_df.quantile(0.90, axis=1).values,
        "mean": trace_df.mean(axis=1).values,
    }).set_index("date")

    return DailyTrajectory(
        basin_short=basin.short_name,
        as_of=as_of,
        horizon_days=horizon_days,
        forecast=forecast,
        analog_traces=trace_df,
        analogs=analogs,
        analog_scales=scales,
        current_recent_cfs=current_recent,
    )
