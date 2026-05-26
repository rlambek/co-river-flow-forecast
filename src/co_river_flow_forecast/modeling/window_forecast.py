"""Scaled-climatology forecast of mean flow in a target date window.

For lead times beyond the NWP horizon (>~14 days), the best simple baseline
is climatology adjusted by current basin state. We compute per-day-of-year
discharge percentiles from the historical record, then scale them by the
ratio of recent flow to same-date climatology. This is an honest operational
baseline used by NRCS and CBRFC for long-lead checks.

**Caveat: the constant-ratio assumption breaks during sharp recessions.** When
the basin has just peaked and flow is dropping fast, the trailing-N-day mean
overshoots reality. Use a short ratio window (e.g. 3 days) in that case, or
compare the multi-window trend (`trends` field) to see whether the basin is
near steady state or rapidly evolving.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from co_river_flow_forecast.data import fetch_daily_streamflow

# Multi-window trends always reported so a sharp recession is visible.
DEFAULT_TREND_WINDOWS = (1, 3, 7, 14)


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.dropna(subset=["discharge_cfs"])
    return df


def _per_doy_climatology(historical: pd.DataFrame) -> pd.DataFrame:
    by_doy = historical.groupby("doy")["discharge_cfs"]
    return pd.DataFrame(
        {
            "doy": sorted(historical["doy"].unique()),
            "median": by_doy.median().values,
            "p10": by_doy.quantile(0.10).values,
            "p90": by_doy.quantile(0.90).values,
            "n_years": by_doy.count().values,
        }
    )


def _ratio_for_window(flow: pd.DataFrame, climo: pd.DataFrame, window_days: int) -> dict:
    recent = flow.tail(window_days)
    if recent.empty:
        return {"mean": np.nan, "climo": np.nan, "ratio": np.nan}
    mean = float(recent["discharge_cfs"].mean())
    doys = recent["doy"].tolist()
    climo_for_dates = float(climo[climo["doy"].isin(doys)]["median"].mean())
    ratio = mean / climo_for_dates if climo_for_dates > 0 else np.nan
    return {"mean": mean, "climo": climo_for_dates, "ratio": ratio}


def forecast_flow_window(
    site_id: str,
    target_start: str | date,
    target_end: str | date,
    climatology_start: str = "1990-10-01",
    ratio_window_days: int = 3,
    today: date | None = None,
) -> dict[str, Any]:
    """Forecast mean discharge in [target_start, target_end] for a USGS gauge.

    The scaling ratio uses the trailing `ratio_window_days` of observed flow
    by default (3 days). Use a shorter window during sharp recessions; a
    longer window during steady-state periods. The full 1/3/7/14-day trend
    is always returned in the `trends` field so the recession (or rise) is
    visible.
    """
    target_start = pd.Timestamp(target_start)
    target_end = pd.Timestamp(target_end)
    today_ts = pd.Timestamp(today or datetime.utcnow().date())

    flow = _normalize_index(fetch_daily_streamflow(site_id, start=climatology_start))
    if flow.empty:
        raise ValueError(f"No discharge data for USGS {site_id}.")
    flow["doy"] = flow.index.dayofyear
    flow["year"] = flow.index.year

    historical = flow[flow["year"] < today_ts.year]
    climo = _per_doy_climatology(historical)

    target_doys = list(range(target_start.dayofyear, target_end.dayofyear + 1))
    target_climo = climo[climo["doy"].isin(target_doys)]
    climo_median_target = float(target_climo["median"].mean())
    climo_p10_target = float(target_climo["p10"].mean())
    climo_p90_target = float(target_climo["p90"].mean())
    n_years = int(target_climo["n_years"].mean())

    trends = {f"{w}d": _ratio_for_window(flow, climo, w) for w in DEFAULT_TREND_WINDOWS}
    primary = _ratio_for_window(flow, climo, ratio_window_days)
    ratio = primary["ratio"] if not np.isnan(primary["ratio"]) else 1.0

    return {
        "site_id": site_id,
        "target_start": target_start.date().isoformat(),
        "target_end": target_end.date().isoformat(),
        "data_through": flow.index.max().date().isoformat(),
        "most_recent_cfs": float(flow["discharge_cfs"].iloc[-1]),
        "climo_median_target_cfs": climo_median_target,
        "climo_p10_target_cfs": climo_p10_target,
        "climo_p90_target_cfs": climo_p90_target,
        "n_years_climo": n_years,
        "ratio_window_days": ratio_window_days,
        "recent_mean_cfs": primary["mean"],
        "same_window_climo_mean_cfs": primary["climo"],
        "scaling_ratio": ratio,
        "forecast_median_cfs": climo_median_target * ratio,
        "forecast_p10_cfs": climo_p10_target * ratio,
        "forecast_p90_cfs": climo_p90_target * ratio,
        "trends": trends,
    }
