"""Scaled-climatology forecast of mean flow in a target date window.

For lead times beyond the NWP horizon (>~14 days), the best simple baseline
is climatology adjusted by current basin state. We compute per-day-of-year
discharge percentiles from the historical record, then scale them by the
ratio of recent flow to same-date climatology. This is an honest operational
baseline used by NRCS and CBRFC for long-lead checks.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from co_river_flow_forecast.data import fetch_daily_streamflow


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.dropna(subset=["discharge_cfs"])
    return df


def forecast_flow_window(
    site_id: str,
    target_start: str | date,
    target_end: str | date,
    climatology_start: str = "1990-10-01",
    ratio_window_days: int = 14,
    today: date | None = None,
) -> dict[str, Any]:
    """Forecast mean discharge in [target_start, target_end] for a USGS gauge.

    Strategy:
      1. Build per-day-of-year climatology (median, P10, P90) from the
         historical record, excluding the current calendar year.
      2. Compute a smoothed scaling ratio = recent flow / same-date climatology
         using the trailing `ratio_window_days` of observed data.
      3. Forecast = target-window climatology median * ratio (uncertainty
         band by scaling P10/P90 the same way).
    """
    target_start = pd.Timestamp(target_start)
    target_end = pd.Timestamp(target_end)
    if today is None:
        today_ts = pd.Timestamp(datetime.utcnow().date())
    else:
        today_ts = pd.Timestamp(today)

    flow = _normalize_index(fetch_daily_streamflow(site_id, start=climatology_start))
    if flow.empty:
        raise ValueError(f"No discharge data for USGS {site_id}.")
    flow["doy"] = flow.index.dayofyear
    flow["year"] = flow.index.year

    historical = flow[flow["year"] < today_ts.year]
    by_doy = historical.groupby("doy")["discharge_cfs"]
    climo = pd.DataFrame(
        {
            "doy": sorted(historical["doy"].unique()),
            "median": by_doy.median().values,
            "p10": by_doy.quantile(0.10).values,
            "p90": by_doy.quantile(0.90).values,
            "n_years": by_doy.count().values,
        }
    )

    target_doys = list(range(target_start.dayofyear, target_end.dayofyear + 1))
    target_climo = climo[climo["doy"].isin(target_doys)]
    climo_median_target = float(target_climo["median"].mean())
    climo_p10_target = float(target_climo["p10"].mean())
    climo_p90_target = float(target_climo["p90"].mean())
    n_years = int(target_climo["n_years"].mean())

    # Recent state - last `ratio_window_days` of observed flow vs same-date climatology.
    recent = flow[flow.index >= flow.index.max() - pd.Timedelta(days=ratio_window_days - 1)]
    recent_mean = float(recent["discharge_cfs"].mean())
    recent_doys = recent["doy"].tolist()
    same_window_climo = float(
        climo[climo["doy"].isin(recent_doys)]["median"].mean()
    )

    ratio = recent_mean / same_window_climo if same_window_climo > 0 else 1.0

    return {
        "site_id": site_id,
        "target_start": target_start.date().isoformat(),
        "target_end": target_end.date().isoformat(),
        "data_through": flow.index.max().date().isoformat(),
        "climo_median_target_cfs": climo_median_target,
        "climo_p10_target_cfs": climo_p10_target,
        "climo_p90_target_cfs": climo_p90_target,
        "n_years_climo": n_years,
        "recent_window_days": ratio_window_days,
        "recent_mean_cfs": recent_mean,
        "same_window_climo_mean_cfs": same_window_climo,
        "scaling_ratio": ratio,
        "forecast_median_cfs": climo_median_target * ratio,
        "forecast_p10_cfs": climo_p10_target * ratio,
        "forecast_p90_cfs": climo_p90_target * ratio,
    }
