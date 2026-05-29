"""Candidate features to add to the analog-forecast state vector.

Each entry is (name, function, description). A function takes
    (outlet_df, contributors_dict, swe_indexed, swe_climo, as_of, lookback_days)
and returns a `dict[str, float]` to be merged into the state vector. Missing
values are returned as NaN; the analog matcher drops rows with any NaN
before computing distances.

The function signature is intentionally identical to the inputs the existing
`_forecast_from_loaded` already has, so adding a feature is "just" appending
it to FEATURE_CATALOG and re-running the eval harness.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

FeatureFn = Callable[..., dict[str, float]]


def _wy_start_for(as_of: pd.Timestamp) -> pd.Timestamp:
    wy = as_of.year if as_of.month < 10 else as_of.year + 1
    return pd.Timestamp(year=wy - 1, month=10, day=1)


# ---------------------------------------------------------------------------
# Outlet-flow-only features
# ---------------------------------------------------------------------------

def feature_outlet_log_recession_rate(outlet, contributors, swe_indexed, swe_climo, as_of, lookback_days):
    """Log-linear slope of outlet discharge over the lookback window.

    A more negative value implies a steeper recession. Climatology assumes
    a roughly constant recession slope; this lets the matcher distinguish
    fast-recession years from gradual ones at the same flow level."""
    window = outlet.loc[as_of - pd.Timedelta(days=lookback_days - 1) : as_of, "discharge_cfs"]
    if len(window) < 3:
        return {"outlet_log_recession_rate": float("nan")}
    log_q = np.log(np.clip(window.values, 1.0, None))
    x = np.arange(len(log_q))
    slope, _ = np.polyfit(x, log_q, 1)
    return {"outlet_log_recession_rate": float(slope)}


def feature_outlet_doy_since_peak(outlet, contributors, swe_indexed, swe_climo, as_of, lookback_days):
    """Days since this water year's outlet peak observed so far.

    A small value means the basin is still near peak; a large value means
    we are deep in recession. Matches the basin's position on its
    hydrograph rather than absolute discharge."""
    wy_data = outlet.loc[_wy_start_for(as_of) : as_of, "discharge_cfs"]
    if wy_data.empty:
        return {"outlet_doy_since_peak": float("nan")}
    return {"outlet_doy_since_peak": float((as_of - wy_data.idxmax()).days)}


def feature_outlet_peak_so_far(outlet, contributors, swe_indexed, swe_climo, as_of, lookback_days):
    """Max outlet discharge observed so far this water year."""
    wy_data = outlet.loc[_wy_start_for(as_of) : as_of, "discharge_cfs"]
    if wy_data.empty:
        return {"outlet_peak_so_far": float("nan")}
    return {"outlet_peak_so_far": float(wy_data.max())}


def feature_outlet_volume_to_date(outlet, contributors, swe_indexed, swe_climo, as_of, lookback_days):
    """Cumulative outlet volume Oct 1 -> as_of, in acre-feet.

    Captures total water-year yield to date, an integrated metric that's
    less sensitive to recent noise than instantaneous discharge."""
    wy_data = outlet.loc[_wy_start_for(as_of) : as_of, "discharge_cfs"]
    if wy_data.empty:
        return {"outlet_volume_to_date_af": float("nan")}
    return {"outlet_volume_to_date_af": float(wy_data.sum() * 1.9835)}


def feature_outlet_volatility(outlet, contributors, swe_indexed, swe_climo, as_of, lookback_days):
    """Coefficient of variation of recent outlet discharge."""
    window = outlet.loc[as_of - pd.Timedelta(days=lookback_days - 1) : as_of, "discharge_cfs"]
    if len(window) < 3:
        return {"outlet_volatility": float("nan")}
    m = window.mean()
    if m <= 0:
        return {"outlet_volatility": float("nan")}
    return {"outlet_volatility": float(window.std() / m)}


# ---------------------------------------------------------------------------
# SWE features (require swe_indexed)
# ---------------------------------------------------------------------------

def feature_swe_trend(outlet, contributors, swe_indexed, swe_climo, as_of, lookback_days):
    """Basin-mean SWE slope (inches per day) over the lookback window.

    Strongly negative means active melt; near zero means snowpack stable.
    Separates this-year-melted-fast from this-year-melted-slow at the
    same basin SWE level."""
    if swe_indexed is None or swe_indexed.empty:
        return {"swe_trend_in_per_day": float("nan")}
    start = as_of - pd.Timedelta(days=lookback_days - 1)
    window = swe_indexed[(swe_indexed["date"] >= start) & (swe_indexed["date"] <= as_of)]
    if window.empty:
        return {"swe_trend_in_per_day": float("nan")}
    slopes: list[float] = []
    for _, grp in window.groupby("triplet"):
        if len(grp) < 3:
            continue
        x = (grp["date"] - grp["date"].min()).dt.days.values
        if x.max() == 0:
            continue
        slope, _b = np.polyfit(x, grp["swe_in"].values, 1)
        slopes.append(float(slope))
    if not slopes:
        return {"swe_trend_in_per_day": float("nan")}
    return {"swe_trend_in_per_day": float(np.mean(slopes))}


def feature_swe_peak_to_date(outlet, contributors, swe_indexed, swe_climo, as_of, lookback_days):
    """Basin-mean peak SWE observed so far this water year.

    Total snowpack reservoir size, regardless of how much remains."""
    if swe_indexed is None or swe_indexed.empty:
        return {"swe_peak_to_date_in": float("nan")}
    wy_data = swe_indexed[
        (swe_indexed["date"] >= _wy_start_for(as_of)) & (swe_indexed["date"] <= as_of)
    ]
    if wy_data.empty:
        return {"swe_peak_to_date_in": float("nan")}
    per_station_peak = wy_data.groupby("triplet")["swe_in"].max()
    return {"swe_peak_to_date_in": float(per_station_peak.mean())}


def feature_swe_doy_since_peak(outlet, contributors, swe_indexed, swe_climo, as_of, lookback_days):
    """Days since basin-mean SWE peaked this water year (melt-start proxy)."""
    if swe_indexed is None or swe_indexed.empty:
        return {"swe_doy_since_peak": float("nan")}
    wy_data = swe_indexed[
        (swe_indexed["date"] >= _wy_start_for(as_of)) & (swe_indexed["date"] <= as_of)
    ]
    if wy_data.empty:
        return {"swe_doy_since_peak": float("nan")}
    daily_mean = wy_data.groupby("date")["swe_in"].mean()
    if daily_mean.empty:
        return {"swe_doy_since_peak": float("nan")}
    peak_date = daily_mean.idxmax()
    return {"swe_doy_since_peak": float((as_of - pd.Timestamp(peak_date)).days)}


# ---------------------------------------------------------------------------
# Catalog (ordered)
# ---------------------------------------------------------------------------

FEATURE_CATALOG: list[tuple[str, FeatureFn, str]] = [
    ("outlet_log_recession_rate", feature_outlet_log_recession_rate,
     "Log-linear slope of outlet discharge over lookback (recession steepness)"),
    ("outlet_doy_since_peak", feature_outlet_doy_since_peak,
     "Days since this water-year's outlet peak observed so far"),
    ("outlet_peak_so_far", feature_outlet_peak_so_far,
     "Maximum outlet discharge observed this water year"),
    ("outlet_volume_to_date", feature_outlet_volume_to_date,
     "Cumulative outlet volume Oct-1..as_of (acre-feet)"),
    ("outlet_volatility", feature_outlet_volatility,
     "Coefficient of variation of recent outlet discharge"),
    ("swe_trend", feature_swe_trend,
     "Basin-mean SWE slope (in/day) over lookback (negative = melt)"),
    ("swe_peak_to_date", feature_swe_peak_to_date,
     "Basin-mean peak SWE observed so far this water year"),
    ("swe_doy_since_peak", feature_swe_doy_since_peak,
     "Days since basin-mean SWE peak this water year (melt-start proxy)"),
]


def get_feature_fn(name: str) -> FeatureFn:
    for n, fn, _desc in FEATURE_CATALOG:
        if n == name:
            return fn
    raise KeyError(f"Unknown feature {name!r}. Known: {[n for n, _, _ in FEATURE_CATALOG]}")
