"""Peak-flow-day baseline modeling for a single basin.

Assembles a per-water-year feature matrix from SNOTEL SWE peaks and USGS
peak-flow records, then evaluates several baselines via leave-one-water-year-out
cross-validation. The target is the day-of-water-year (Oct 1 = 1) on which
peak discharge occurs at the basin's outlet gauge.

Features are all knowable by mid-May, after most basins' peak-SWE date,
so the resulting LOO MAE is an upper bound on how well a *snowpack-only*
model can do for the seasonal peak-timing target. Weather forecasts,
reservoir state, and intra-melt-season observations will tighten this.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from co_river_flow_forecast.basins import Basin
from co_river_flow_forecast.data import fetch_daily_streamflow, fetch_daily_swe
from co_river_flow_forecast.water_year import water_year, water_year_doy

MIN_DAYS_FOR_COMPLETE_WY = 360


@dataclass
class TrainingFrame:
    X: pd.DataFrame              # feature matrix, indexed by water_year
    y: pd.Series                 # target: peak flow water-year day-of-year
    feature_cols: list[str]
    target_name: str = "peak_flow_wy_doy"


def build_feature_frame(basin: Basin, snotel_start: str = "1980-10-01") -> TrainingFrame:
    """Per-water-year features + peak-flow-day target for a basin."""
    snotel_daily = fetch_daily_swe(basin.snotel_triplets, start=snotel_start)
    if snotel_daily.empty:
        raise ValueError(f"No SNOTEL data returned for {basin.short_name}.")
    snotel_daily["water_year"] = snotel_daily["date"].map(water_year)
    snotel_daily["wy_doy"] = snotel_daily["date"].map(water_year_doy)

    # Per-station, per-water-year peak SWE and its DOY.
    annual = (
        snotel_daily.sort_values("swe_in", ascending=False)
        .drop_duplicates(["triplet", "water_year"])
        .rename(columns={"swe_in": "peak_swe_in", "wy_doy": "peak_swe_doy"})
        [["triplet", "water_year", "peak_swe_in", "peak_swe_doy"]]
    )

    # Basin-aggregate snowpack features.
    agg = (
        annual.groupby("water_year")
        .agg(
            snow_peak_mean=("peak_swe_in", "mean"),
            snow_peak_max=("peak_swe_in", "max"),
            snow_peak_min=("peak_swe_in", "min"),
            snow_doy_mean=("peak_swe_doy", "mean"),
            snow_doy_max=("peak_swe_doy", "max"),
            snow_doy_std=("peak_swe_doy", "std"),
            n_stations=("triplet", "nunique"),
        )
        .reset_index()
    )
    # Single-station basins have NaN std; replace with 0.
    agg["snow_doy_std"] = agg["snow_doy_std"].fillna(0.0)

    # Flow data: peak DOY per water year.
    flow = fetch_daily_streamflow(basin.usgs_id).dropna(subset=["discharge_cfs"])
    flow["water_year"] = flow.index.map(water_year)
    flow["wy_doy"] = flow.index.map(water_year_doy)
    counts = flow.groupby("water_year").size()
    complete_years = counts[counts >= MIN_DAYS_FOR_COMPLETE_WY].index
    flow = flow[flow["water_year"].isin(complete_years)]

    peak_idx = flow.groupby("water_year")["discharge_cfs"].idxmax()
    peaks = (
        flow.loc[peak_idx, ["water_year", "wy_doy", "discharge_cfs"]]
        .rename(columns={"wy_doy": "peak_flow_wy_doy", "discharge_cfs": "peak_flow_cfs"})
    )

    df = agg.merge(peaks, on="water_year", how="inner").dropna()
    # Keep water_year as the index, NOT a feature - the model should
    # generalize across years rather than memorize a trend.
    feature_cols = [
        "snow_peak_mean",
        "snow_peak_max",
        "snow_peak_min",
        "snow_doy_mean",
        "snow_doy_max",
        "snow_doy_std",
    ]
    indexed = df.set_index("water_year")
    return TrainingFrame(
        X=indexed[feature_cols].copy(),
        y=indexed["peak_flow_wy_doy"].astype(float).copy(),
        feature_cols=feature_cols,
    )


class ClimatologyMean:
    """Always predict the in-fold mean of the target. The floor any ML must beat."""

    def fit(self, X, y):
        self.mean_ = float(np.asarray(y).mean())
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_, dtype=float)


def evaluate_loo(
    tf: TrainingFrame,
    model_factory: Callable[[], object],
) -> dict:
    """Leave-one-water-year-out evaluation.

    Returns MAE, RMSE, bias, and hit rates for symmetric 7-day and 14-day
    windows centered on the prediction.
    """
    preds: list[float] = []
    truths: list[float] = []
    for wy in tf.X.index:
        train_X = tf.X.drop(index=wy)
        train_y = tf.y.drop(index=wy)
        model = model_factory()
        model.fit(train_X.values, train_y.values)
        pred = float(model.predict(tf.X.loc[[wy]].values)[0])
        preds.append(pred)
        truths.append(float(tf.y.loc[wy]))

    preds_arr = np.asarray(preds)
    truths_arr = np.asarray(truths)
    err = preds_arr - truths_arr
    abs_err = np.abs(err)
    return {
        "n": int(len(truths_arr)),
        "mae_days": float(abs_err.mean()),
        "rmse_days": float(np.sqrt((err ** 2).mean())),
        "bias_days": float(err.mean()),
        "hit_rate_7d_window": float((abs_err <= 3.5).mean()),
        "hit_rate_14d_window": float((abs_err <= 7.0).mean()),
        "preds": preds_arr,
        "truths": truths_arr,
        "water_years": list(tf.X.index),
    }
