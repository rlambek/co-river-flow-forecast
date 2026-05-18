"""First look at the Yampa River at Steamboat Springs streamflow record.

Pulls daily discharge for USGS site 09239500 and reports basic peak-timing
climatology, mainly to verify the data pipe end to end.
"""
from __future__ import annotations

import os

import pandas as pd

from co_river_flow_forecast.data import fetch_daily_streamflow

SITE_ID = "09239500"  # Yampa River at Steamboat Springs, CO
MIN_DAYS_FOR_COMPLETE_WY = 360


def water_year(ts: pd.Timestamp) -> int:
    return ts.year if ts.month < 10 else ts.year + 1


def water_year_doy(ts: pd.Timestamp) -> int:
    wy = water_year(ts)
    wy_start = pd.Timestamp(year=wy - 1, month=10, day=1)
    return (ts.normalize().tz_localize(None) - wy_start).days + 1


def doy_to_calendar(doy: float) -> str:
    base = pd.Timestamp("2001-10-01")  # non-leap reference water year
    return (base + pd.Timedelta(days=doy - 1)).strftime("%b %d")


def main() -> None:
    print(f"Fetching daily discharge for USGS {SITE_ID} (Yampa @ Steamboat Springs)...")
    df = fetch_daily_streamflow(SITE_ID)
    df = df.dropna(subset=["discharge_cfs"])
    if df.empty:
        raise SystemExit("No data returned.")

    print(f"  rows:       {len(df):,}")
    print(f"  date span:  {df.index.min().date()} -> {df.index.max().date()}")

    df = df.copy()
    df["water_year"] = df.index.map(water_year)
    df["wy_doy"] = df.index.map(water_year_doy)

    # Annual peak per water year
    peak_idx = df.groupby("water_year")["discharge_cfs"].idxmax()
    peaks = (
        df.loc[peak_idx, ["water_year", "wy_doy", "discharge_cfs"]]
        .reset_index()
        .rename(columns={"datetime": "peak_date"})
    )

    counts = df.groupby("water_year").size()
    complete = counts[counts >= MIN_DAYS_FOR_COMPLETE_WY].index
    peaks = peaks[peaks["water_year"].isin(complete)].reset_index(drop=True)
    print(f"  complete water years (>= {MIN_DAYS_FOR_COMPLETE_WY} days): {len(peaks)}")

    if peaks.empty:
        raise SystemExit("No complete water years in record; nothing to summarize.")

    print()
    print(f"Long-term peak climatology ({len(peaks)} water years):")
    _summarize(peaks)

    most_recent = peaks["water_year"].max()
    recent = peaks[peaks["water_year"] >= most_recent - 9]
    print()
    print(f"Most recent {len(recent)} water years ({recent['water_year'].min()}-{most_recent}):")
    _summarize(recent)

    # Pre-2000 vs 2000+ — relevant to the SIR 2021-5016 trend finding
    pre = peaks[peaks["water_year"] < 2000]
    post = peaks[peaks["water_year"] >= 2000]
    print()
    print(f"Pre-2000 ({len(pre)} years) vs 2000+ ({len(post)} years):")
    print(
        f"  mean peak DOY shifted {post['wy_doy'].mean() - pre['wy_doy'].mean():+.1f} days "
        f"({doy_to_calendar(pre['wy_doy'].mean())} -> {doy_to_calendar(post['wy_doy'].mean())})"
    )
    print(
        f"  mean peak discharge changed {post['discharge_cfs'].mean() - pre['discharge_cfs'].mean():+,.0f} cfs "
        f"({pre['discharge_cfs'].mean():,.0f} -> {post['discharge_cfs'].mean():,.0f})"
    )

    out_dir = os.path.join("data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "yampa_annual_peaks.csv")
    peaks.to_csv(out_path, index=False)
    print()
    print(f"Wrote {out_path}")


def _summarize(peaks: pd.DataFrame) -> None:
    mean_doy = peaks["wy_doy"].mean()
    median_doy = peaks["wy_doy"].median()
    print(f"  mean peak DOY (Oct-1 = 1):    {mean_doy:5.1f}   ({doy_to_calendar(mean_doy)})")
    print(f"  median peak DOY:              {median_doy:5.0f}   ({doy_to_calendar(median_doy)})")
    print(f"  std of peak DOY:              {peaks['wy_doy'].std():5.1f} days")
    print(f"  mean peak discharge:          {peaks['discharge_cfs'].mean():>7,.0f} cfs")
    print(f"  median peak discharge:        {peaks['discharge_cfs'].median():>7,.0f} cfs")


if __name__ == "__main__":
    main()
