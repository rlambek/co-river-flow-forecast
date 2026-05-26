"""Cache daily SNOTEL SWE for a basin's discovered stations.

The Green River basin polygon contains ~55 stations across WY/CO/UT and
metloom does one HTTP round trip per (station, variable). To keep first-
fetch latency tolerable we (a) limit to the top-N stations by elevation
and (b) cache to CSV. After the first fetch, subsequent runs read the
cached CSV in milliseconds.
"""
from __future__ import annotations

import os

import pandas as pd

from co_river_flow_forecast.basins import Basin
from co_river_flow_forecast.data import discover_snotel_sites, fetch_daily_swe

CACHE_DIR = os.path.join("data", "cache", "swe")
DEFAULT_MAX_STATIONS = 20


def _cache_path(basin: Basin) -> str:
    return os.path.join(CACHE_DIR, f"{basin.short_name}.csv")


def get_basin_snotel_triplets(
    basin: Basin,
    max_stations: int | None = DEFAULT_MAX_STATIONS,
) -> tuple[str, ...]:
    """Return curated SNOTEL triplets if set, else discover from polygon.

    Caps to the highest-elevation `max_stations` because (a) higher
    stations contribute more to snowmelt runoff and (b) the metloom
    fetch is sequential and grows linearly with station count.
    """
    if basin.snotel_triplets:
        return basin.snotel_triplets
    sites = discover_snotel_sites(basin)  # already sorted by elevation desc
    if max_stations is not None:
        sites = sites[:max_stations]
    return tuple(s["triplet"] for s in sites)


def load_basin_swe(
    basin: Basin,
    start: str = "1990-10-01",
    refresh: bool = False,
    max_stations: int | None = DEFAULT_MAX_STATIONS,
) -> pd.DataFrame:
    """Daily SWE for the basin's SNOTEL stations, cached to CSV.

    Returns long-format: ['date', 'triplet', 'station_name', 'swe_in'].
    """
    path = _cache_path(basin)
    if not refresh and os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["date"])
        return df

    triplets = get_basin_snotel_triplets(basin, max_stations=max_stations)
    print(f"  Fetching SWE for {len(triplets)} SNOTEL stations in {basin.short_name}...")
    df = fetch_daily_swe(triplets, start=start)
    if df.empty:
        return df

    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  Wrote SWE cache: {path}  ({len(df):,} rows)")
    return df


def prepare_swe_indexed(swe_long: pd.DataFrame) -> pd.DataFrame:
    """One-time preprocessing of long-format SWE so per-call feature
    extraction is O(1) lookups instead of repeated parse + groupby.

    Returns a copy with `date`, `doy`, `year` columns and a stable order.
    """
    df = swe_long.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["doy"] = df["date"].dt.dayofyear
    df["year"] = df["date"].dt.year
    return df.sort_values("date").reset_index(drop=True)


def precompute_swe_climatology(swe_indexed: pd.DataFrame) -> pd.DataFrame:
    """Median SWE per (triplet, doy) across all years.

    Returns a small DataFrame with columns triplet, doy, climo_swe.
    This is the basin's static SWE climatology; recomputing it per
    forecast call dominated the previous backtest runtime.
    """
    return (
        swe_indexed.groupby(["triplet", "doy"])["swe_in"].median()
        .rename("climo_swe").reset_index()
    )


def basin_swe_features(
    swe_long: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback_days: int = 7,
    swe_indexed: pd.DataFrame | None = None,
    climo: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Compute basin-aggregate SWE features as of a date.

    Performance: pass pre-built `swe_indexed` (from `prepare_swe_indexed`)
    and `climo` (from `precompute_swe_climatology`) to avoid recomputing
    them on every call. Without them this function will build them each
    time which is slow inside an analog-forecast loop.

    Features:
      basin_swe_recent_in     mean SWE across stations over last `lookback_days`
      basin_swe_pct_of_climo  same, expressed as % of same-DOY median
      basin_swe_n_stations    count of stations contributing
    """
    if swe_long.empty and swe_indexed is None:
        return {
            "basin_swe_recent_in": float("nan"),
            "basin_swe_pct_of_climo": float("nan"),
            "basin_swe_n_stations": 0,
        }

    df = swe_indexed if swe_indexed is not None else prepare_swe_indexed(swe_long)
    climo_df = climo if climo is not None else precompute_swe_climatology(df)

    start_date = as_of - pd.Timedelta(days=lookback_days - 1)
    recent = df[(df["date"] >= start_date) & (df["date"] <= as_of)]
    if recent.empty:
        return {
            "basin_swe_recent_in": float("nan"),
            "basin_swe_pct_of_climo": float("nan"),
            "basin_swe_n_stations": 0,
        }

    recent = recent.merge(climo_df, on=["triplet", "doy"], how="left")
    recent["pct_of_climo"] = (recent["swe_in"] / recent["climo_swe"]).where(
        recent["climo_swe"] > 0
    )

    per_station = recent.groupby("triplet").agg(
        recent_swe=("swe_in", "mean"),
        pct=("pct_of_climo", "mean"),
    )
    return {
        "basin_swe_recent_in": float(per_station["recent_swe"].mean()),
        "basin_swe_pct_of_climo": float(per_station["pct"].mean() * 100.0)
        if per_station["pct"].notna().any()
        else float("nan"),
        "basin_swe_n_stations": int(per_station.shape[0]),
    }
