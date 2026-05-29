"""Monthly climate indices: ENSO (MEI v2), PDO, AMO.

Fetched from NOAA PSL whitespace-delimited data files and cached to
`data/cache/climate/<index>.csv`. Each cache has columns `year`, `month`,
and the index value. Out-of-range sentinel values (-9999, -99.99, etc.)
are converted to NaN.
"""
from __future__ import annotations

import os
import re
from typing import Iterable

import pandas as pd
import requests

CACHE_DIR = os.path.join("data", "cache", "climate")

INDEX_URLS = {
    # MEI v2 - Multivariate ENSO Index v2 (bimonthly seasons; we map to month
    # by taking the first month of each season pair as the index for that month).
    "enso_mei": "https://psl.noaa.gov/enso/mei/data/meiv2.data",
    # PDO - Pacific Decadal Oscillation (NCEI 1854-present canonical).
    "pdo":      "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat",
    # AMO - Atlantic Multidecadal Oscillation (unsmoothed).
    "amo":      "https://psl.noaa.gov/data/correlation/amon.us.data",
}

SENTINEL_VALUES = (-99.99, -9999.0, -999.9, -999.0, 9999.0)


def _cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, f"{name}.csv")


def _parse_psl_format(text: str) -> pd.DataFrame:
    """Parse NOAA PSL-style monthly index text.

    Format:
      <start_year> <end_year>      (header line)
      <year> <jan> <feb> ... <dec> (one row per year, 12 monthly values)
      ...optional footer lines...
    """
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    rows: list[tuple[int, int, float]] = []
    for ln in lines:
        # Skip header-ish lines that have only two numeric tokens or non-numeric content.
        tokens = ln.split()
        if len(tokens) < 13:
            continue
        try:
            year = int(tokens[0])
            values = [float(t) for t in tokens[1:13]]
        except ValueError:
            continue
        for m, v in enumerate(values, start=1):
            if v in SENTINEL_VALUES or abs(v) > 90:
                continue
            rows.append((year, m, v))
    if not rows:
        return pd.DataFrame(columns=["year", "month", "value"])
    return pd.DataFrame(rows, columns=["year", "month", "value"])


def _parse_ncei_pdo(text: str) -> pd.DataFrame:
    """NCEI ERSSTv5 PDO file: header row of month abbreviations, then
    YYYYMM values? Actually it's YYYY followed by 12 monthly columns labeled
    Jan..Dec. Year + 12 monthly columns per row, with a header.
    """
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    # Locate the header line containing month abbreviations.
    header_idx = None
    for i, ln in enumerate(lines):
        if re.match(r"^\s*Year\s+Jan\s+Feb", ln, re.IGNORECASE):
            header_idx = i
            break
    if header_idx is None:
        # Fall back to PSL-style parser.
        return _parse_psl_format(text)
    rows: list[tuple[int, int, float]] = []
    for ln in lines[header_idx + 1 :]:
        tokens = ln.split()
        if len(tokens) < 13:
            continue
        try:
            year = int(tokens[0])
            values = [float(t) for t in tokens[1:13]]
        except ValueError:
            continue
        for m, v in enumerate(values, start=1):
            if v in SENTINEL_VALUES or abs(v) > 90:
                continue
            rows.append((year, m, v))
    return pd.DataFrame(rows, columns=["year", "month", "value"])


def load_climate_index(name: str, refresh: bool = False, history_start_year: int = 1950) -> pd.DataFrame:
    """Return DataFrame with columns year, month, value for an index."""
    path = _cache_path(name)
    if not refresh and os.path.exists(path):
        return pd.read_csv(path)

    if name not in INDEX_URLS:
        raise KeyError(f"Unknown climate index {name!r}. Known: {list(INDEX_URLS)}")
    url = INDEX_URLS[name]
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    if name == "pdo":
        df = _parse_ncei_pdo(resp.text)
    else:
        df = _parse_psl_format(resp.text)

    df = df[df["year"] >= history_start_year].reset_index(drop=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_all_climate_indices(refresh: bool = False) -> dict[str, pd.DataFrame]:
    return {name: load_climate_index(name, refresh=refresh) for name in INDEX_URLS}


def index_value_for(
    df: pd.DataFrame,
    as_of: pd.Timestamp,
    months_back: int = 3,
    lag_months: int = 1,
) -> float:
    """Mean index value over `months_back` months ending `lag_months` before
    `as_of`'s month.

    Default: for an as-of date in May, average Feb/Mar/Apr values (3 months
    ending 1 month before May). This avoids depending on the current month,
    which is usually not yet published.

    Returns NaN if any required month is missing in `df`.
    """
    if df.empty:
        return float("nan")
    end_period = pd.Period(as_of.to_pydatetime(), freq="M") - lag_months
    start_period = end_period - (months_back - 1)
    months = [(p.year, p.month) for p in pd.period_range(start_period, end_period, freq="M")]
    vals = []
    for y, m in months:
        row = df[(df["year"] == y) & (df["month"] == m)]
        if row.empty:
            return float("nan")
        vals.append(float(row["value"].iloc[0]))
    return float(sum(vals) / len(vals))
