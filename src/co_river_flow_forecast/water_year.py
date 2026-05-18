"""Water-year date arithmetic (Oct 1 -> Sep 30)."""
from __future__ import annotations

import pandas as pd

_REFERENCE_WY_START = pd.Timestamp("2001-10-01")  # non-leap reference for label formatting


def water_year(ts: pd.Timestamp) -> int:
    return ts.year if ts.month < 10 else ts.year + 1


def water_year_doy(ts: pd.Timestamp) -> int:
    wy = water_year(ts)
    wy_start = pd.Timestamp(year=wy - 1, month=10, day=1)
    naive = ts.tz_localize(None) if ts.tzinfo is not None else ts
    return (naive.normalize() - wy_start).days + 1


def doy_to_calendar(doy: float) -> str:
    return (_REFERENCE_WY_START + pd.Timedelta(days=doy - 1)).strftime("%b %d")
