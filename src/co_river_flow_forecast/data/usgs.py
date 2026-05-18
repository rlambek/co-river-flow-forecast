"""USGS NWIS streamflow retrieval."""
from __future__ import annotations

from datetime import date

import pandas as pd
from dataretrieval import nwis

DISCHARGE_PARAM_CD = "00060"  # cubic feet per second


def fetch_daily_streamflow(
    site_id: str,
    start: str | date | None = "1900-01-01",
    end: str | date | None = None,
) -> pd.DataFrame:
    """Daily mean discharge (cfs) for a USGS NWIS site.

    Returns a DataFrame indexed by datetime with columns:
      - discharge_cfs : daily mean discharge in cubic feet per second
      - qualifier    : USGS data qualifier code (e.g. "A" approved, "P" provisional)

    `start` defaults to a date well before any modern USGS gauge began
    reporting, so the default call returns the full available record.
    """
    if end is None:
        end = date.today()
    df, _ = nwis.get_dv(
        sites=site_id,
        parameterCd=DISCHARGE_PARAM_CD,
        start=_iso(start),
        end=_iso(end),
    )
    if df.empty:
        return pd.DataFrame(columns=["discharge_cfs", "qualifier"])

    flow_col = f"{DISCHARGE_PARAM_CD}_Mean"
    qual_col = f"{DISCHARGE_PARAM_CD}_Mean_cd"
    out = pd.DataFrame(
        {
            "discharge_cfs": df[flow_col],
            "qualifier": df[qual_col] if qual_col in df.columns else pd.NA,
        },
        index=df.index,
    )
    return out


def _iso(d: str | date | None) -> str | None:
    if d is None:
        return None
    return d.isoformat() if isinstance(d, date) else d
