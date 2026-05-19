"""Reservoir inflow/outflow retrieval via CO DWR CDSS REST API.

For Colorado basins, the most useful upstream-reservoir signal is the
*inflow* and *outflow* discharge measured at gauges immediately above and
below each dam. The net (inflow - outflow) approximates the daily change
in storage; the outflow itself is what physically reaches downstream gauges.

CDSS exposes these as "surface water" stations with abbrevs (e.g. YAMSTACO
above Stagecoach, YAMBSRCO below). This module hits the CDSS REST API
directly - there is no first-class Python client we want to take a dep on.
"""
from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import requests

CDSS_BASE = "https://dwr.state.co.us/Rest/GET"
REQUEST_TIMEOUT_SEC = 60

# CDSS returns HTTP 404 when a properly-formatted query matches zero records.
# (See body text "This URL is properly formatted, but returns zero records from CDSS.")
# We treat 404 as an empty result rather than an error.


def search_cdss_stations(
    water_district: int | None = None,
    county: str | None = None,
    name_contains: str | None = None,
) -> list[dict[str, Any]]:
    """Search CDSS surface water stations.

    `name_contains` is applied client-side because the CDSS server-side
    `stationName=` parameter requires an exact match.
    """
    params: dict[str, Any] = {"format": "json", "pageSize": 500}
    if water_district is not None:
        params["waterDistrict"] = water_district
    if county is not None:
        params["county"] = county.upper()

    url = f"{CDSS_BASE}/api/v2/surfacewater/surfacewaterstations"
    resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SEC)
    if resp.status_code == 404:
        return []
    resp.raise_for_status()
    results = resp.json().get("ResultList", []) or []
    if name_contains:
        needle = name_contains.upper()
        results = [s for s in results if needle in (s.get("stationName") or "").upper()]
    return results


def fetch_daily_streamflow_cdss(
    abbrev: str,
    start: str | date | None = None,
    end: str | date | None = None,
) -> pd.DataFrame:
    """Daily streamflow (cfs) for a CDSS surface water station abbrev.

    Returns a DataFrame indexed by date with column `discharge_cfs`. CDSS
    timestamps are MST; we drop the timezone for ease of joining.
    """
    params: dict[str, Any] = {"format": "json", "abbrev": abbrev, "pageSize": 50000}
    if start is not None:
        params["min-measDate"] = start if isinstance(start, str) else start.isoformat()
    if end is not None:
        params["max-measDate"] = end if isinstance(end, str) else end.isoformat()
    url = f"{CDSS_BASE}/api/v2/surfacewater/surfacewatertsday"
    resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SEC)
    if resp.status_code == 404:
        return pd.DataFrame(columns=["discharge_cfs"])
    resp.raise_for_status()
    rows = resp.json().get("ResultList", []) or []
    if not rows:
        return pd.DataFrame(columns=["discharge_cfs"])

    df = pd.DataFrame(rows)
    # CDSS returns ISO timestamps with mixed MST/MDT offsets; force UTC then drop tz.
    df["date"] = pd.to_datetime(df["measDate"], utc=True).dt.tz_convert(None).dt.normalize()
    out = (
        df[["date", "value"]]
        .rename(columns={"value": "discharge_cfs"})
        .set_index("date")
        .sort_index()
    )
    out["discharge_cfs"] = pd.to_numeric(out["discharge_cfs"], errors="coerce")
    return out


def fetch_reservoir_pair_daily(
    inflow_abbrev: str,
    outflow_abbrev: str,
    start: str | date | None = None,
    end: str | date | None = None,
) -> pd.DataFrame:
    """Daily inflow, outflow, and (inflow - outflow) for an above/below pair.

    Net is a proxy for daily storage change in cfs-days. To convert to
    acre-feet, multiply by ~1.9835.
    """
    inflow = fetch_daily_streamflow_cdss(inflow_abbrev, start, end).rename(
        columns={"discharge_cfs": "inflow_cfs"}
    )
    outflow = fetch_daily_streamflow_cdss(outflow_abbrev, start, end).rename(
        columns={"discharge_cfs": "outflow_cfs"}
    )
    joined = inflow.join(outflow, how="outer")
    joined["net_cfs"] = joined["inflow_cfs"] - joined["outflow_cfs"]
    return joined
