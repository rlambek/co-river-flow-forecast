"""Find candidate reservoir inflow/outflow gauges in CDSS for a basin.

Lists CDSS surface-water stations in the basin's water district whose name
mentions a reservoir or lake. Prints them so they can be hand-curated into
basins.py as ReservoirPair entries.

Usage:
    python scripts/basin_discover_reservoirs.py --basin yampa_steamboat
"""
from __future__ import annotations

import argparse

from co_river_flow_forecast.basins import get_basin
from co_river_flow_forecast.data.reservoirs import search_cdss_stations

RESERVOIR_KEYWORDS = ("RES", "LAKE", "DAM", "RESERVOIR", "POND", "STORAGE")


def main(basin_short: str) -> None:
    basin = get_basin(basin_short)
    if basin.cdss_water_district is None:
        raise SystemExit(
            f"Basin {basin.short_name} has no `cdss_water_district` set; "
            "add one to basins.py before running reservoir discovery."
        )

    print(f"Searching CDSS water district {basin.cdss_water_district} for "
          f"reservoir-related stations...")
    stations = search_cdss_stations(water_district=basin.cdss_water_district)
    print(f"  total surface-water stations in district: {len(stations)}")

    hits = [
        s for s in stations
        if any(kw in (s.get("stationName") or "").upper() for kw in RESERVOIR_KEYWORDS)
    ]
    print(f"  candidates mentioning a reservoir/lake/dam: {len(hits)}")
    print()
    print(f"  {'abbrev':<12}  {'station name'}")
    print(f"  {'-'*12}  {'-'*60}")
    for s in hits:
        print(f"  {s.get('abbrev','?'):<12}  {s.get('stationName','')}")

    print()
    print("Pair these as ReservoirPair(name, inflow_abbrev, outflow_abbrev) in basins.py.")
    print("Typical naming: an inflow gauge is ABOVE or NEAR the reservoir; outflow is BELOW.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--basin", default="yampa_steamboat", help="Basin short_name")
    args = p.parse_args()
    main(args.basin)
