"""Registry of Colorado river basins we forecast for.

A `Basin` bundles everything needed to fetch inputs for one outlet:
the USGS gauge ID, the basin's HUC code, and its associated SNOTEL
stations. Add a new basin by appending a `Basin` instance to `BASINS`.
Scripts and CLIs reference basins by `short_name`.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Basin:
    name: str                               # human-readable
    short_name: str                         # slug; filesystem- and CLI-safe
    usgs_id: str                            # USGS NWIS site number for the outlet
    huc8: str | None = None                 # USGS 8-digit hydrologic unit code
    snotel_triplets: tuple[str, ...] = field(default_factory=tuple)  # metloom-format "<id>:<state>:SNTL"
    notes: str = ""


YAMPA_AT_STEAMBOAT = Basin(
    name="Yampa River at Steamboat Springs",
    short_name="yampa_steamboat",
    usgs_id="09239500",
    huc8="14050001",  # Upper Yampa
    snotel_triplets=(
        # Discovered via NLDI upstream-of-gauge polygon + metloom on 2026-05-18.
        # To refresh: scripts/basin_discover_sites.py --basin yampa_steamboat
        "825:CO:SNTL",   # Tower (10,610 ft)
        "709:CO:SNTL",   # Rabbit Ears (9,390 ft)
        "1061:CO:SNTL",  # Bear River (9,100 ft)
        "426:CO:SNTL",   # Crosho (8,960 ft)
        "457:CO:SNTL",   # Dry Lake (8,240 ft)
    ),
    notes="Park Range / Rabbit Ears Pass headwaters. Daily record from 1904.",
)


CLEAR_CREEK_AT_GOLDEN = Basin(
    name="Clear Creek at Golden",
    short_name="clear_creek_golden",
    usgs_id="06719505",
    huc8="10190004",  # Clear Creek
    snotel_triplets=(
        # Discovered via NLDI on 2026-05-18 (only 1 SWE-reporting SNOTEL in basin).
        # Front Range basins are SNOTEL-sparse; gridded SWE (SNODAS/SWANN) will
        # matter more here than for Yampa.
        "602:CO:SNTL",  # Loveland Basin (11,410 ft)
    ),
    notes="Front Range / Loveland Pass headwaters; heavy upstream diversion and reservoir activity.",
)


_ALL: tuple[Basin, ...] = (
    YAMPA_AT_STEAMBOAT,
    CLEAR_CREEK_AT_GOLDEN,
)

BASINS: dict[str, Basin] = {b.short_name: b for b in _ALL}


def get_basin(short_name: str) -> Basin:
    try:
        return BASINS[short_name]
    except KeyError:
        known = ", ".join(sorted(BASINS))
        raise KeyError(f"Unknown basin '{short_name}'. Known: {known}") from None
