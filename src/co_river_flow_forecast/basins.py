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
        # Two confirmed Upper-Yampa SNOTELs. TODO: enrich via HUC-polygon
        # intersection (NRCS WBD + metloom.SnotelPointData.points_from_geometry)
        # rather than hand-curating. Tower SNOTEL in particular needs verifying;
        # site 482 is the snow course (monthly only), and 1041 appears to be
        # a different, lower-elevation station.
        "709:CO:SNTL",   # Rabbit Ears
        "840:CO:SNTL",   # Walton Creek
    ),
    notes="Park Range / Rabbit Ears Pass headwaters. Daily record from 1904.",
)


_ALL: tuple[Basin, ...] = (
    YAMPA_AT_STEAMBOAT,
)

BASINS: dict[str, Basin] = {b.short_name: b for b in _ALL}


def get_basin(short_name: str) -> Basin:
    try:
        return BASINS[short_name]
    except KeyError:
        known = ", ".join(sorted(BASINS))
        raise KeyError(f"Unknown basin '{short_name}'. Known: {known}") from None
