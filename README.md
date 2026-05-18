# co-river-flow-forecast

Forecast **peak flow timing** (~7-day window) and **7-day flow projections** for rivers in Colorado.

The aim is to throw the whole kitchen sink at the problem: snowpack, terrain insolation on snowfields, weather forecasts, basin history, and dam operations.

## Phases

1. **Gauged rivers.** Build, validate, and report forecasts for Colorado rivers with USGS streamflow gauges. The forecast horizon is 7 days; the peak-window prediction is a ~7-day envelope around expected peak discharge.
2. **Ungauged rivers.** Once Phase 1 is reporting reliably, extend the model to ungauged reaches by transferring skill from analog gauged basins (regionalization).

## Inputs we want to use

| Signal                          | Candidate source                                                           |
| ------------------------------- | -------------------------------------------------------------------------- |
| Streamflow (target + history)   | USGS NWIS (`waterservices.usgs.gov`) via `dataRetrieval` / direct API      |
| Snowpack (SWE, depth)           | NRCS SNOTEL / AWDB REST API; SNODAS gridded SWE                            |
| Snow-covered area               | MODIS MOD10A1 / VIIRS snow products                                        |
| Terrain & insolation            | USGS 3DEP DEM → slope, aspect, modeled potential solar radiation per pixel |
| Weather forecast                | NOAA NWS NDFD / GFS / HRRR; ECMWF where available                          |
| Observed weather                | NOAA GHCN-D, RAWS, MesoWest                                                |
| Basin history & climatology     | Derived from USGS + PRISM / Daymet                                         |
| Reservoir storage & outflow     | USBR Reclamation `RISE`; CO DWR; USACE for select dams                     |
| Planned releases                | USBR 24-Month Study / Annual Operating Plan; basin-specific schedules      |

## Approach (sketch, will evolve)

- Per basin, assemble a daily feature table: SWE, melt-degree-days, accumulated insolation on snow-bearing aspects, antecedent precipitation, upstream reservoir state, and forecast weather out to D+7.
- Train two heads:
  - **Peak-window classifier/regressor** — probability mass over the next 14 days that the seasonal peak falls in each day; collapse to a 7-day window.
  - **Flow regressor** — sequence-to-sequence model producing daily discharge D+1..D+7 with prediction intervals.
- Validate with leave-one-water-year-out splits to avoid leakage from autocorrelated series.

## Repository layout

```
src/co_river_flow_forecast/   Python package (empty stub for now)
data/                         Local data cache (gitignored)
tests/                        Tests
```

## Status

Project skeleton only. Nothing runs yet.

## License

MIT — see [LICENSE](LICENSE).
