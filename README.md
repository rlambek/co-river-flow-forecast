# co-river-flow-forecast

Forecast **peak flow timing** (~7-day window) and **7-day flow projections** for rivers in Colorado.

The aim is to throw the whole kitchen sink at the problem: snowpack, terrain insolation on snowfields, weather forecasts, basin history, and dam operations.

## Phases

1. **Gauged rivers.** Build, validate, and report forecasts for Colorado rivers with USGS streamflow gauges. The forecast horizon is 7 days; the peak-window prediction is a ~7-day envelope around expected peak discharge.
2. **Ungauged rivers.** Once Phase 1 is reporting reliably, extend the model to ungauged reaches via regionalization (donor-basin transfer from analog gauged basins).

Phase 1 prototype basin: **Yampa River at Steamboat Springs, CO** — USGS gauge `09239500`, daily record back to 1904.

## Prior art we're standing on

- **DrivenData Water Supply Forecast Rodeo** (2023–24) — predicts seasonal volumes at 26 Western US sites *including Yampa @ Steamboat*. The [runtime repo](https://github.com/drivendataorg/water-supply-forecast-rodeo-runtime) bundles ingest for SNOTEL, SWANN, USGS, CDEC, PDSI, ECMWF, MJO. [Winners' code](https://github.com/drivendataorg/water-supply-forecast-rodeo) (CatBoost quantile ensembles) is also public. Our horizon is shorter (7-day vs seasonal) but their data plumbing is reusable.
- **NeuralHydrology** ([repo](https://github.com/neuralhydrology/neuralhydrology), Kratzert et al.) — the de-facto regional LSTM library for CAMELS-style daily discharge. Our continuous-flow head starts here, pretrained on Caravan / CAMELS-US.
- **NRCS M4** ([repo](https://github.com/nrcs-nwcc/M4)) — what NRCS actually runs operationally for seasonal water-supply forecasts; CatBoost/XGB multi-model. Confirms that gradient-boosted quantile models work for the seasonal target.
- **Operational baselines we ingest as features and must beat:** NWM medium/long-range, CBRFC ESP traces, NRCS monthly volume forecasts. If we don't beat CBRFC ESP on Yampa, we don't have a product.
- **USGS SIR 2021-5016** (Day, 2021) — Upper Yampa assessment showing a ~22% downward trend in April mean flow at 09239500 and earlier snowmelt timing since 1992. Useful prior on peak-timing drift.

## Inputs and tooling

| Signal                        | Source                            | Python tool                          |
| ----------------------------- | --------------------------------- | ------------------------------------ |
| Streamflow (target + history) | USGS NWIS                         | `dataretrieval`                      |
| Snowpack (SWE)                | NRCS SNOTEL / CDEC                | `metloom`                            |
| Snow-covered area, DEMs       | MODIS / VIIRS / Sentinel, 3DEP    | `easysnowdata`                       |
| Gridded SWE                   | UA SWANN, SNODAS                  | direct                               |
| Weather forecast              | HRRR / GFS / GEFS                 | `herbie-data` + `xarray` / `cfgrib`  |
| Climatology forcings          | Daymet, ERA5-Land (Caravan)       | direct                               |
| Reservoir storage & outflow   | USBR RISE, CO DWR (CDSS)          | direct REST                          |
| Operational baselines         | NWM, CBRFC ESP, NRCS M4 monthly   | direct + NWM AWS retrospective       |
| Model library (continuous)    | regional LSTM                     | `neuralhydrology`                    |
| Model library (seasonal)      | quantile gradient boosting        | `catboost` (Rodeo-winner pattern)    |

## Approach

Two heads, two timescales:

- **7-day continuous flow.** Regional LSTM (NeuralHydrology) pretrained on Caravan / CAMELS-US, fine-tuned on Yampa and analog Upper Colorado headwater basins. Features: gauge antecedents, SWE, melt-degree-days, antecedent precip, NWP forecast to D+7, upstream reservoir state. Output: P10 / P50 / P90 daily discharge for D+1..D+7.
- **7-day peak window.** Quantile gradient-boosting model over engineered seasonal features: cumulative SWE, aspect-weighted insolation on snow-bearing terrain, antecedent flow ratios, reservoir state, NWP melt drivers. Output: probability mass over the next 14 days that the seasonal peak falls in each day, collapsed to a 7-day envelope.

Validation: leave-one-water-year-out splits to avoid autocorrelation leakage. Headline metrics: KGE / NSE / PBIAS for continuous flow; peak-window hit rate and peak-day MAE for the seasonal head. **Required to beat:** CBRFC ESP and NWM medium-range on the same windows.

## Repository layout

```
src/co_river_flow_forecast/
  basins.py                   Basin registry (the source of truth)
  water_year.py               Water-year arithmetic shared by all scripts
  data/                       Data fetchers (USGS, SNOTEL; NWP / reservoir next)
scripts/                      Basin-parameterized entry points (--basin <slug>)
data/processed/<basin>/       Per-basin outputs (gitignored)
tests/                        Tests
```

## Adding a new basin

All basin-specific metadata lives in [src/co_river_flow_forecast/basins.py](src/co_river_flow_forecast/basins.py). To add a basin, append a `Basin` instance:

```python
NEW_BASIN = Basin(
    name="Eagle River at Avon",
    short_name="eagle_avon",
    usgs_id="09067020",
    huc8="14010003",
    snotel_triplets=("...:CO:SNTL", ...),
)
```

Add it to `_ALL`, and every `scripts/basin_*.py` script will pick it up via `--basin eagle_avon`. Outputs land in `data/processed/eagle_avon/`. All new modeling code should follow the same pattern: take a `Basin`, derive everything else from it. No basin-specific entry points.

## Getting started

```powershell
# From the repo root, on Windows + Python 3.13:
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e .

# Streamflow climatology for a basin:
.venv\Scripts\python.exe scripts\basin_flow_climatology.py --basin yampa_steamboat

# Discover SNOTEL stations in the upstream-of-gauge polygon (NLDI + metloom):
.venv\Scripts\python.exe scripts\basin_discover_sites.py --basin yampa_steamboat

# SNOTEL SWE summary (curated list, or --discover to re-derive):
.venv\Scripts\python.exe scripts\basin_snotel_summary.py --basin yampa_steamboat

# GFS forecast at the basin centroid for D+0..D+7:
.venv\Scripts\python.exe scripts\basin_nwp_latest.py --basin yampa_steamboat
```

## Status

Phase 1 ingest is wired up for Yampa @ Steamboat:

- USGS daily streamflow via `dataretrieval`.
- SNOTEL daily SWE via `metloom`, with NLDI-based upstream-polygon discovery of stations.
- GFS surface temperature and precipitation via `herbie-data` (smoke test only — point extraction at basin centroid, D+0..D+7).

Adding a new basin requires only an outlet USGS gauge ID; SNOTEL stations are discoverable, basin polygon is fetched from NLDI on demand. Reservoir/operational baselines and the modeling heads are not wired up yet.

## License

MIT — see [LICENSE](LICENSE).
