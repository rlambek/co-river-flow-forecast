"""Auto-managed model configuration (see scripts/confirm_active_config.py).

ACTIVE_EXTRA_FEATURES: extra state-vector feature names to include.
ACTIVE_FEATURE_WEIGHTS: per-feature weights applied in the analog distance
    metric (unspecified features default to weight 1.0).

Current config: up-weight the two SWE features 2x in the analog distance
("swe_2x"). Confirmed via full leave-one-year-out across all 3 basins to
reduce MAE on every basin (Yampa -1.1%, Clear Creek -7.3%, Green River
-8.4%; mean -5.6%).
"""

ACTIVE_EXTRA_FEATURES: list[str] = []

ACTIVE_FEATURE_WEIGHTS: dict[str, float] = {
    "basin_swe_recent_in": 2.0,
    "basin_swe_pct_of_climo": 2.0,
}
