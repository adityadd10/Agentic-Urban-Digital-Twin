"""Scenario generation (dev doc §4.3, module M3 — flood-only scope).

Prototype-1 deviation: the dev doc's full frozen-suite machinery (20
train / 10 val / 10 test scenarios per incident type, written once and
never regenerated) is still not built. Dev doc §4.3 (revised 2026-09-26)
says to build it only once scenarios are shown to actually differ.

**2026-09-26 revision (dev doc §4.3).** Scenarios used to differ only in
severity (0.5-1.0), and every flood covered the same area with the same
timing, so the Stage 2 review found "40 copies of one flood". Each scenario
now samples:
- severity ~ U(0.2, 1.0);
- a rainfall footprint: centroid uniform inside the ward polygon,
  sigma ~ U(800, 2500) m (the ward is ~4.7 x 9 km);
- grow/hold/recede durations ~ U(1,3) / U(4,10) / U(4,8) h;
- a `fragility_seed` for the facilities' critical flood depths;
- `initial_conditions`: `onset_hour_of_day` ~ U(0, 24) and, per hospital,
  `bed_occupancy_fraction` ~ U(0.6, 0.9) (dev doc §4.3's "bed occupancy
  60-90%, time-of-day"), applied with `apply_initial_conditions`.

All ranges are disclosed design choices, not fitted to Mumbai flood records.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from shapely.geometry import Point, shape

from udt.common.models import AssetType, DependencyGraph, Incident, Scenario

DEFAULT_SEVERITY_RANGE = (0.2, 1.0)
FOOTPRINT_SIGMA_RANGE_M = (800.0, 2500.0)
GROWTH_HOURS_RANGE = (1.0, 3.0)
HOLD_HOURS_RANGE = (4.0, 10.0)
RECEDE_HOURS_RANGE = (4.0, 8.0)
BED_OCCUPANCY_RANGE = (0.6, 0.9)
_MAX_CENTROID_ATTEMPTS = 1000


def _sample_point_in(polygon: Any, rng: np.random.Generator) -> tuple[float, float]:
    minx, miny, maxx, maxy = polygon.bounds
    for _ in range(_MAX_CENTROID_ATTEMPTS):
        lon, lat = float(rng.uniform(minx, maxx)), float(rng.uniform(miny, maxy))
        if polygon.contains(Point(lon, lat)):
            return lon, lat
    centroid = polygon.representative_point()
    return float(centroid.x), float(centroid.y)


def generate_flood_scenario(
    *,
    scenario_id: str,
    ward_boundary_geojson: dict[str, Any],
    seed: int,
    onset_tick: int = 0,
    severity_range: tuple[float, float] = DEFAULT_SEVERITY_RANGE,
) -> Scenario:
    """Dev doc §4.3: samples one flood scenario (see module docstring for
    every sampled parameter). Same `seed` -> same scenario."""
    rng = np.random.default_rng(seed)
    ward_geometry = ward_boundary_geojson["features"][0]["geometry"]
    severity = float(rng.uniform(*severity_range))
    c_lon, c_lat = _sample_point_in(shape(ward_geometry), rng)

    incident = Incident(
        incident_id=f"{scenario_id}_incident",
        type="flood",
        location=ward_geometry,
        onset_tick=onset_tick,
        severity=severity,
        raw_signal={"severity_range": list(severity_range)},
        directly_affected_assets=[],  # flood has no single directly-hit asset — spatial, not point
        profile={
            "growth_hours": float(rng.uniform(*GROWTH_HOURS_RANGE)),
            "hold_hours": float(rng.uniform(*HOLD_HOURS_RANGE)),
            "recede_hours": float(rng.uniform(*RECEDE_HOURS_RANGE)),
            "footprint": {
                "lon": c_lon,
                "lat": c_lat,
                "sigma_m": float(rng.uniform(*FOOTPRINT_SIGMA_RANGE_M)),
            },
            "fragility_seed": int(rng.integers(2**31 - 1)),
        },
    )
    return Scenario(
        scenario_id=scenario_id,
        incident=incident,
        initial_conditions={
            "onset_hour_of_day": float(rng.uniform(0.0, 24.0)),
            # Drawn per hospital at apply time from this seed, so the
            # scenario doesn't need to know the graph's hospital ids.
            "bed_occupancy_seed": int(rng.integers(2**31 - 1)),
            "bed_occupancy_range": list(BED_OCCUPANCY_RANGE),
        },
        seed=seed,
    )


def apply_initial_conditions(dep_graph: DependencyGraph, scenario: Scenario) -> DependencyGraph:
    """Returns a copy of `dep_graph` with the scenario's hospital bed
    occupancy applied. The onset hour is passed to `Simulator` separately
    (`scenario.initial_conditions["onset_hour_of_day"]`)."""
    graph = dep_graph.model_copy(deep=True)
    ic = scenario.initial_conditions
    if "bed_occupancy_seed" not in ic:
        return graph
    low, high = ic.get("bed_occupancy_range", BED_OCCUPANCY_RANGE)
    rng = np.random.default_rng(int(ic["bed_occupancy_seed"]))
    for asset in sorted(graph.assets, key=lambda a: a.asset_id):
        if asset.asset_type != AssetType.HOSPITAL:
            continue
        beds_total = int(asset.attributes.get("beds_total", 0))
        asset.attributes["beds_occupied"] = round(beds_total * float(rng.uniform(low, high)))
    return graph


def onset_hour_of_day(scenario: Scenario) -> float:
    return float(scenario.initial_conditions.get("onset_hour_of_day", 0.0))
