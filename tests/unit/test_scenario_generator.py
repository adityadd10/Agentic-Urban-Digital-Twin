"""Phase 3 tests for `udt.scenarios.generator` (dev doc §4.3, flood-only scope)."""

from __future__ import annotations

import pytest

from udt.scenarios.generator import generate_flood_scenario

WARD_BOUNDARY = {
    "features": [
        {
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[72.86, 19.05], [72.91, 19.05], [72.91, 19.13], [72.86, 19.13], [72.86, 19.05]]
                ],
            }
        }
    ]
}


@pytest.mark.phase3
def test_severity_is_within_configured_range() -> None:
    scenario = generate_flood_scenario(
        scenario_id="s1", ward_boundary_geojson=WARD_BOUNDARY, seed=0, severity_range=(0.4, 0.6)
    )
    assert 0.4 <= scenario.incident.severity <= 0.6


@pytest.mark.phase3
def test_same_seed_is_reproducible() -> None:
    a = generate_flood_scenario(scenario_id="s1", ward_boundary_geojson=WARD_BOUNDARY, seed=42)
    b = generate_flood_scenario(scenario_id="s1", ward_boundary_geojson=WARD_BOUNDARY, seed=42)
    assert a.incident.severity == b.incident.severity


@pytest.mark.phase3
def test_different_seeds_usually_differ() -> None:
    a = generate_flood_scenario(scenario_id="s1", ward_boundary_geojson=WARD_BOUNDARY, seed=1)
    b = generate_flood_scenario(scenario_id="s1", ward_boundary_geojson=WARD_BOUNDARY, seed=2)
    assert a.incident.severity != b.incident.severity


@pytest.mark.phase3
def test_incident_type_is_flood_and_location_is_ward_boundary() -> None:
    scenario = generate_flood_scenario(
        scenario_id="s1", ward_boundary_geojson=WARD_BOUNDARY, seed=0
    )
    assert scenario.incident.type == "flood"
    assert scenario.incident.location == WARD_BOUNDARY["features"][0]["geometry"]
