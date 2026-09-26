"""Phase 3 acceptance tests (dev doc §4.2, flood-only scope): temporal
envelope shape, depth formula, road blockage, and persistent fragility-based
facility damage (dev doc §3.8, revised 2026-09-26)."""

from __future__ import annotations

import pytest

from udt.common.models import Asset, AssetType, Incident
from udt.incidents.degradations.flood import (
    FAILED_RESIDUAL,
    MAX_DEPTH_AT_SEVERITY_1_M,
    ROAD_BLOCKAGE_DEPTH_SCALE_M,
    flood_degradation,
    flood_depth_m,
    make_flood_degradation_fn,
    temporal_multiplier,
)


class _FakeRaster:
    """Duck-typed stand-in for `SusceptibilityRaster` — only `value_at` is
    used by the degradation functions, so no real GeoTIFF I/O is needed
    for these tests."""

    def __init__(self, value: float) -> None:
        self._value = value

    def value_at(self, lon: float, lat: float) -> float:
        return self._value


POINT = {"type": "Point", "coordinates": [72.88, 19.07]}


def _incident(severity: float = 1.0, onset_tick: int = 0) -> Incident:
    return Incident(
        incident_id="i1",
        type="flood",
        location=POINT,
        onset_tick=onset_tick,
        severity=severity,
        directly_affected_assets=[],
    )


@pytest.mark.phase3
def test_temporal_multiplier_growth_hold_recede_shape() -> None:
    assert temporal_multiplier(-1.0) == 0.0  # before onset
    assert temporal_multiplier(0.0) == pytest.approx(0.0)
    assert temporal_multiplier(1.0) == pytest.approx(0.5)  # halfway through 2h growth
    assert temporal_multiplier(2.0) == pytest.approx(1.0)  # growth complete
    assert temporal_multiplier(5.0) == pytest.approx(1.0)  # mid-hold (2-10h)
    assert temporal_multiplier(10.0) == pytest.approx(1.0)  # hold ends, recede starts
    assert temporal_multiplier(13.0) == pytest.approx(0.5)  # halfway through 6h recede
    assert temporal_multiplier(16.0) == pytest.approx(0.0)  # fully receded
    assert temporal_multiplier(20.0) == 0.0  # long after


@pytest.mark.phase3
def test_flood_depth_scales_with_severity_and_susceptibility() -> None:
    asset = Asset(
        asset_id="H1",
        asset_type=AssetType.HOSPITAL,
        geometry=POINT,
        intrinsic_level=1.0,
        functional_level=1.0,
    )
    raster = _FakeRaster(1.0)
    incident = _incident(severity=1.0)
    # at peak (tick within hold phase), full severity x full susceptibility x full envelope
    depth = flood_depth_m(incident, asset, tick=int(5 * 60 / 5), susceptibility_raster=raster)
    assert depth == pytest.approx(MAX_DEPTH_AT_SEVERITY_1_M)

    raster_half = _FakeRaster(0.5)
    depth_half = flood_depth_m(
        incident, asset, tick=int(5 * 60 / 5), susceptibility_raster=raster_half
    )
    assert depth_half == pytest.approx(MAX_DEPTH_AT_SEVERITY_1_M * 0.5)


@pytest.mark.phase3
def test_flood_degradation_returns_zero_for_roads() -> None:
    road = Asset(
        asset_id="R1",
        asset_type=AssetType.ROAD,
        geometry=POINT,
        intrinsic_level=1.0,
        functional_level=1.0,
        attributes={"blockage": 0.0},
    )
    result = flood_degradation(_incident(), road, tick=24, susceptibility_raster=_FakeRaster(1.0))
    assert result == 0.0


@pytest.mark.phase3
def test_flood_damage_persists_after_recession() -> None:
    """Dev doc §3.8 item 4 (2026-09-26): a flooded facility drops to its
    residual and does NOT recover when the water recedes. This replaces
    the old recession-recovery behaviour, which repaired facilities for free."""
    hospital = Asset(
        asset_id="H1",
        asset_type=AssetType.HOSPITAL,
        geometry=POINT,
        intrinsic_level=1.0,
        functional_level=1.0,
    )
    raster = _FakeRaster(1.0)
    incident = _incident(severity=1.0)

    peak_tick = int(5 * 60 / 5)  # 5h since onset, inside the 2-10h hold window; depth = 2.0 m
    reduction = flood_degradation(incident, hospital, peak_tick, raster, critical_depth_m=0.6)
    assert reduction == pytest.approx(1.0 - FAILED_RESIDUAL[AssetType.HOSPITAL])
    hospital.intrinsic_level -= reduction

    late_tick = int(20 * 60 / 5)  # fully receded
    assert flood_degradation(incident, hospital, late_tick, raster, critical_depth_m=0.6) == 0.0
    assert hospital.intrinsic_level == pytest.approx(FAILED_RESIDUAL[AssetType.HOSPITAL])


@pytest.mark.phase3
def test_flood_degradation_no_damage_below_critical_depth() -> None:
    hospital = Asset(
        asset_id="H1",
        asset_type=AssetType.HOSPITAL,
        geometry=POINT,
        intrinsic_level=1.0,
        functional_level=1.0,
    )
    peak_tick = int(5 * 60 / 5)
    # depth = 1.0 x 0.2 x 2.0 = 0.4 m, below a 0.6 m critical depth
    assert (
        flood_degradation(
            _incident(severity=1.0), hospital, peak_tick, _FakeRaster(0.2), critical_depth_m=0.6
        )
        == 0.0
    )


@pytest.mark.phase3
def test_make_flood_degradation_fn_sets_road_blockage_not_intrinsic_reduction() -> None:
    import networkx as nx

    road = Asset(
        asset_id="R1",
        asset_type=AssetType.ROAD,
        geometry=POINT,
        intrinsic_level=1.0,
        functional_level=1.0,
        attributes={"blockage": 0.0, "flood_depth_m": 0.0},
    )
    hospital = Asset(
        asset_id="H1",
        asset_type=AssetType.HOSPITAL,
        geometry=POINT,
        intrinsic_level=1.0,
        functional_level=1.0,
    )
    g = nx.DiGraph()
    g.add_node("R1", asset=road)
    g.add_node("H1", asset=hospital)

    incident = _incident(severity=1.0)
    degradation_fn = make_flood_degradation_fn(incident, _FakeRaster(1.0))
    peak_tick = int(5 * 60 / 5)
    reductions = degradation_fn(peak_tick, g)

    assert "R1" not in reductions  # roads affected via blockage, not the reductions dict
    assert road.attributes["blockage"] == pytest.approx(
        min(1.0, MAX_DEPTH_AT_SEVERITY_1_M / ROAD_BLOCKAGE_DEPTH_SCALE_M)
    )
    assert "H1" in reductions
    # 2.0 m at peak exceeds any hospital critical depth -> drops to its residual
    assert reductions["H1"] == pytest.approx(1.0 - FAILED_RESIDUAL[AssetType.HOSPITAL])


@pytest.mark.phase4
def test_flood_degradation_does_not_damage_ambulances() -> None:
    """Regression test for the M4 ambulance-dispatch bug: ambulances
    (added as regular graph nodes once `twin/ambulances.py` existed)
    silently took real flood damage here — nothing about their dispatch/
    movement ever reads intrinsic_level, so it only corrupted
    `cascading_failure_count` with meaningless "failures". Vehicles
    aren't a depth-damaged facility in this model, same as roads."""
    ambulance = Asset(
        asset_id="AMB1",
        asset_type=AssetType.AMBULANCE,
        geometry=POINT,
        intrinsic_level=1.0,
        functional_level=1.0,
    )
    incident = _incident(severity=1.0)
    peak_tick = int(5 * 60 / 5)
    reduction = flood_degradation(incident, ambulance, peak_tick, _FakeRaster(1.0))
    assert reduction == 0.0
