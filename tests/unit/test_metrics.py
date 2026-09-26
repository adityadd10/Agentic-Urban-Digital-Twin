"""Phase 4 acceptance tests: `logging.metrics.compute_episode_metrics`
(dev doc §5.4, reduced scope — see the module's docstring)."""

from __future__ import annotations

import pytest

from udt.common.models import Asset, AssetType, TwinState
from udt.logging.metrics import compute_episode_metrics

POINT = {"type": "Point", "coordinates": [72.88, 19.07]}


def _snapshot(
    tick: int,
    cascading_failure_count: int,
    levels: dict[str, float],
    *,
    patient_deaths_cumulative: int = 0,
    hospital_queues: dict[str, int] | None = None,
) -> TwinState:
    type_map = {"H1": AssetType.HOSPITAL, "S1": AssetType.SUBSTATION, "R1": AssetType.ROAD}
    hospital_queues = hospital_queues or {}
    assets = [
        Asset(
            asset_id=asset_id,
            asset_type=type_map[asset_id],
            geometry=POINT,
            intrinsic_level=level,
            functional_level=level,
            attributes={"patient_queue": hospital_queues.get(asset_id, 0)},
        )
        for asset_id, level in levels.items()
    ]
    return TwinState(
        tick=tick,
        assets=assets,
        cascading_failure_count=cascading_failure_count,
        patient_deaths_cumulative=patient_deaths_cumulative,
    )


@pytest.mark.phase4
def test_compute_episode_metrics_averages_functional_levels() -> None:
    trace = [
        _snapshot(0, 0, {"H1": 1.0, "S1": 1.0, "R1": 1.0}),
        _snapshot(1, 1, {"H1": 0.6, "S1": 0.4, "R1": 0.9}, patient_deaths_cumulative=1),
    ]
    metrics = compute_episode_metrics("scn_1", "rule_based", trace)
    assert metrics.scenario_id == "scn_1"
    assert metrics.agent_name == "rule_based"
    assert metrics.ticks_run == 2
    assert metrics.cascading_failure_count == 1  # last snapshot's cumulative count
    assert metrics.mean_hospital_functional_level == pytest.approx((1.0 + 0.6) / 2)
    # critical = hospital + substation (not road)
    assert metrics.mean_critical_functional_level == pytest.approx((1.0 + 1.0 + 0.6 + 0.4) / 4)
    assert metrics.unmet_patient_hours == pytest.approx(0.0)
    assert metrics.patient_deaths == 1


@pytest.mark.phase4
def test_compute_episode_metrics_unmet_patient_hours() -> None:
    trace = [
        _snapshot(0, 0, {"H1": 1.0}, hospital_queues={"H1": 4}),
        _snapshot(1, 0, {"H1": 1.0}, hospital_queues={"H1": 2}),
    ]
    metrics = compute_episode_metrics("scn_1", "rule_based", trace, dt_hours=5.0 / 60.0)
    # (4 + 2) queued-patient-ticks x 5/60 hours per tick
    assert metrics.unmet_patient_hours == pytest.approx((4 + 2) * (5.0 / 60.0))


@pytest.mark.phase4
def test_compute_episode_metrics_empty_trace() -> None:
    metrics = compute_episode_metrics("scn_1", "do_nothing", [])
    assert metrics.ticks_run == 0
    assert metrics.cascading_failure_count == 0
    assert metrics.mean_hospital_functional_level == 1.0
    assert metrics.mean_critical_functional_level == 1.0
    assert metrics.unmet_patient_hours == 0.0
    assert metrics.patient_deaths == 0
