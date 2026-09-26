"""Phase 4 acceptance tests: the repair mechanic `Simulator.apply_repair`/
`step(repair_target=...)` adds (dev doc §5.6 — "send a repair crew")."""

from __future__ import annotations

import pytest

from udt.common.models import Asset, AssetType, DependencyGraph
from udt.twin.simulator import Simulator

POINT = {"type": "Point", "coordinates": [72.88, 19.07]}


def _single_asset_graph(intrinsic_level: float) -> DependencyGraph:
    asset = Asset(
        asset_id="S1",
        asset_type=AssetType.SUBSTATION,
        geometry=POINT,
        intrinsic_level=intrinsic_level,
        functional_level=intrinsic_level,
        attributes={"capacity_mw": 10.0},
    )
    return DependencyGraph(assets=[asset], edges=[])


@pytest.mark.phase4
def test_apply_repair_increases_intrinsic_level() -> None:
    sim = Simulator(_single_asset_graph(0.5))
    sim.apply_repair("S1", rate=0.1)
    assert sim.asset("S1").intrinsic_level == pytest.approx(0.6)


@pytest.mark.phase4
def test_apply_repair_clips_at_one() -> None:
    sim = Simulator(_single_asset_graph(0.97))
    sim.apply_repair("S1", rate=0.1)
    assert sim.asset("S1").intrinsic_level == pytest.approx(1.0)


@pytest.mark.phase4
def test_step_with_repair_target_restores_over_several_ticks() -> None:
    sim = Simulator(_single_asset_graph(0.5))
    for _ in range(5):
        sim.step(repair_target="S1", repair_rate=0.05)
    # 5 ticks x 0.05 = 0.25 restored, no degradation active this run.
    assert sim.asset("S1").intrinsic_level == pytest.approx(0.75)


@pytest.mark.phase4
def test_step_without_repair_target_leaves_intrinsic_level_unchanged() -> None:
    sim = Simulator(_single_asset_graph(0.5))
    sim.step(repair_target=None)
    assert sim.asset("S1").intrinsic_level == pytest.approx(0.5)
