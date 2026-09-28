"""Repair-crew movement (twin-v3, dev doc §3.9 mechanic 5): `twin.crew`."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from udt.common.models import Asset, AssetType, DependencyGraph, Incident
from udt.twin import crew
from udt.twin.road_network import RoadNetwork
from udt.twin.simulator import DEFAULT_REPAIR_RATE_PER_TICK, Simulator

NO_FLOOD = Incident(
    incident_id="crew_test",
    type="flood",
    location={"type": "Point", "coordinates": [72.88, 19.07]},
    onset_tick=0,
    severity=0.0,
    directly_affected_assets=[],
)


def _chain(n_edges: int, length_m: float) -> RoadNetwork:
    g = nx.DiGraph()
    for i in range(n_edges + 1):
        g.add_node(str(i), x=72.88 + i * 0.001, y=19.07)
    for i in range(n_edges):
        g.add_edge(str(i), str(i + 1), length=length_m, susceptibility=0.0)
        g.add_edge(str(i + 1), str(i), length=length_m, susceptibility=0.0)
    ids = list(g.nodes)
    rn = RoadNetwork(g, ids, np.array([[g.nodes[n]["x"], g.nodes[n]["y"]] for n in ids]))
    rn.update_for_tick(NO_FLOOD, tick=0, dt_minutes=5.0)
    return rn


def _sim(rn: RoadNetwork, damaged: float = 0.5) -> Simulator:
    def sub(asset_id: str, lon: float, level: float) -> Asset:
        return Asset(
            asset_id=asset_id,
            asset_type=AssetType.SUBSTATION,
            geometry={"type": "Point", "coordinates": [lon, 19.07]},
            intrinsic_level=level,
            functional_level=level,
            attributes={"capacity_mw": 30.0, "load_mw": 10.0, "shed_tier": 0},
        )

    return Simulator(
        DependencyGraph(assets=[sub("S0", 72.88, 1.0), sub("S1", 72.883, damaged)], edges=[]),
        road_network=rn,
    )


@pytest.fixture
def travel_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(crew, "REPAIR_CREW_TRAVEL", True)


@pytest.mark.phase1
def test_switch_off_keeps_instant_repair() -> None:
    assert crew.REPAIR_CREW_TRAVEL is False  # twin-v2 default
    sim = _sim(_chain(3, 2000.0))
    sim.step(repair_target="S1")
    assert sim.asset("S1").intrinsic_level == pytest.approx(0.5 + DEFAULT_REPAIR_RATE_PER_TICK)


@pytest.mark.phase1
def test_crew_travels_before_repairing_then_repairs_at_the_v2_speed(travel_on: None) -> None:
    sim = _sim(_chain(3, 2000.0))  # 6 km at 30 km/h = 12 min = 3 ticks away
    sim.step(repair_target="S1")
    assert sim.graph.graph[crew.CREW_KEY]["status"] == "travelling"
    assert sim.asset("S1").intrinsic_level == pytest.approx(0.5)  # no repair while driving
    for _ in range(3):
        sim.step()  # repair_target=None: carry on
    assert sim.graph.graph[crew.CREW_KEY]["status"] == "working"
    before = sim.asset("S1").intrinsic_level
    for _ in range(3):  # one decision interval of work = one v2 repair application
        sim.step()
    assert sim.asset("S1").intrinsic_level == pytest.approx(
        before + DEFAULT_REPAIR_RATE_PER_TICK, abs=1e-9
    )


@pytest.mark.phase1
def test_crew_goes_idle_once_the_site_is_repaired(travel_on: None) -> None:
    sim = _sim(_chain(1, 100.0), damaged=0.99)
    sim.step(repair_target="S1")
    for _ in range(5):
        state = sim.step()
    assert sim.asset("S1").intrinsic_level == pytest.approx(1.0)
    assert state.repair_crew is not None and state.repair_crew["status"] == "idle"


@pytest.mark.phase1
def test_unreachable_target_is_refused_and_the_crew_keeps_its_job(travel_on: None) -> None:
    rn = _chain(3, 2000.0)
    rn.graph.add_node("island", x=99.0, y=99.0)
    rn._node_ids.append("island")
    rn._node_xy = np.vstack([rn._node_xy, [[99.0, 99.0]]])
    sim = _sim(rn)
    island = Asset(
        asset_id="S_far",
        asset_type=AssetType.SUBSTATION,
        geometry={"type": "Point", "coordinates": [99.0, 99.0]},
        intrinsic_level=0.2,
        attributes={"capacity_mw": 30.0, "load_mw": 10.0, "shed_tier": 0},
    )
    sim.graph.add_node("S_far", asset=island)
    sim.step(repair_target="S1")
    assert crew.assign_crew(sim.graph, rn, "S_far") is False
    assert sim.graph.graph[crew.CREW_KEY]["target"] == "S1"
