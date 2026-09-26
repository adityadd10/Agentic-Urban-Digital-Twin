"""Phase 7 acceptance tests: `constraints.engine` (dev doc §8), module M7.

Dev doc §13's "Unit — constraints" row: "Each rule: violating action
produces the specified repair, and repaired action passes." Every test
below follows that exact shape — construct an action that violates one
rule, check the violation + repair, then re-run `check(...)` on the
repaired action and assert it now passes.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from udt.common.models import AgentAction, Asset, AssetType, DependencyGraph, Incident
from udt.constraints.engine import (
    BED_CAPACITY_RULE_ID,
    ROUTE_FLOOD_SAFETY_MAX_DEPTH_M,
    ROUTE_FLOOD_SAFETY_RULE_ID,
    check,
)
from udt.twin.graph import build_networkx_graph
from udt.twin.road_network import RoadNetwork


def _point(lon: float, lat: float) -> dict:
    return {"type": "Point", "coordinates": [lon, lat]}


def _hospital_graph() -> nx.DiGraph[str]:
    """Two hospitals: H1 has 40 free beds (150 total, 110 occupied), H2
    has 50 free beds (200 total, 150 occupied) — same numbers as
    `tests/fixtures/sample_graph.py`, kept in sync deliberately so a
    reader cross-checking one against the other sees the same story."""
    dep_graph = DependencyGraph(
        assets=[
            Asset(
                asset_id="H1",
                asset_type=AssetType.HOSPITAL,
                geometry=_point(72.880, 19.072),
                attributes={"beds_total": 150, "beds_occupied": 110},
            ),
            Asset(
                asset_id="H2",
                asset_type=AssetType.HOSPITAL,
                geometry=_point(72.897, 19.092),
                attributes={"beds_total": 200, "beds_occupied": 150},
            ),
        ],
        edges=[],
    )
    return build_networkx_graph(dep_graph)


# ---------------------------------------------------------------------------
# bed_capacity
# ---------------------------------------------------------------------------
@pytest.mark.phase7
def test_bed_capacity_passes_when_transfer_fits() -> None:
    graph = _hospital_graph()
    action = AgentAction(patient_transfer=("H1", "H2", 10))  # H2 has 50 free
    report = check(graph, action)
    assert report.passed
    assert report.violations == []
    assert report.repaired_action.patient_transfer == ("H1", "H2", 10)


@pytest.mark.phase7
def test_bed_capacity_clips_oversized_transfer() -> None:
    graph = _hospital_graph()
    action = AgentAction(patient_transfer=("H1", "H2", 999))  # only 50 free at H2
    report = check(graph, action)
    assert not report.passed
    assert len(report.violations) == 1
    v = report.violations[0]
    assert v.rule_id == BED_CAPACITY_RULE_ID
    assert v.scope == "health"
    assert v.on_violation == "clip"
    assert report.repaired_action.patient_transfer == ("H1", "H2", 50)

    # Repaired action passes on re-check (dev doc §13's acceptance shape).
    re_report = check(graph, report.repaired_action)
    assert re_report.passed


@pytest.mark.phase7
def test_bed_capacity_clips_to_none_when_destination_has_no_free_beds() -> None:
    dep_graph = DependencyGraph(
        assets=[
            Asset(
                asset_id="H1",
                asset_type=AssetType.HOSPITAL,
                geometry=_point(72.880, 19.072),
                attributes={"beds_total": 150, "beds_occupied": 110},
            ),
            Asset(
                asset_id="H2",
                asset_type=AssetType.HOSPITAL,
                geometry=_point(72.897, 19.092),
                attributes={"beds_total": 100, "beds_occupied": 100},  # full
            ),
        ],
        edges=[],
    )
    graph = build_networkx_graph(dep_graph)
    action = AgentAction(patient_transfer=("H1", "H2", 5))
    report = check(graph, action)
    assert not report.passed
    assert report.repaired_action.patient_transfer is None


@pytest.mark.phase8
def test_bed_capacity_uses_functional_level_scaled_capacity_not_nominal() -> None:
    """M8: a destination hospital with plenty of *nominal* free beds but
    a low `functional_level` must still be treated as having reduced
    capacity — `constraints/engine.py` and `twin/demand.py`'s
    `apply_patient_transfer` must agree on what "free" means."""
    dep_graph = DependencyGraph(
        assets=[
            Asset(
                asset_id="H1",
                asset_type=AssetType.HOSPITAL,
                geometry=_point(72.880, 19.072),
                attributes={"beds_total": 150, "beds_occupied": 110},
            ),
            Asset(
                asset_id="H2",
                asset_type=AssetType.HOSPITAL,
                geometry=_point(72.897, 19.092),
                functional_level=0.1,
                attributes={"beds_total": 200, "beds_occupied": 0},  # looks wide open nominally
            ),
        ],
        edges=[],
    )
    graph = build_networkx_graph(dep_graph)
    action = AgentAction(patient_transfer=("H1", "H2", 50))
    report = check(graph, action)
    assert not report.passed
    # effective free beds = round(200 * 0.1) - 0 = 20, not the nominal 200
    assert report.repaired_action.patient_transfer == ("H1", "H2", 20)


@pytest.mark.phase7
def test_no_transfer_action_never_violates_bed_capacity() -> None:
    graph = _hospital_graph()
    report = check(graph, AgentAction())
    assert report.passed
    assert report.repaired_action.patient_transfer is None


# ---------------------------------------------------------------------------
# route_flood_safety
# ---------------------------------------------------------------------------
FLOOD_INCIDENT = Incident(
    incident_id="constraint_test",
    type="flood",
    location=_point(72.88, 19.07),
    onset_tick=0,
    severity=1.0,
    directly_affected_assets=[],
)


def _road_network(susceptibility: float) -> RoadNetwork:
    """A straight A(home) -> B(pickup) road, same `_linear_graph`
    pattern `tests/unit/test_road_network.py` uses. `susceptibility`
    controls how deep it floods at full severity/envelope."""
    g: nx.DiGraph = nx.DiGraph()
    g.add_node("home", x=0.0, y=0.0)
    g.add_node("pickup", x=0.001, y=0.0)
    g.add_edge("home", "pickup", length=300.0, susceptibility=susceptibility)
    node_ids = list(g.nodes)
    node_xy = np.array([[float(g.nodes[n]["x"]), float(g.nodes[n]["y"])] for n in node_ids])
    rn = RoadNetwork(g, node_ids, node_xy)
    rn.update_for_tick(FLOOD_INCIDENT, tick=60, dt_minutes=5.0)  # hold window, envelope=1.0
    return rn


def _ambulance_graph() -> nx.DiGraph[str]:
    dep_graph = DependencyGraph(
        assets=[
            Asset(
                asset_id="AMB_1",
                asset_type=AssetType.AMBULANCE,
                geometry=_point(72.88, 19.07),
                attributes={"status": "idle", "home_node": "home"},
            )
        ],
        edges=[],
    )
    g = build_networkx_graph(dep_graph)
    g.graph["pending_requests"] = [
        {"request_id": "REQ_1", "location": (0.001, 0.0), "requested_at_tick": 0}
    ]
    return g


@pytest.mark.phase7
def test_route_flood_safety_passes_dry_route() -> None:
    graph = _ambulance_graph()
    rn = _road_network(susceptibility=0.0)  # dry, depth 0.0 m < 0.4 m threshold
    action = AgentAction(ambulance_assignment={"AMB_1": "REQ_1"})
    report = check(graph, action, road_network=rn)
    assert report.passed
    assert report.repaired_action.ambulance_assignment == {"AMB_1": "REQ_1"}


@pytest.mark.phase7
def test_route_flood_safety_drops_unsafe_dispatch() -> None:
    # ROAD_BLOCKAGE_DEPTH_SCALE_M=0.6, susceptibility=1.0 -> raw depth
    # 0.6 m, well past the rule's 0.4 m threshold but still < the
    # separate IMPASSABLE_BLOCKAGE routing cutoff, so a route still
    # exists (this is a "your route is unsafe", not "no route").
    graph = _ambulance_graph()
    rn = _road_network(susceptibility=1.0)
    action = AgentAction(ambulance_assignment={"AMB_1": "REQ_1"})
    report = check(graph, action, road_network=rn)
    assert not report.passed
    assert len(report.violations) == 1
    v = report.violations[0]
    assert v.rule_id == ROUTE_FLOOD_SAFETY_RULE_ID
    assert v.scope == "transport"
    assert v.on_violation == "drop"
    assert report.repaired_action.ambulance_assignment is None

    re_report = check(graph, report.repaired_action, road_network=rn)
    assert re_report.passed


@pytest.mark.phase7
def test_route_flood_safety_depth_exactly_at_threshold_is_unsafe() -> None:
    """Rule text is `< 0.4`, so exactly 0.4 m must NOT pass."""
    graph = _ambulance_graph()
    # susceptibility x MAX_DEPTH_AT_SEVERITY_1_M(from flood.py, severity 1,
    # envelope 1) tuned so raw depth == ROUTE_FLOOD_SAFETY_MAX_DEPTH_M.
    from udt.incidents.degradations.flood import MAX_DEPTH_AT_SEVERITY_1_M

    susceptibility = ROUTE_FLOOD_SAFETY_MAX_DEPTH_M / MAX_DEPTH_AT_SEVERITY_1_M
    rn = _road_network(susceptibility=susceptibility)
    action = AgentAction(ambulance_assignment={"AMB_1": "REQ_1"})
    report = check(graph, action, road_network=rn)
    assert not report.passed


@pytest.mark.phase7
def test_route_flood_safety_skipped_without_road_network() -> None:
    graph = _ambulance_graph()
    action = AgentAction(ambulance_assignment={"AMB_1": "REQ_1"})
    report = check(graph, action, road_network=None)
    assert report.passed
    assert report.repaired_action.ambulance_assignment == {"AMB_1": "REQ_1"}


@pytest.mark.phase7
def test_route_flood_safety_ignores_stale_request() -> None:
    """A dispatch to a request that's already gone (picked up/expired
    since the agent decided) isn't this rule's concern — `Simulator.step`
    itself already no-ops on a missing request."""
    graph = _ambulance_graph()
    rn = _road_network(susceptibility=1.0)  # would be unsafe if it existed
    action = AgentAction(ambulance_assignment={"AMB_1": "REQ_GONE"})
    report = check(graph, action, road_network=rn)
    assert report.passed
    assert report.repaired_action.ambulance_assignment == {"AMB_1": "REQ_GONE"}


# ---------------------------------------------------------------------------
# power_balance — disclosed no-op (see engine.py's module docstring)
# ---------------------------------------------------------------------------
@pytest.mark.phase7
def test_power_balance_is_a_disclosed_noop() -> None:
    graph = _hospital_graph()
    action = AgentAction(shed_tier={"S1": 3})
    report = check(graph, action)
    assert report.passed
    assert report.repaired_action.shed_tier == {"S1": 3}


# ---------------------------------------------------------------------------
# check() aggregation across rules
# ---------------------------------------------------------------------------
@pytest.mark.phase7
def test_check_aggregates_violations_from_multiple_rules() -> None:
    graph = _hospital_graph()
    for asset_id, attrs in (("AMB_1", {"status": "idle", "home_node": "home"}),):
        graph.add_node(asset_id, asset=Asset(
            asset_id=asset_id,
            asset_type=AssetType.AMBULANCE,
            geometry=_point(72.88, 19.07),
            attributes=attrs,
        ))
    graph.graph["pending_requests"] = [
        {"request_id": "REQ_1", "location": (0.001, 0.0), "requested_at_tick": 0}
    ]
    rn = _road_network(susceptibility=1.0)

    action = AgentAction(
        patient_transfer=("H1", "H2", 999),  # violates bed_capacity
        ambulance_assignment={"AMB_1": "REQ_1"},  # violates route_flood_safety
    )
    report = check(graph, action, road_network=rn)
    assert not report.passed
    assert {v.rule_id for v in report.violations} == {
        BED_CAPACITY_RULE_ID,
        ROUTE_FLOOD_SAFETY_RULE_ID,
    }
    assert report.repaired_action.patient_transfer == ("H1", "H2", 50)
    assert report.repaired_action.ambulance_assignment is None
