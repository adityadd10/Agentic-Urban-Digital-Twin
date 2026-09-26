"""Tests for the 2026-09-26 twin-realism revision (dev doc §3.8, §4.2, §4.3).

One test group per revised behaviour: supply competition, access gating,
the cascade metric, fragility sampling, footprint, casualty delivery, the
scenario generator's sampled parameters, and the Simulator's defensive copy.
Small hand-built graphs, same convention as `test_cascade.py`.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from udt.common.models import (
    AgentAction,
    Asset,
    AssetType,
    DependencyEdge,
    DependencyGraph,
    Incident,
)
from udt.incidents.degradations.flood import (
    FloodDegradation,
    footprint_weight,
    fragility_probability,
    sample_critical_depth,
)
from udt.scenarios.generator import apply_initial_conditions, generate_flood_scenario
from udt.twin.ambulances import advance_ambulances
from udt.twin.cascade import EdgeRuntimeState, allocation_ratios, resolve_functional_levels
from udt.twin.counterfactual import simulate
from udt.twin.graph import build_networkx_graph
from udt.twin.simulator import Simulator

POINT = {"type": "Point", "coordinates": [72.88, 19.07]}
WARD = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[72.86, 19.05], [72.90, 19.05], [72.90, 19.12], [72.86, 19.12], [72.86, 19.05]]
                ],
            },
        }
    ],
}


def _asset(asset_id: str, asset_type: AssetType, **attrs: object) -> Asset:
    return Asset(asset_id=asset_id, asset_type=asset_type, geometry=POINT, attributes=dict(attrs))


def _edge(
    edge_id: str, supplier: str, consumer: str, kind: str, demand: float, **kw: float
) -> DependencyEdge:
    return DependencyEdge(
        edge_id=edge_id,
        supplier=supplier,
        consumer=consumer,
        kind=kind,
        demand=demand,
        criticality=0.9,
        buffer_hours=kw.get("buffer_hours", 0.0),
        floor=kw.get("floor", 0.0),
    )


def _two_consumer_graph(load_mw: float = 0.0, shed_tier: int = 0) -> DependencyGraph:
    """S1 (capacity 10 MW) powers H1 and W1, each demanding 0.3 of it."""
    return DependencyGraph(
        assets=[
            _asset(
                "S1", AssetType.SUBSTATION, capacity_mw=10.0, load_mw=load_mw, shed_tier=shed_tier
            ),
            _asset("H1", AssetType.HOSPITAL, beds_total=10, beds_occupied=0),
            _asset("W1", AssetType.WATER),
        ],
        edges=[_edge("E1", "S1", "H1", "power", 0.3), _edge("E2", "S1", "W1", "power", 0.3)],
    )


def _levels(dep: DependencyGraph, s1_level: float) -> dict[str, float]:
    g = build_networkx_graph(dep)
    g.nodes["S1"]["asset"].intrinsic_level = s1_level
    g.nodes["S1"]["asset"].functional_level = s1_level
    states = {e.edge_id: EdgeRuntimeState.initial(e) for e in dep.edges}
    return resolve_functional_levels(g, states)


# --- supply competition (dev doc §3.8 item 3) ---------------------------------------


@pytest.mark.phase1
def test_consumers_share_a_degraded_supplier() -> None:
    # 0.3 + 0.3 = 0.6 demanded; S1 at 0.3 -> each gets half its demand.
    # (The old per-edge rule gave each consumer 0.3/0.3 = its full demand.)
    levels = _levels(_two_consumer_graph(), s1_level=0.3)
    assert levels["H1"] == pytest.approx(0.5)
    assert levels["W1"] == pytest.approx(0.5)


@pytest.mark.phase1
def test_background_load_competes_and_shedding_frees_capacity() -> None:
    # load 9 MW of 10: background = 0.9 - 0.6 = 0.3 -> total 0.9 of capacity.
    g = build_networkx_graph(_two_consumer_graph(load_mw=9.0))
    r = allocation_ratios(g, {"S1": 0.6, "H1": 1.0, "W1": 1.0})
    assert r["E1"] == pytest.approx(0.6 / 0.9)
    # tier 3 sheds 60% of the background -> 0.12, total 0.72
    g_shed = build_networkx_graph(_two_consumer_graph(load_mw=9.0, shed_tier=3))
    r_shed = allocation_ratios(g_shed, {"S1": 0.6, "H1": 1.0, "W1": 1.0})
    assert r_shed["E1"] == pytest.approx(0.6 / 0.72)
    assert r_shed["E1"] > r["E1"]


@pytest.mark.phase1
def test_access_edges_do_not_gate_substation_output() -> None:
    dep = DependencyGraph(
        assets=[
            _asset("S1", AssetType.SUBSTATION, capacity_mw=10.0, load_mw=0.0),
            _asset("R1", AssetType.ROAD, blockage=1.0),
        ],
        edges=[_edge("A1", "R1", "S1", "access", 1.0, floor=0.5)],
    )
    g = build_networkx_graph(dep)
    states = {e.edge_id: EdgeRuntimeState.initial(e) for e in dep.edges}
    assert resolve_functional_levels(g, states)["S1"] == pytest.approx(1.0)


# --- cascade metric (dev doc §3.5, revised) -----------------------------------------


@pytest.mark.phase1
def test_cascade_counts_only_dependency_caused_facility_failures() -> None:
    dep = DependencyGraph(
        assets=[
            _asset("S1", AssetType.SUBSTATION, capacity_mw=10.0, load_mw=0.0),
            _asset("H1", AssetType.HOSPITAL, beds_total=10, beds_occupied=0),
            _asset("R1", AssetType.ROAD, blockage=1.0),
        ],
        edges=[_edge("E1", "S1", "H1", "power", 0.5, floor=0.0)],
    )
    sim = Simulator(dep)
    sim.apply_degradation("S1", 1.0)  # S1 destroyed directly
    snap = sim.step()
    assert sim._ever_cascaded == {"H1"}  # not S1 (own damage), not R1 (a road)
    assert snap.cascading_failure_count == 1


# --- fragility (dev doc §3.8 item 4) ------------------------------------------------


@pytest.mark.phase3
def test_substation_fragility_matches_lab_table() -> None:
    for depth, p in [(0.1, 0.333), (0.3, 0.67), (0.5, 0.968), (0.6, 1.0)]:
        assert fragility_probability(AssetType.SUBSTATION, depth) == pytest.approx(p)
    assert fragility_probability(AssetType.HOSPITAL, 0.6) == pytest.approx(0.5)
    # twin-v2: no failures below the source's 0.1 m starting depth
    assert fragility_probability(AssetType.SUBSTATION, 0.05) == 0.0
    rng = np.random.default_rng(0)
    draws = [sample_critical_depth(AssetType.SUBSTATION, rng) for _ in range(500)]
    assert min(draws) == pytest.approx(0.1)
    assert 0.25 < np.mean(np.isclose(draws, 0.1)) < 0.42  # ~1/3 fail right at 0.1 m


@pytest.mark.phase3
def test_critical_depth_sampling_is_conditional_on_survival() -> None:
    rng = np.random.default_rng(0)
    samples = [
        sample_critical_depth(AssetType.SUBSTATION, rng, survived_depth_m=0.3) for _ in range(200)
    ]
    assert min(samples) > 0.3


@pytest.mark.phase3
def test_critical_depths_are_reproducible_and_seed_dependent() -> None:
    incident = Incident(incident_id="i", type="flood", location=POINT, onset_tick=0, severity=1.0)
    asset = _asset("S1", AssetType.SUBSTATION)
    a = FloodDegradation(incident, None, fragility_seed=1).critical_depth(asset)  # type: ignore[arg-type]
    b = FloodDegradation(incident, None, fragility_seed=1).critical_depth(asset)  # type: ignore[arg-type]
    draws = {
        FloodDegradation(incident, None, fragility_seed=s).critical_depth(asset)  # type: ignore[arg-type]
        for s in range(20)
    }
    assert a == b
    assert len(draws) > 10


class _Raster:
    def __init__(self, value: float) -> None:
        self.value = value

    def value_at(self, lon: float, lat: float) -> float:
        return self.value


@pytest.mark.phase8
def test_counterfactual_rollouts_disagree_under_fragility_uncertainty() -> None:
    """With depth near a substation's median critical depth, resampled
    fragility makes some rollouts fail and some not: P(failure) is no
    longer forced to 0 or 1 (dev doc §3.8 item 5)."""
    incident = Incident(
        incident_id="i",
        type="flood",
        location=POINT,
        onset_tick=0,
        severity=1.0,
        profile={"growth_hours": 0.1, "hold_hours": 24.0, "recede_hours": 1.0},
    )
    # depth = 1.0 x 0.1 x 2.0 = 0.2 m ~ substation median
    fn = FloodDegradation(incident, _Raster(0.1))  # type: ignore[arg-type]
    dep = DependencyGraph(
        assets=[
            _asset("S1", AssetType.SUBSTATION, capacity_mw=10.0, load_mw=0.0),
            _asset("H1", AssetType.HOSPITAL, beds_total=10, beds_occupied=0),
        ],
        edges=[_edge("E1", "S1", "H1", "power", 0.5, floor=0.0)],
    )
    sim = Simulator(dep)
    result = simulate(sim, AgentAction(), degradation_fn=fn, horizon_ticks=6, n_rollouts=20)
    assert 0.0 < result.p_failure < 1.0


# --- footprint (dev doc §4.2) -------------------------------------------------------


@pytest.mark.phase3
def test_footprint_is_one_at_centroid_and_decays() -> None:
    incident = Incident(
        incident_id="i",
        type="flood",
        location=POINT,
        onset_tick=0,
        severity=1.0,
        profile={"footprint": {"lon": 72.88, "lat": 19.07, "sigma_m": 1000.0}},
    )
    assert footprint_weight(incident, 72.88, 19.07) == pytest.approx(1.0)
    near = footprint_weight(incident, 72.88, 19.07 + 1000 / 110_540)  # 1 km north
    far = footprint_weight(incident, 72.88, 19.07 + 3000 / 110_540)  # 3 km north
    assert near == pytest.approx(np.exp(-0.5), rel=1e-3)
    assert far < near < 1.0


# --- casualty delivery (dev doc §3.8 item 6) ----------------------------------------


@pytest.mark.phase4
def test_picked_up_casualty_joins_home_hospital_queue() -> None:
    g: nx.DiGraph[str] = nx.DiGraph()
    g.add_node("H1", asset=_asset("H1", AssetType.HOSPITAL, beds_total=10, beds_occupied=0))
    amb = _asset(
        "AMB",
        AssetType.AMBULANCE,
        status="enroute",
        home_hospital_id="H1",
        assigned_request_id="REQ",
        remaining_travel_min=0.0,
        return_travel_min=5.0,
    )
    g.add_node("AMB", asset=amb)
    g.graph["pending_requests"] = [
        {"request_id": "REQ", "location": (0, 0), "requested_at_tick": 0}
    ]
    advance_ambulances(g, tick=1, dt_minutes=5.0)  # pickup
    assert amb.attributes["carrying_patient"] is True
    advance_ambulances(g, tick=2, dt_minutes=5.0)  # return leg completes
    assert g.nodes["H1"]["asset"].attributes["queue_arrivals"] == [2]
    assert g.nodes["H1"]["asset"].attributes["patient_queue"] == 1


# --- scenario generator (dev doc §4.3) ----------------------------------------------


@pytest.mark.phase3
def test_scenarios_differ_in_footprint_timing_and_severity() -> None:
    scenarios = [
        generate_flood_scenario(scenario_id=f"s{i}", ward_boundary_geojson=WARD, seed=i)
        for i in range(10)
    ]
    centroids = {
        (s.incident.profile["footprint"]["lon"], s.incident.profile["footprint"]["lat"])
        for s in scenarios
    }
    assert len(centroids) == 10
    for s in scenarios:
        p = s.incident.profile
        assert 0.2 <= s.incident.severity <= 1.0
        assert 800 <= p["footprint"]["sigma_m"] <= 2500
        assert 1 <= p["growth_hours"] <= 3 and 4 <= p["hold_hours"] <= 10
        assert 0 <= s.initial_conditions["onset_hour_of_day"] < 24


@pytest.mark.phase3
def test_apply_initial_conditions_sets_bed_occupancy_without_mutating_input() -> None:
    dep = DependencyGraph(
        assets=[_asset("H1", AssetType.HOSPITAL, beds_total=200, beds_occupied=0)], edges=[]
    )
    scenario = generate_flood_scenario(scenario_id="s", ward_boundary_geojson=WARD, seed=3)
    out = apply_initial_conditions(dep, scenario)
    occupied = out.assets[0].attributes["beds_occupied"]
    assert 120 <= occupied <= 180
    assert dep.assets[0].attributes["beds_occupied"] == 0


# --- defensive copy (dev doc §3.8 item 7) -------------------------------------------


@pytest.mark.phase1
def test_simulators_built_from_one_graph_do_not_share_damage() -> None:
    dep = DependencyGraph(assets=[_asset("S1", AssetType.SUBSTATION)], edges=[])
    first = Simulator(dep)
    first.apply_degradation("S1", 0.7)
    second = Simulator(dep)
    assert second.asset("S1").intrinsic_level == pytest.approx(1.0)
    assert dep.assets[0].intrinsic_level == pytest.approx(1.0)
