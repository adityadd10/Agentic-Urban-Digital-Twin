"""Phase 4 acceptance tests: `twin.ambulances` (dev doc §3.2 Ambulance
attributes, §3.5 step 4's "ambulances advance along routes", §5.6's
dispatch rule)."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from udt.common.models import Asset, AssetType, DependencyGraph, Incident
from udt.twin import ambulances
from udt.twin.ambulances import (
    add_transfer_requests,
    advance_ambulances,
    dispatch_ambulance,
    expire_uncollected_requests,
    generate_requests,
    spawn_ambulances,
)
from udt.twin.graph import build_networkx_graph
from udt.twin.road_network import RoadNetwork
from udt.twin.simulator import Simulator

POINT = {"type": "Point", "coordinates": [72.88, 19.07]}

FLOOD_INCIDENT = Incident(
    incident_id="amb_test",
    type="flood",
    location=POINT,
    onset_tick=0,
    severity=0.0,  # no flooding -> every route stays passable, isolates the mechanic under test
    directly_affected_assets=[],
)


def _road_network_chain(n_edges: int, length_m: float = 300.0) -> RoadNetwork:
    g = nx.DiGraph()
    for i in range(n_edges + 1):
        g.add_node(str(i), x=72.88 + i * 0.001, y=19.07)
    for i in range(n_edges):
        g.add_edge(str(i), str(i + 1), length=length_m, susceptibility=0.0)
        g.add_edge(str(i + 1), str(i), length=length_m, susceptibility=0.0)
    node_ids = list(g.nodes)
    node_xy = np.array([[float(g.nodes[n]["x"]), float(g.nodes[n]["y"])] for n in node_ids])
    rn = RoadNetwork(g, node_ids, node_xy)
    rn.update_for_tick(FLOOD_INCIDENT, tick=0, dt_minutes=5.0)
    return rn


def _hospital_graph() -> nx.DiGraph:
    hospital = Asset(
        asset_id="H1",
        asset_type=AssetType.HOSPITAL,
        geometry={"type": "Point", "coordinates": [72.88, 19.07]},  # matches node "0"
        attributes={},
    )
    return build_networkx_graph(DependencyGraph(assets=[hospital], edges=[]))


@pytest.mark.phase4
def test_spawn_ambulances_creates_idle_ambulances_at_home_node() -> None:
    graph = _hospital_graph()
    rn = _road_network_chain(2)
    ids = spawn_ambulances(graph, rn, n_per_hospital=2)
    assert len(ids) == 2
    for amb_id in ids:
        amb = graph.nodes[amb_id]["asset"]
        assert amb.asset_type == AssetType.AMBULANCE
        assert amb.attributes["status"] == "idle"
        assert amb.attributes["home_node"] == "0"  # nearest node to H1's coordinates


@pytest.mark.phase4
def test_generate_requests_stays_inside_polygon() -> None:
    graph = _hospital_graph()
    polygon = Polygon([(72.87, 19.06), (72.89, 19.06), (72.89, 19.08), (72.87, 19.08)])
    incident = FLOOD_INCIDENT.model_copy(update={"severity": 1.0})
    rng = np.random.default_rng(0)
    for tick in range(20):  # enough ticks that at least one request is virtually certain
        generate_requests(
            graph, tick, dt_hours=5.0 / 60.0, incident=incident, ward_polygon=polygon, rng=rng
        )
    pending = graph.graph.get("pending_requests", [])
    assert len(pending) > 0
    for request in pending:
        lon, lat = request["location"]
        assert polygon.contains(Point(lon, lat))


@pytest.mark.phase4
def test_dispatch_ambulance_sets_enroute_with_both_legs_computed() -> None:
    graph = _hospital_graph()
    rn = _road_network_chain(2)
    (amb_id,) = spawn_ambulances(graph, rn, n_per_hospital=1)
    request = {"request_id": "REQ1", "location": (72.882, 19.07), "requested_at_tick": 0}

    ok = dispatch_ambulance(graph, rn, amb_id, request)
    assert ok is True
    amb = graph.nodes[amb_id]["asset"]
    assert amb.attributes["status"] == "enroute"
    assert amb.attributes["assigned_request_id"] == "REQ1"
    assert amb.attributes["remaining_travel_min"] > 0
    assert amb.attributes["return_travel_min"] > 0


@pytest.mark.phase4
def test_dispatch_ambulance_fails_when_unreachable() -> None:
    graph = _hospital_graph()
    rn = _road_network_chain(1)
    rn.graph.add_node("island", x=99.0, y=99.0)  # disconnected from everything
    rn._node_ids.append("island")
    rn._node_xy = np.vstack([rn._node_xy, [[99.0, 99.0]]])
    (amb_id,) = spawn_ambulances(graph, rn, n_per_hospital=1)
    request = {"request_id": "REQ1", "location": (99.0, 99.0), "requested_at_tick": 0}

    ok = dispatch_ambulance(graph, rn, amb_id, request)
    assert ok is False
    amb = graph.nodes[amb_id]["asset"]
    assert amb.attributes["status"] == "idle"


@pytest.mark.phase4
def test_advance_ambulances_completes_pickup_then_returns_to_idle() -> None:
    graph = _hospital_graph()
    rn = _road_network_chain(2)
    (amb_id,) = spawn_ambulances(graph, rn, n_per_hospital=1)
    request = {"request_id": "REQ1", "location": (72.882, 19.07), "requested_at_tick": 0}
    graph.graph["pending_requests"] = [request]
    dispatch_ambulance(graph, rn, amb_id, request)

    amb = graph.nodes[amb_id]["asset"]
    # Force both legs to complete within one tick each, regardless of the
    # real computed travel time, to test the state machine in isolation.
    amb.attributes["remaining_travel_min"] = 1.0
    amb.attributes["return_travel_min"] = 1.0

    response_times = advance_ambulances(graph, tick=3, dt_minutes=5.0)
    assert response_times == [3 * (5.0 / 60.0)]
    assert amb.attributes["status"] == "returning"
    assert graph.graph["pending_requests"] == []  # request consumed on pickup

    response_times_2 = advance_ambulances(graph, tick=4, dt_minutes=5.0)
    assert response_times_2 == []  # no *new* pickup this tick
    assert amb.attributes["status"] == "idle"
    assert amb.attributes["assigned_request_id"] is None


@pytest.mark.phase4
def test_advance_ambulances_idle_ambulance_is_untouched() -> None:
    graph = _hospital_graph()
    rn = _road_network_chain(1)
    (amb_id,) = spawn_ambulances(graph, rn, n_per_hospital=1)
    response_times = advance_ambulances(graph, tick=5, dt_minutes=5.0)
    assert response_times == []
    assert graph.nodes[amb_id]["asset"].attributes["status"] == "idle"


# --- Uncollected-casualty deaths (dev doc §3.9 item 2, metric-v2) ---------

DT_HOURS = 5.0 / 60.0  # 4 h deadline = 48 ticks


@pytest.mark.phase4
def test_expire_uncollected_requests_uses_the_hospital_queue_deadline() -> None:
    graph = _hospital_graph()
    graph.graph["pending_requests"] = [
        {"request_id": "OLD", "location": (72.882, 19.07), "requested_at_tick": 0},
        {"request_id": "NEW", "location": (72.882, 19.07), "requested_at_tick": 10},
    ]
    assert expire_uncollected_requests(graph, tick=48, dt_hours=DT_HOURS) == 0  # exactly 4 h
    assert expire_uncollected_requests(graph, tick=49, dt_hours=DT_HOURS) == 1
    assert [r["request_id"] for r in graph.graph["pending_requests"]] == ["NEW"]


@pytest.mark.phase4
def test_ambulance_sent_to_an_expired_request_returns_empty() -> None:
    graph = _hospital_graph()
    rn = _road_network_chain(2)
    (amb_id,) = spawn_ambulances(graph, rn, n_per_hospital=1)
    request = {"request_id": "REQ1", "location": (72.882, 19.07), "requested_at_tick": 0}
    graph.graph["pending_requests"] = [request]
    dispatch_ambulance(graph, rn, amb_id, request)
    expire_uncollected_requests(graph, tick=49, dt_hours=DT_HOURS)
    amb = graph.nodes[amb_id]["asset"]
    amb.attributes["remaining_travel_min"] = 1.0
    assert advance_ambulances(graph, tick=49, dt_minutes=5.0) == []  # no pickup
    assert amb.attributes["status"] == "returning"
    assert not amb.attributes.get("carrying_patient")


def _simulator_with_one_unanswered_call() -> Simulator:
    hospital = Asset(
        asset_id="H1",
        asset_type=AssetType.HOSPITAL,
        geometry={"type": "Point", "coordinates": [72.88, 19.07]},
        attributes={},
    )
    sim = Simulator(
        DependencyGraph(assets=[hospital], edges=[]), road_network=_road_network_chain(2)
    )
    sim.graph.graph["pending_requests"] = [
        {"request_id": "R1", "location": (72.882, 19.07), "requested_at_tick": 0}
    ]
    return sim


@pytest.mark.phase4
def test_unanswered_call_is_not_a_death_when_the_switch_is_off() -> None:
    assert ambulances.COUNT_UNCOLLECTED_CASUALTY_DEATHS is False  # twin-v2 default
    sim = _simulator_with_one_unanswered_call()
    for _ in range(60):
        state = sim.step()
    assert state.pending_requests_count == 1
    assert state.uncollected_casualty_deaths_cumulative == 0


@pytest.mark.phase4
def test_unanswered_call_counts_as_a_death_when_the_switch_is_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sim = _simulator_with_one_unanswered_call()
    monkeypatch.setattr(ambulances, "COUNT_UNCOLLECTED_CASUALTY_DEATHS", True)
    deaths_before_deadline = None
    for _ in range(60):
        state = sim.step()
        if state.tick == 48:
            deaths_before_deadline = state.uncollected_casualty_deaths_cumulative
    assert deaths_before_deadline == 0
    assert state.pending_requests_count == 0
    assert state.uncollected_casualty_deaths_cumulative == 1
    assert state.patient_deaths_cumulative >= 1  # included in the death proxy total


@pytest.mark.phase4
def test_rule_v2_dispatches_every_idle_ambulance_and_skips_unreachable_calls() -> None:
    from udt.agents.rule_based import RuleBasedAgent, RuleBasedAgentV2

    graph = _hospital_graph()
    rn = _road_network_chain(3)
    rn.graph.add_node("island", x=99.0, y=99.0)  # unreachable call location
    rn._node_ids.append("island")
    rn._node_xy = np.vstack([rn._node_xy, [[99.0, 99.0]]])
    spawn_ambulances(graph, rn, n_per_hospital=2)
    graph.graph["pending_requests"] = [
        {"request_id": "CUT_OFF", "location": (99.0, 99.0), "requested_at_tick": 0},
        {"request_id": "A", "location": (72.882, 19.07), "requested_at_tick": 1},
        {"request_id": "B", "location": (72.883, 19.07), "requested_at_tick": 2},
    ]
    # v1: head-of-line blocking; the oldest call is unreachable, so nothing moves.
    assert RuleBasedAgent().act(graph, tick=3, road_network=rn).ambulance_assignment is None
    # v2: both idle ambulances go, to the two reachable calls.
    assignment = RuleBasedAgentV2().act(graph, tick=3, road_network=rn).ambulance_assignment
    assert assignment is not None
    assert sorted(assignment.values()) == ["A", "B"]


# --- Destination choice (dev doc §3.9 mechanic 1) -----------------------------


def _two_hospital_graph() -> nx.DiGraph:
    """H1 at node "0", H2 at node "3" of a 3-edge chain."""
    h1 = Asset(
        asset_id="H1",
        asset_type=AssetType.HOSPITAL,
        geometry={"type": "Point", "coordinates": [72.88, 19.07]},
        attributes={},
    )
    h2 = Asset(
        asset_id="H2",
        asset_type=AssetType.HOSPITAL,
        geometry={"type": "Point", "coordinates": [72.883, 19.07]},
        attributes={},
    )
    return build_networkx_graph(DependencyGraph(assets=[h1, h2], edges=[]))


def _complete_leg(graph: nx.DiGraph, amb_id: str, tick: int) -> list[float]:
    graph.nodes[amb_id]["asset"].attributes["remaining_travel_min"] = 1.0
    return advance_ambulances(graph, tick=tick, dt_minutes=5.0)


@pytest.mark.phase4
def test_destination_home_is_identical_to_no_destination() -> None:
    rn = _road_network_chain(3)
    states = []
    for destination in (None, "H1"):
        graph = _two_hospital_graph()
        spawn_ambulances(graph, rn, n_per_hospital=1)
        request = {"request_id": "R", "location": (72.882, 19.07), "requested_at_tick": 0}
        assert dispatch_ambulance(graph, rn, "AMB_H1_0", request, destination)
        states.append(dict(graph.nodes["AMB_H1_0"]["asset"].attributes))
    assert states[0] == states[1]
    assert "delivery_hospital_id" not in states[0]


@pytest.mark.phase4
def test_casualty_is_delivered_to_the_chosen_hospital_then_ambulance_returns_home() -> None:
    graph = _two_hospital_graph()
    rn = _road_network_chain(3)
    spawn_ambulances(graph, rn, n_per_hospital=1)
    amb = "AMB_H1_0"
    request = {"request_id": "R", "location": (72.882, 19.07), "requested_at_tick": 0}
    graph.graph["pending_requests"] = [request]
    assert dispatch_ambulance(graph, rn, amb, request, "H2")
    attrs = graph.nodes[amb]["asset"].attributes
    assert attrs["delivery_hospital_id"] == "H2" and attrs["final_return_min"] > 0

    _complete_leg(graph, amb, tick=1)  # pickup
    assert attrs["carrying_patient"] is True
    _complete_leg(graph, amb, tick=2)  # arrive at H2, deliver
    assert graph.nodes["H2"]["asset"].attributes["queue_arrivals"] == [2]
    assert "queue_arrivals" not in graph.nodes["H1"]["asset"].attributes
    assert attrs["status"] == "returning" and not attrs["carrying_patient"]
    _complete_leg(graph, amb, tick=3)  # empty leg back home
    assert attrs["status"] == "idle"
    assert "delivery_hospital_id" not in attrs and "final_return_min" not in attrs


@pytest.mark.phase4
def test_destination_must_be_a_hospital() -> None:
    graph = _two_hospital_graph()
    rn = _road_network_chain(3)
    spawn_ambulances(graph, rn, n_per_hospital=1)
    request = {"request_id": "R", "location": (72.882, 19.07), "requested_at_tick": 0}
    with pytest.raises(ValueError):
        dispatch_ambulance(graph, rn, "AMB_H1_0", request, "AMB_H2_0")


@pytest.mark.phase4
def test_simulator_passes_the_destination_through() -> None:
    h1 = Asset(
        asset_id="H1",
        asset_type=AssetType.HOSPITAL,
        geometry={"type": "Point", "coordinates": [72.88, 19.07]},
        attributes={},
    )
    h2 = h1.model_copy(
        update={"asset_id": "H2", "geometry": {"type": "Point", "coordinates": [72.883, 19.07]}}
    )
    rn = _road_network_chain(3)
    sim = Simulator(DependencyGraph(assets=[h1, h2], edges=[]), road_network=rn)
    spawn_ambulances(sim.graph, rn, n_per_hospital=1)
    request = {"request_id": "R", "location": (72.882, 19.07), "requested_at_tick": 0}
    sim.graph.graph["pending_requests"] = [request]
    sim.step(ambulance_assignment={"AMB_H1_0": "R"}, ambulance_destination={"AMB_H1_0": "H2"})
    attrs = sim.graph.nodes["AMB_H1_0"]["asset"].attributes
    assert attrs["delivery_hospital_id"] == "H2"


# --- Transfer requests (dev doc §3.9 mechanic 3) -------------------------------


def _transfer_setup(queue: list[int], beds_occupied: int = 5) -> tuple[nx.DiGraph, RoadNetwork]:
    graph = _two_hospital_graph()
    rn = _road_network_chain(3)
    spawn_ambulances(graph, rn, n_per_hospital=1)
    h1 = graph.nodes["H1"]["asset"].attributes
    h1.update(
        {"queue_arrivals": queue, "patient_queue": len(queue), "beds_occupied": beds_occupied}
    )
    return graph, rn


@pytest.mark.phase4
def test_transfer_requests_become_one_job_per_patient_at_the_source() -> None:
    graph, _ = _transfer_setup([])
    assert add_transfer_requests(graph, tick=4, requests=[("H1", 3, "urgent")]) == 3
    jobs = graph.graph["transfer_requests"]
    assert len(jobs) == 3 and len({j["request_id"] for j in jobs}) == 3
    assert all(j["from_hospital_id"] == "H1" and j["urgency"] == "urgent" for j in jobs)
    assert jobs[0]["location"] == (72.88, 19.07)
    with pytest.raises(ValueError):
        add_transfer_requests(graph, tick=4, requests=[("AMB_H1_0", 1, "routine")])


@pytest.mark.phase4
def test_transfer_moves_the_longest_waiting_patient_and_keeps_their_wait_start() -> None:
    graph, rn = _transfer_setup(queue=[2, 5])
    add_transfer_requests(graph, tick=6, requests=[("H1", 1, "urgent")])
    job = graph.graph["transfer_requests"][0]
    amb = "AMB_H2_0"  # stationed at H2, collects from H1, brings back to H2
    assert dispatch_ambulance(graph, rn, amb, job, "H2")
    assert _complete_leg(graph, amb, tick=7) == []  # pickup: no response time for transfers
    h1 = graph.nodes["H1"]["asset"].attributes
    assert h1["queue_arrivals"] == [5] and graph.graph["transfer_requests"] == []
    _complete_leg(graph, amb, tick=8)  # delivered at H2 (home)
    assert graph.nodes["H2"]["asset"].attributes["queue_arrivals"] == [2]  # original wait start
    assert graph.graph["transfers_completed"] == 1


@pytest.mark.phase4
def test_transfer_takes_an_admitted_patient_when_nobody_is_queued() -> None:
    graph, rn = _transfer_setup(queue=[], beds_occupied=5)
    add_transfer_requests(graph, tick=6, requests=[("H1", 1, "routine")])
    job = graph.graph["transfer_requests"][0]
    assert dispatch_ambulance(graph, rn, "AMB_H2_0", job, "H2")
    _complete_leg(graph, "AMB_H2_0", tick=7)
    assert graph.nodes["H1"]["asset"].attributes["beds_occupied"] == 4
    _complete_leg(graph, "AMB_H2_0", tick=8)
    assert graph.nodes["H2"]["asset"].attributes["queue_arrivals"] == [8]  # delivery tick


@pytest.mark.phase4
def test_simulator_creates_and_serves_transfer_jobs() -> None:
    h1 = Asset(
        asset_id="H1",
        asset_type=AssetType.HOSPITAL,
        geometry={"type": "Point", "coordinates": [72.88, 19.07]},
        attributes={"beds_total": 10, "beds_occupied": 10, "queue_arrivals": [0]},
    )
    h2 = h1.model_copy(
        update={
            "asset_id": "H2",
            "geometry": {"type": "Point", "coordinates": [72.883, 19.07]},
            "attributes": {"beds_total": 10, "beds_occupied": 0, "queue_arrivals": []},
        }
    )
    rn = _road_network_chain(3)
    sim = Simulator(DependencyGraph(assets=[h1, h2], edges=[]), road_network=rn)
    spawn_ambulances(sim.graph, rn, n_per_hospital=1)
    state = sim.step(transfer_requests=[("H1", 1, "urgent")])
    assert state.pending_transfers_count == 1
    job_id = sim.graph.graph["transfer_requests"][0]["request_id"]
    sim.step(ambulance_assignment={"AMB_H1_0": job_id}, ambulance_destination={"AMB_H1_0": "H2"})
    assert sim.graph.nodes["AMB_H1_0"]["asset"].attributes["assigned_request_id"] == job_id


@pytest.mark.phase4
def test_route_safety_check_covers_transfer_jobs() -> None:
    from udt.common.models import AgentAction
    from udt.constraints.engine import check

    graph, rn = _transfer_setup([])
    rn.graph.add_node("island", x=99.0, y=99.0)
    rn._node_ids.append("island")
    rn._node_xy = np.vstack([rn._node_xy, [[99.0, 99.0]]])
    graph.graph["transfer_requests"] = [
        {
            "request_id": "TR_X",
            "kind": "transfer",
            "from_hospital_id": "H1",
            "location": (99.0, 99.0),
            "requested_at_tick": 0,
            "urgency": "urgent",
        }
    ]
    report = check(graph, AgentAction(ambulance_assignment={"AMB_H1_0": "TR_X"}), road_network=rn)
    assert report.repaired_action.ambulance_assignment is None  # dropped, not waved through


# --- Shared fleet metrics (dev doc §3.9 mechanic 4) ----------------------------


@pytest.mark.phase4
def test_casualty_time_to_admission_is_call_to_bed() -> None:
    from udt.twin.demand import CASUALTY_OUTCOMES_KEY, consume_demand

    graph = _two_hospital_graph()
    rn = _road_network_chain(3)
    spawn_ambulances(graph, rn, n_per_hospital=1)
    graph.nodes["H1"]["asset"].attributes.update({"beds_total": 10, "beds_occupied": 0})
    request = {"request_id": "R", "location": (72.882, 19.07), "requested_at_tick": 0}
    graph.graph["pending_requests"] = [request]
    dispatch_ambulance(graph, rn, "AMB_H1_0", request)
    _complete_leg(graph, "AMB_H1_0", tick=6)  # pickup
    _complete_leg(graph, "AMB_H1_0", tick=12)  # delivered to H1's queue
    assert graph.nodes["H1"]["asset"].attributes["queue_call_ticks"] == [0]
    consume_demand(graph, tick=13, dt_hours=5.0 / 60.0, rng=np.random.default_rng(0))
    assert graph.graph[CASUALTY_OUTCOMES_KEY] == [pytest.approx(13 * 5.0 / 60.0)]


@pytest.mark.phase4
def test_casualty_who_dies_waiting_is_recorded_at_death_and_walk_ins_are_not() -> None:
    from udt.twin.demand import CASUALTY_OUTCOMES_KEY, consume_demand

    graph = _two_hospital_graph()
    h1 = graph.nodes["H1"]["asset"].attributes
    # full hospital: one casualty (call at tick 2) and one walk-in, both past the deadline
    h1.update(
        {
            "beds_total": 1,
            "beds_occupied": 1,
            "queue_arrivals": [3, 3],
            "queue_call_ticks": [2, None],
        }
    )
    consume_demand(graph, tick=60, dt_hours=5.0 / 60.0, rng=np.random.default_rng(1))
    assert graph.graph[CASUALTY_OUTCOMES_KEY] == [pytest.approx(58 * 5.0 / 60.0)]


@pytest.mark.phase4
def test_uncollected_casualty_is_recorded_at_expiry() -> None:
    from udt.twin.demand import CASUALTY_OUTCOMES_KEY

    graph = _hospital_graph()
    graph.graph["pending_requests"] = [
        {"request_id": "OLD", "location": (72.882, 19.07), "requested_at_tick": 0}
    ]
    expire_uncollected_requests(graph, tick=49, dt_hours=DT_HOURS)
    assert graph.graph[CASUALTY_OUTCOMES_KEY] == [pytest.approx(49 * DT_HOURS)]


@pytest.mark.phase4
def test_fleet_contention_needs_calls_transfers_and_too_few_idle_ambulances() -> None:
    h1 = Asset(
        asset_id="H1",
        asset_type=AssetType.HOSPITAL,
        geometry={"type": "Point", "coordinates": [72.88, 19.07]},
        attributes={"beds_total": 10, "beds_occupied": 10, "queue_arrivals": [0, 0]},
    )
    rn = _road_network_chain(3)
    sim = Simulator(DependencyGraph(assets=[h1], edges=[]), road_network=rn)
    spawn_ambulances(sim.graph, rn, n_per_hospital=1)  # one idle ambulance
    sim.graph.graph["pending_requests"] = [
        {"request_id": "R", "location": (72.882, 19.07), "requested_at_tick": 0}
    ]
    assert sim.step().fleet_contention is False  # a call, no transfer
    state = sim.step(transfer_requests=[("H1", 1, "urgent")])
    assert state.fleet_contention is True  # 1 idle ambulance, 2 jobs of both kinds


@pytest.mark.phase4
def test_episode_metrics_aggregate_the_new_fleet_metrics() -> None:
    from udt.common.models import TwinState
    from udt.logging.metrics import compute_episode_metrics

    trace = [
        TwinState(tick=0, assets=[], fleet_contention=True, casualty_outcome_hours_this_tick=[1.0]),
        TwinState(
            tick=1,
            assets=[],
            ambulance_response_times_this_tick=[0.5],
            transfers_completed_cumulative=2,
            casualty_outcome_hours_this_tick=[3.0],
        ),
    ]
    m = compute_episode_metrics("s", "a", trace)
    assert m.mean_casualty_time_to_admission_hours == pytest.approx(2.0)
    assert m.fleet_contention_fraction == pytest.approx(0.5)
    assert m.jobs_completed == 3
