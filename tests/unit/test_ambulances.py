"""Phase 4 acceptance tests: `twin.ambulances` (dev doc §3.2 Ambulance
attributes, §3.5 step 4's "ambulances advance along routes", §5.6's
dispatch rule)."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from udt.common.models import Asset, AssetType, DependencyGraph, Incident
from udt.twin.ambulances import (
    advance_ambulances,
    dispatch_ambulance,
    generate_requests,
    spawn_ambulances,
)
from udt.twin.graph import build_networkx_graph
from udt.twin.road_network import RoadNetwork

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
