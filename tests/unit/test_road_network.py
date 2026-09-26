"""Phase 4 acceptance tests: `twin.road_network.RoadNetwork` (dev doc
§3.3's "recomputed by Dijkstra", never implemented until this module)."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from udt.common.models import Incident
from udt.incidents.degradations.flood import MAX_DEPTH_AT_SEVERITY_1_M
from udt.twin.road_network import RoadNetwork

FLOOD_INCIDENT = Incident(
    incident_id="road_net_test",
    type="flood",
    location={"type": "Point", "coordinates": [72.88, 19.07]},
    onset_tick=0,
    severity=1.0,
    directly_affected_assets=[],
)


def _linear_graph(n_edges: int, length_m: float = 300.0, susceptibility: float = 0.0) -> nx.DiGraph:
    """A -> B -> C -> ... a straight chain, each edge `length_m` long, all
    with the same disclosed `susceptibility` (0.0 = never floods)."""
    g = nx.DiGraph()
    for i in range(n_edges + 1):
        g.add_node(str(i), x=float(i) * 0.001, y=0.0)
    for i in range(n_edges):
        g.add_edge(str(i), str(i + 1), length=length_m, susceptibility=susceptibility)
    return g


def _road_network(g: nx.DiGraph) -> RoadNetwork:
    node_ids = list(g.nodes)
    node_xy = np.array([[float(g.nodes[n]["x"]), float(g.nodes[n]["y"])] for n in node_ids])
    return RoadNetwork(g, node_ids, node_xy)


@pytest.mark.phase4
def test_nearest_node_finds_closest_by_coordinates() -> None:
    rn = _road_network(_linear_graph(3))
    assert rn.nearest_node(0.0021, 0.0) == "2"


@pytest.mark.phase4
def test_travel_time_scales_with_route_length() -> None:
    rn = _road_network(_linear_graph(3, length_m=300.0))
    rn.update_for_tick(FLOOD_INCIDENT, tick=0, dt_minutes=5.0)
    one_hop = rn.travel_time_minutes("0", "1")
    three_hops = rn.travel_time_minutes("0", "3")
    assert one_hop is not None and three_hops is not None
    assert three_hops == pytest.approx(one_hop * 3, rel=1e-6)


@pytest.mark.phase4
def test_flooded_edge_increases_travel_time_but_stays_routable() -> None:
    g = _linear_graph(2, length_m=300.0, susceptibility=0.0)
    # Flood only the first edge moderately (susceptibility 0.2, at full
    # incident severity, mid-flood so the temporal envelope is 1.0) —
    # still passable, just slower, per dev doc §3.3.
    g["0"]["1"]["susceptibility"] = 0.2
    rn = _road_network(g)
    rn.update_for_tick(FLOOD_INCIDENT, tick=60, dt_minutes=5.0)  # 5h in -> within the hold window
    flooded = rn.travel_time_minutes("0", "1")
    dry = rn.travel_time_minutes("1", "2")
    assert flooded is not None and dry is not None
    assert flooded > dry


@pytest.mark.phase4
def test_fully_blocked_edge_has_no_route() -> None:
    rn = _road_network(_linear_graph(1, length_m=300.0, susceptibility=1.0))
    rn.update_for_tick(FLOOD_INCIDENT, tick=60, dt_minutes=5.0)  # hold window -> envelope = 1.0
    assert rn.travel_time_minutes("0", "1") is None


@pytest.mark.phase4
def test_no_route_between_disconnected_nodes() -> None:
    g = _linear_graph(1)
    g.add_node("island", x=1.0, y=1.0)
    rn = _road_network(g)
    rn.update_for_tick(FLOOD_INCIDENT, tick=0, dt_minutes=5.0)
    assert rn.travel_time_minutes("0", "island") is None


# ---------------------------------------------------------------------------
# `shortest_path_max_depth` — module M7's `route_flood_safety` constraint
# ---------------------------------------------------------------------------
@pytest.mark.phase7
def test_shortest_path_max_depth_zero_on_dry_route() -> None:
    rn = _road_network(_linear_graph(2, susceptibility=0.0))
    rn.update_for_tick(FLOOD_INCIDENT, tick=60, dt_minutes=5.0)
    assert rn.shortest_path_max_depth("0", "2") == pytest.approx(0.0)


@pytest.mark.phase7
def test_shortest_path_max_depth_matches_worst_edge_on_route() -> None:
    """A 3-edge chain with one badly-flooded edge in the middle — the
    route's max depth should reflect that edge, not an average or the
    last edge."""
    # susceptibility=0.2 -> depth 0.4 m, blockage 0.4/0.6=0.67 - flooded
    # and slow, but still under IMPASSABLE_BLOCKAGE (0.95), so a real
    # route exists through it (this is the "safety" case the rule is
    # meant to catch, not the "no route" case).
    g = _linear_graph(3, susceptibility=0.0)
    g["1"]["2"]["susceptibility"] = 0.2
    rn = _road_network(g)
    rn.update_for_tick(FLOOD_INCIDENT, tick=60, dt_minutes=5.0)  # severity=1, envelope=1
    max_depth = rn.shortest_path_max_depth("0", "3")
    assert max_depth is not None
    assert max_depth == pytest.approx(0.2 * MAX_DEPTH_AT_SEVERITY_1_M)


@pytest.mark.phase7
def test_shortest_path_max_depth_none_when_fully_blocked() -> None:
    rn = _road_network(_linear_graph(1, susceptibility=1.0))
    rn.update_for_tick(FLOOD_INCIDENT, tick=60, dt_minutes=5.0)
    assert rn.shortest_path_max_depth("0", "1") is None


@pytest.mark.phase7
def test_shortest_path_max_depth_none_when_disconnected() -> None:
    g = _linear_graph(1)
    g.add_node("island", x=1.0, y=1.0)
    rn = _road_network(g)
    rn.update_for_tick(FLOOD_INCIDENT, tick=0, dt_minutes=5.0)
    assert rn.shortest_path_max_depth("0", "island") is None
