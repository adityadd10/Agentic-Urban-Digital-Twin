"""Repair-crew movement (twin-v3, dev doc §3.9 mechanic 5).

In twin-v2 a repair is instantaneous: `repair_target` adds `repair_rate` to
the asset's intrinsic level on each decision tick, wherever the asset is. With
`REPAIR_CREW_TRAVEL` on (twin-v3), `repair_target` instead *sends* the one
repair crew:

- The crew starts at its depot, the road node nearest `CREW_DEPOT_ASSET_ID`
  (S0, the Tata Power Kurla Receiving Station; a disclosed assumption).
- Assigning a target routes the crew there over the current flooded road
  network. No route right now means the order is refused (returns False) and
  the crew keeps its current job. Travel time is fixed at assignment, like
  ambulance legs. A crew reassigned mid-journey restarts from the last node it
  left (disclosed approximation).
- On arrival it works on the target every tick at `repair_rate /
  REPAIR_TICKS_PER_DECISION`, the same speed as twin-v2's one application per
  15-minute decision (a full repair in about 5 h). It keeps working until the
  target is fully repaired or it is reassigned. `repair_target=None` means
  "carry on", not "stop". Flood damage still wins while the site is flooded
  (`incidents/degradations/flood.py`).

Off by default: twin-v2 behaviour and results are unchanged.
"""

from __future__ import annotations

from typing import Any

import networkx as nx

from udt.common.models import AssetType
from udt.twin.road_network import RoadNetwork

REPAIR_CREW_TRAVEL = False  # twin-v2: False; twin-v3: True
CREW_DEPOT_ASSET_ID = "S0"
REPAIR_TICKS_PER_DECISION = 3  # decision interval (15 min) in 5-min ticks
CREW_KEY = "repair_crew"


def _node_of(graph: nx.DiGraph[str], road_network: RoadNetwork, asset_id: str) -> str:
    lon, lat = graph.nodes[asset_id]["asset"].geometry["coordinates"][:2]
    return road_network.nearest_node(lon, lat)


def crew_state(graph: nx.DiGraph[str], road_network: RoadNetwork) -> dict[str, Any]:
    """The crew, created at its depot the first time it is needed."""
    crew: dict[str, Any] | None = graph.graph.get(CREW_KEY)
    if crew is None:
        depot = CREW_DEPOT_ASSET_ID if CREW_DEPOT_ASSET_ID in graph.nodes else None
        if depot is None:
            depot = next(
                a
                for a in graph.nodes
                if graph.nodes[a]["asset"].asset_type in (AssetType.SUBSTATION, AssetType.HOSPITAL)
            )
        crew = {
            "node": _node_of(graph, road_network, depot),
            "status": "idle",
            "target": None,
            "remaining_travel_min": 0.0,
        }
        graph.graph[CREW_KEY] = crew
    return crew


def assign_crew(graph: nx.DiGraph[str], road_network: RoadNetwork, target_id: str) -> bool:
    """Send the crew to `target_id`. False (nothing changes) if unreachable now."""
    crew = crew_state(graph, road_network)
    if crew["target"] == target_id:
        return True  # already on its way or working there
    target_node = _node_of(graph, road_network, target_id)
    travel = road_network.travel_time_minutes(crew["node"], target_node)
    if travel is None:
        return False
    crew.update(
        {
            "status": "travelling" if travel > 0 else "working",
            "target": target_id,
            "target_node": target_node,
            "remaining_travel_min": float(travel),
        }
    )
    if travel <= 0:
        crew["node"] = target_node
    return True


def advance_crew(graph: nx.DiGraph[str], dt_minutes: float) -> str | None:
    """Move the crew one tick. Returns the asset to repair this tick, or None."""
    crew: dict[str, Any] | None = graph.graph.get(CREW_KEY)
    if crew is None or crew["status"] == "idle":
        return None
    if crew["status"] == "travelling":
        crew["remaining_travel_min"] = float(crew["remaining_travel_min"]) - dt_minutes
        if crew["remaining_travel_min"] <= 0:
            crew["status"], crew["node"], crew["remaining_travel_min"] = (
                "working",
                crew["target_node"],
                0.0,
            )
        return None  # starts repairing on the next tick
    target = str(crew["target"])
    if graph.nodes[target]["asset"].intrinsic_level >= 1.0:
        crew["status"], crew["target"] = "idle", None
        return None
    return target
