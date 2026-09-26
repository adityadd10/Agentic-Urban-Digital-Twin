"""Ambulance fleet + emergency-request lifecycle (dev doc §3.2 `Ambulance`
attributes, §3.5 step 4's "ambulances advance along routes", §5.3's
dispatch action, module M4).

**Disclosed prototype-1 design choices:**
- Ambulances are `Asset` nodes (matching every other asset type's
  convention) added directly to the live graph — not persisted in
  `dependency_graph.json`, since they aren't real GIS data, they're a
  simulation/agent-scope construct. `AssetType.AMBULANCE` already existed
  in the enum, unused until now.
- **Fleet size** (`spawn_ambulances`'s `n_per_hospital`) and **request
  rate** (`generate_requests`'s `rate_per_hour_at_severity_1`) are
  disclosed placeholders — no real ambulance-fleet roster or 911/108-call
  dataset was sourced. Same status as the twin's other placeholder
  defaults (e.g. `twin/demand.py`'s arrival rate).
- **Requests** are simplified to 3 states implicitly (pending -> enroute
  -> returning -> gone), collapsing the dev doc's separate `loading`
  state into the instant of pickup (disclosed simplification).
- Pending requests live in `graph.graph["pending_requests"]` (NetworkX's
  own graph-level attribute dict) rather than as graph nodes — a request
  isn't infrastructure, so it doesn't belong in the Asset model; this is
  the simplest place for something the agent still needs to see via the
  same `graph` object `Agent.act` already receives.
- Response time (dev doc §5.4 `ambulance_response_delay_hours`) is
  measured request-creation to **pickup** (industry-standard "call to
  scene arrival"), not to hospital drop-off.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
from shapely.geometry import Point, Polygon

from udt.common.models import Asset, AssetType, Incident
from udt.twin.road_network import RoadNetwork

N_AMBULANCES_PER_HOSPITAL_DEFAULT = 2
REQUEST_RATE_PER_HOUR_AT_SEVERITY_1_DEFAULT = 2.0


def spawn_ambulances(
    graph: nx.DiGraph[str],
    road_network: RoadNetwork,
    n_per_hospital: int = N_AMBULANCES_PER_HOSPITAL_DEFAULT,
) -> list[str]:
    """Stations `n_per_hospital` idle ambulances at each hospital's
    nearest road node. Returns the new ambulance asset ids."""
    hospital_ids = [
        asset_id
        for asset_id in list(graph.nodes)
        if graph.nodes[asset_id]["asset"].asset_type == AssetType.HOSPITAL
    ]
    new_ids: list[str] = []
    for hospital_id in hospital_ids:
        hospital: Asset = graph.nodes[hospital_id]["asset"]
        lon, lat = hospital.geometry["coordinates"][0], hospital.geometry["coordinates"][1]
        home_node = road_network.nearest_node(lon, lat)
        for i in range(n_per_hospital):
            asset_id = f"AMB_{hospital_id}_{i}"
            ambulance = Asset(
                asset_id=asset_id,
                asset_type=AssetType.AMBULANCE,
                geometry=hospital.geometry,
                attributes={
                    "status": "idle",
                    "home_node": home_node,
                    "home_hospital_id": hospital_id,
                    "assigned_request_id": None,
                    "remaining_travel_min": 0.0,
                    "return_travel_min": 0.0,
                },
            )
            graph.add_node(asset_id, asset=ambulance)
            new_ids.append(asset_id)
    return new_ids


def generate_requests(
    graph: nx.DiGraph[str],
    tick: int,
    dt_hours: float,
    incident: Incident,
    ward_polygon: Polygon,
    rng: np.random.Generator,
    *,
    rate_per_hour_at_severity_1: float = REQUEST_RATE_PER_HOUR_AT_SEVERITY_1_DEFAULT,
) -> None:
    """Poisson-samples new emergency requests at random points inside the
    ward, scaled by the incident's severity — disclosed placeholder, not
    derived from any real emergency-call dataset (see module docstring)."""
    rate = rate_per_hour_at_severity_1 * incident.severity
    n_new = int(rng.poisson(max(0.0, rate * dt_hours)))
    if n_new == 0:
        return
    pending: list[dict[str, Any]] = graph.graph.setdefault("pending_requests", [])
    minx, miny, maxx, maxy = ward_polygon.bounds
    for _ in range(n_new):
        lon, lat = minx, miny  # fallback if rejection sampling exhausts its budget
        for _attempt in range(50):
            candidate_lon = float(rng.uniform(minx, maxx))
            candidate_lat = float(rng.uniform(miny, maxy))
            if ward_polygon.contains(Point(candidate_lon, candidate_lat)):
                lon, lat = candidate_lon, candidate_lat
                break
        pending.append(
            {
                "request_id": f"REQ_{tick}_{len(pending)}_{int(rng.integers(1_000_000))}",
                "location": (lon, lat),
                "requested_at_tick": tick,
            }
        )


def dispatch_ambulance(
    graph: nx.DiGraph[str],
    road_network: RoadNetwork,
    ambulance_id: str,
    request: dict[str, Any],
) -> bool:
    """Commits an idle ambulance to a request: computes both legs'
    real travel times up front (pickup, then return) via `road_network`
    (assumes `road_network.update_for_tick` was already called for the
    current tick — same requirement as the deciding agent's own query),
    and sets the ambulance to `enroute`. Returns `False` (no state change)
    if no route currently exists to or from the pickup point — a flood
    can genuinely cut a request off, and that's a real outcome, not an
    error to hide."""
    ambulance: Asset = graph.nodes[ambulance_id]["asset"]
    home_node = ambulance.attributes["home_node"]
    pickup_node = road_network.nearest_node(*request["location"])
    to_pickup = road_network.travel_time_minutes(home_node, pickup_node)
    to_home = road_network.travel_time_minutes(pickup_node, home_node)
    if to_pickup is None or to_home is None:
        return False
    ambulance.attributes["status"] = "enroute"
    ambulance.attributes["assigned_request_id"] = request["request_id"]
    ambulance.attributes["remaining_travel_min"] = to_pickup
    ambulance.attributes["return_travel_min"] = to_home
    return True


def advance_ambulances(graph: nx.DiGraph[str], tick: int, dt_minutes: float) -> list[float]:
    """dev doc §3.5 step 4's "ambulances advance along routes". Returns
    the response times (hours) of every request picked up this tick, for
    the caller (`Simulator.step`) to accumulate into episode metrics."""
    dt_hours = dt_minutes / 60.0
    pending: list[dict[str, Any]] = graph.graph.get("pending_requests", [])
    response_times_hours: list[float] = []

    for asset_id in list(graph.nodes):
        asset: Asset = graph.nodes[asset_id]["asset"]
        if asset.asset_type != AssetType.AMBULANCE:
            continue
        attrs = asset.attributes
        status = attrs.get("status", "idle")
        if status == "idle":
            continue

        remaining = float(attrs.get("remaining_travel_min", 0.0)) - dt_minutes
        if remaining > 0:
            attrs["remaining_travel_min"] = remaining
            continue

        if status == "enroute":
            request_id = attrs.get("assigned_request_id")
            request = next((r for r in pending if r["request_id"] == request_id), None)
            if request is not None:
                response_times_hours.append((tick - request["requested_at_tick"]) * dt_hours)
                pending.remove(request)
            attrs["status"] = "returning"
            attrs["remaining_travel_min"] = float(attrs.get("return_travel_min", 0.0))
        else:  # "returning" leg just completed -> back home, idle again
            attrs["status"] = "idle"
            attrs["assigned_request_id"] = None
            attrs["remaining_travel_min"] = 0.0
            attrs["return_travel_min"] = 0.0

    graph.graph["pending_requests"] = pending
    return response_times_hours
