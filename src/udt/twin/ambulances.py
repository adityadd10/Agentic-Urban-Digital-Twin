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
- **Casualty delivery (2026-09-26, dev doc §3.8 item 6).** Previously the
  patient simply vanished at pickup: the ambulance drove home empty and no
  hospital ever saw the casualty, so the flood never added hospital demand.
  Now a pickup sets `carrying_patient`, and when the return leg completes the
  casualty joins the **home hospital's** queue (`queue_arrivals`/
  `patient_queue`), subject to the same wait-deadline rule as walk-ins.
  Delivering to the home hospital (not the best-placed one) is a disclosed
  simplification: the return route is already computed to there.
- **Uncollected casualties (2026-09-29, dev doc §3.9 item 2, metric-v2).**
  In twin-v2 a call that is never answered costs nothing and never
  expires, so the death proxy only counts patients who waited in a
  *hospital* queue: answering calls could only raise it. With
  `COUNT_UNCOLLECTED_CASUALTY_DEATHS` on, a request still pending after
  `twin/demand.py`'s `PATIENT_WAIT_DEADLINE_HOURS` (the same deadline as
  hospital queues) is removed and counted as a death. **Off by default**:
  twin-v2 behaviour, and every v2 result, stays bit-identical; twin-v3
  turns it on.
- **Destination choice (2026-09-29, dev doc §3.9 mechanic 1).**
  `dispatch_ambulance` takes an optional destination hospital. With none, or
  the home hospital, nothing changes from twin-v2 (pickup -> home, deliver
  there). Otherwise the trip has three legs: home -> pickup, pickup ->
  destination (deliver there), destination -> home (empty). All three are
  routed when dispatched and must be reachable then, like v2's two legs.
- **Transfer requests (2026-09-29, dev doc §3.9 mechanic 3).** The health
  agent adds jobs to `graph.graph["transfer_requests"]` (one per patient,
  located at the source hospital; see `add_transfer_requests`). Transport
  serves them with the same dispatch action and fleet as street calls. At
  pickup the ambulance takes the source hospital's longest-waiting *queued*
  patient, who keeps their original wait start (the 4 h deadline keeps
  running), or, if nobody is queued, an admitted patient. On delivery the
  patient joins the destination's queue. No deaths in transit (disclosed).
  Transfer jobs never expire and don't count toward response times.
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
COUNT_UNCOLLECTED_CASUALTY_DEATHS = False  # twin-v2 = False; twin-v3 = True (dev doc §3.9)


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


def find_job(graph: nx.DiGraph[str], job_id: str) -> dict[str, Any] | None:
    """A pending street call or transfer request by id, or None."""
    for key in ("pending_requests", "transfer_requests"):
        for job in graph.graph.get(key, []):
            if job["request_id"] == job_id:
                return job  # type: ignore[no-any-return]
    return None


def add_transfer_requests(
    graph: nx.DiGraph[str], tick: int, requests: list[tuple[str, int, str]]
) -> int:
    """Health's action: `(from_hospital_id, count, urgency)` entries become
    `count` transfer jobs located at that hospital. Returns jobs added."""
    jobs: list[dict[str, Any]] = graph.graph.setdefault("transfer_requests", [])
    added = 0
    for from_id, count, urgency in requests:
        hospital: Asset = graph.nodes[from_id]["asset"]
        if hospital.asset_type != AssetType.HOSPITAL:
            raise ValueError(f"{from_id!r} is not a hospital")
        lon, lat = hospital.geometry["coordinates"][0], hospital.geometry["coordinates"][1]
        for _ in range(max(0, int(count))):
            seq = int(graph.graph.get("transfer_seq", 0))
            graph.graph["transfer_seq"] = seq + 1
            jobs.append(
                {
                    "request_id": f"TR_{tick}_{from_id}_{seq}",
                    "kind": "transfer",
                    "from_hospital_id": from_id,
                    "location": (lon, lat),
                    "requested_at_tick": tick,
                    "urgency": urgency,
                }
            )
            added += 1
    return added


def _take_patient_for_transfer(graph: nx.DiGraph[str], hospital_id: str) -> tuple[bool, int | None]:
    """(anyone taken?, original wait-start tick or None for an admitted patient)."""
    attrs = graph.nodes[hospital_id]["asset"].attributes
    queue: list[int] = list(attrs.get("queue_arrivals", []))
    if queue:
        oldest = queue.pop(0)
        attrs["queue_arrivals"] = queue
        attrs["patient_queue"] = len(queue)
        return True, oldest
    if int(attrs.get("beds_occupied", 0)) > 0:
        attrs["beds_occupied"] = int(attrs["beds_occupied"]) - 1
        return True, None
    return False, None


def dispatch_ambulance(
    graph: nx.DiGraph[str],
    road_network: RoadNetwork,
    ambulance_id: str,
    request: dict[str, Any],
    destination_hospital_id: str | None = None,
) -> bool:
    """Commits an idle ambulance to a request: computes every leg's real
    travel time up front via `road_network` (assumes `road_network.
    update_for_tick` was already called for the current tick, the same
    requirement as the deciding agent's own query) and sets the ambulance
    to `enroute`. Legs: home -> pickup, then pickup -> the destination
    hospital (default: home), then, if the destination is not home, an
    empty leg back home. Returns `False` (no state change) if any leg has no
    route right now: a flood can genuinely cut a request off, and that's a
    real outcome, not an error to hide."""
    ambulance: Asset = graph.nodes[ambulance_id]["asset"]
    home_node = ambulance.attributes["home_node"]
    home_hospital_id = ambulance.attributes["home_hospital_id"]
    pickup_node = road_network.nearest_node(*request["location"])
    to_pickup = road_network.travel_time_minutes(home_node, pickup_node)
    back_home: float | None = 0.0
    if destination_hospital_id is None or destination_hospital_id == home_hospital_id:
        destination_hospital_id = None
        to_destination = road_network.travel_time_minutes(pickup_node, home_node)
    else:
        destination: Asset = graph.nodes[destination_hospital_id]["asset"]
        if destination.asset_type != AssetType.HOSPITAL:
            raise ValueError(f"{destination_hospital_id!r} is not a hospital")
        lon, lat = destination.geometry["coordinates"][0], destination.geometry["coordinates"][1]
        destination_node = road_network.nearest_node(lon, lat)
        to_destination = road_network.travel_time_minutes(pickup_node, destination_node)
        back_home = road_network.travel_time_minutes(destination_node, home_node)
    if to_pickup is None or to_destination is None or back_home is None:
        return False
    ambulance.attributes["status"] = "enroute"
    ambulance.attributes["assigned_request_id"] = request["request_id"]
    ambulance.attributes["remaining_travel_min"] = to_pickup
    ambulance.attributes["return_travel_min"] = to_destination
    if destination_hospital_id is not None:  # only set off-home, so v2 state is untouched
        ambulance.attributes["delivery_hospital_id"] = destination_hospital_id
        ambulance.attributes["final_return_min"] = back_home
    return True


def _deliver_casualty(
    graph: nx.DiGraph[str], hospital_id: str, tick: int, wait_start_tick: int | None = None
) -> None:
    """Casualty joins the hospital's queue; `consume_demand` admits it (or,
    past the deadline, counts it in the death proxy) like any walk-in."""
    attrs = graph.nodes[hospital_id]["asset"].attributes
    queue: list[int] = list(attrs.get("queue_arrivals", []))
    queue.append(tick if wait_start_tick is None else wait_start_tick)
    attrs["queue_arrivals"] = queue
    attrs["patient_queue"] = len(queue)


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
                attrs["carrying_patient"] = True
            else:  # twin-v3 transfer job?
                transfers: list[dict[str, Any]] = graph.graph.get("transfer_requests", [])
                job = next((r for r in transfers if r["request_id"] == request_id), None)
                if job is not None:
                    transfers.remove(job)
                    taken, wait_start = _take_patient_for_transfer(graph, job["from_hospital_id"])
                    if taken:
                        attrs["carrying_patient"] = True
                        attrs["carrying_transfer"] = True
                        attrs["wait_start_tick"] = wait_start
            attrs["status"] = "returning"
            attrs["remaining_travel_min"] = float(attrs.get("return_travel_min", 0.0))
        else:  # "returning" leg just completed: at the destination, or back home
            if attrs.get("carrying_patient"):
                destination = attrs.pop("delivery_hospital_id", None) or attrs["home_hospital_id"]
                wait_start = attrs.pop("wait_start_tick", None)
                _deliver_casualty(graph, str(destination), tick, wait_start)
                attrs["carrying_patient"] = False
                if attrs.pop("carrying_transfer", False):
                    graph.graph["transfers_completed"] = (
                        int(graph.graph.get("transfers_completed", 0)) + 1
                    )
                back_home = float(attrs.pop("final_return_min", 0.0))
                if back_home > 0:  # delivered away from home: drive back empty
                    attrs["remaining_travel_min"] = back_home
                    continue
            attrs["status"] = "idle"
            attrs["assigned_request_id"] = None
            attrs["remaining_travel_min"] = 0.0
            attrs["return_travel_min"] = 0.0

    graph.graph["pending_requests"] = pending
    return response_times_hours


def expire_uncollected_requests(graph: nx.DiGraph[str], tick: int, dt_hours: float) -> int:
    """Removes every pending request older than the patient wait deadline
    and returns how many were removed (each is one uncollected-casualty
    death). Uses the same deadline and comparison as `twin/demand.py`'s
    hospital queues. An ambulance already en route to an expired request
    finds no patient and returns empty (`advance_ambulances` already handles
    a missing request). Only called when `COUNT_UNCOLLECTED_CASUALTY_DEATHS`
    is on."""
    from udt.twin import demand  # read at call time, so evaluation overrides apply

    deadline_ticks = demand.PATIENT_WAIT_DEADLINE_HOURS / dt_hours
    pending: list[dict[str, Any]] = graph.graph.get("pending_requests", [])
    still_pending = [r for r in pending if (tick - r["requested_at_tick"]) <= deadline_ticks]
    graph.graph["pending_requests"] = still_pending
    return len(pending) - len(still_pending)
