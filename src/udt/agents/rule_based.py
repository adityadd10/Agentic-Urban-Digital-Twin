"""Rule-based baseline (dev doc §5.6, Experiment A — "must be credible,
not a strawman").

All four of the dev doc's §5.6 rules are implemented: repair-crew
dispatch, ambulance dispatch, patient transfer, and load-shedding — the
last one landed only once `twin/power.py` gave a substation's
`load_mw`/`capacity_mw` an actual consequence to protect against (see
that module's docstring for the two design decisions it needed).
"""

from __future__ import annotations

import networkx as nx

from udt.common.models import AgentAction, AssetType
from udt.twin.cascade import EdgeRuntimeState
from udt.twin.graph import dependency_edges_of
from udt.twin.power import DESHED_THRESHOLD, MAX_SHED_TIER, OVERLOAD_THRESHOLD, post_shed_ratio
from udt.twin.road_network import RoadNetwork

# Dev doc §5.6 says "highest-criticality failed substation" — generalized
# to any of these types since narrowing the rule to "substation"
# specifically would make it substation-only by accident, not by design
# (before the M2 revision this also compensated for the real Kurla graph
# having 0 water facilities; that gap is closed now — see
# data/manual_facilities.yaml — but the generalization is kept, since it
# was always the more defensible reading of the dev doc's own text).
REPAIRABLE_TYPES = frozenset({AssetType.HOSPITAL, AssetType.SUBSTATION, AssetType.WATER})

# Disclosed placeholder (dev doc §5.6 says "predicted [...] to lose power
# within its buffer window" but doesn't pin a number): a hospital with
# less than this much power-buffer left is "at risk". Not derived from
# any real generator-runtime/response-time data.
PREDICTED_LOSS_THRESHOLD_HOURS = 1.0

# Dev doc §5.3's discrete transfer amounts, largest-fits-first.
TRANSFER_TIERS = (10, 5, 2)


def _pick_repair_target(graph: nx.DiGraph[str]) -> str | None:
    """Dev doc §5.6's repair-crew rule: the most-damaged repairable
    facility, or `None` if everything's healthy. Ties are broken by
    iteration order (stable for a fixed graph) — a disclosed default,
    not dev-doc-specified."""
    worst_id: str | None = None
    worst_level = 1.0
    for asset_id in graph.nodes:
        asset = graph.nodes[asset_id]["asset"]
        if asset.asset_type not in REPAIRABLE_TYPES:
            continue
        if asset.functional_level < worst_level:
            worst_level = asset.functional_level
            worst_id = asset_id
    return worst_id if worst_id is not None and worst_level < 1.0 else None


def _pick_ambulance_dispatch(
    graph: nx.DiGraph[str], road_network: RoadNetwork | None
) -> dict[str, str] | None:
    """Dev doc §5.6: "dispatch nearest idle ambulance to oldest critical
    request" — one dispatch decision per tick (the dev doc's wording is
    singular; dispatching every idle ambulance to every pending request
    in one tick would be a different, more aggressive rule, not what's
    written). Requires `road_network` — with none, this rule can't query
    real travel times, so it does nothing rather than guess with
    straight-line distance."""
    if road_network is None:
        return None
    pending: list[dict[str, object]] = graph.graph.get("pending_requests", [])
    if not pending:
        return None
    idle_ambulance_ids = [
        asset_id
        for asset_id in graph.nodes
        if graph.nodes[asset_id]["asset"].asset_type == AssetType.AMBULANCE
        and graph.nodes[asset_id]["asset"].attributes.get("status") == "idle"
    ]
    if not idle_ambulance_ids:
        return None

    oldest_request = min(pending, key=lambda r: r["requested_at_tick"])  # type: ignore[arg-type,return-value]
    pickup_node = road_network.nearest_node(*oldest_request["location"])  # type: ignore[misc]

    best_ambulance_id: str | None = None
    best_time = float("inf")
    for ambulance_id in idle_ambulance_ids:
        home_node = graph.nodes[ambulance_id]["asset"].attributes["home_node"]
        travel_time = road_network.travel_time_minutes(home_node, pickup_node)
        if travel_time is not None and travel_time < best_time:
            best_time = travel_time
            best_ambulance_id = ambulance_id

    if best_ambulance_id is None:  # every idle ambulance is currently cut off
        return None
    return {best_ambulance_id: str(oldest_request["request_id"])}


def _nearest_hospital(
    graph: nx.DiGraph[str],
    road_network: RoadNetwork | None,
    source_id: str,
    candidate_ids: list[str],
) -> str | None:
    """Real travel time if `road_network` is available and at least one
    candidate is actually reachable; straight-line distance otherwise
    (also the fallback when every route is currently cut off) — a
    disclosed simplification, same status as the twin's other geometry-
    only fallbacks (e.g. `06_build_dependency_graph.py`'s nearest-
    substation rule)."""
    source_geom = graph.nodes[source_id]["asset"].geometry["coordinates"]

    if road_network is not None:
        source_node = road_network.nearest_node(*source_geom)
        best_id: str | None = None
        best_time = float("inf")
        for candidate_id in candidate_ids:
            candidate_geom = graph.nodes[candidate_id]["asset"].geometry["coordinates"]
            candidate_node = road_network.nearest_node(*candidate_geom)
            travel_time = road_network.travel_time_minutes(source_node, candidate_node)
            if travel_time is not None and travel_time < best_time:
                best_time = travel_time
                best_id = candidate_id
        if best_id is not None:
            return best_id

    best_id = None
    best_dist_sq = float("inf")
    for candidate_id in candidate_ids:
        candidate_geom = graph.nodes[candidate_id]["asset"].geometry["coordinates"]
        dist_sq = (candidate_geom[0] - source_geom[0]) ** 2 + (
            candidate_geom[1] - source_geom[1]
        ) ** 2
        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best_id = candidate_id
    return best_id


def _pick_patient_transfer(
    graph: nx.DiGraph[str],
    edge_states: dict[str, EdgeRuntimeState] | None,
    road_network: RoadNetwork | None,
) -> tuple[str, str, int] | None:
    """Dev doc §5.6: "transfer patients out of any hospital predicted
    (by current trends, no simulation) to lose power within its buffer
    window, nearest-available-bed first". "Predicted by current trends,
    no simulation" is read here as "the buffer is already low" (a
    threshold on `EdgeRuntimeState.remaining_hours`), not a trend
    extrapolation — no counterfactual-rollout machinery exists yet
    (`twin/counterfactual.py`'s `simulate()`, §3.7, is still deferred) to
    do anything fancier, and a plain threshold is a legitimate, disclosed
    reading of "predicted [...] no simulation" on its own.
    """
    if edge_states is None:
        return None
    hospital_ids = [
        asset_id
        for asset_id in graph.nodes
        if graph.nodes[asset_id]["asset"].asset_type == AssetType.HOSPITAL
    ]
    if len(hospital_ids) < 2:  # no other hospital to transfer to
        return None

    worst_hospital_id: str | None = None
    worst_remaining_hours = float("inf")
    for hospital_id in hospital_ids:
        for edge in dependency_edges_of(graph, hospital_id):
            if edge.kind != "power":
                continue
            state = edge_states.get(edge.edge_id)
            if state is None:
                continue
            if state.remaining_hours < worst_remaining_hours:
                worst_remaining_hours = state.remaining_hours
                worst_hospital_id = hospital_id

    if worst_hospital_id is None or worst_remaining_hours >= PREDICTED_LOSS_THRESHOLD_HOURS:
        return None  # nothing at risk this tick

    def free_beds(hospital_id: str) -> int:
        attrs = graph.nodes[hospital_id]["asset"].attributes
        return int(attrs.get("beds_total", 0)) - int(attrs.get("beds_occupied", 0))

    candidates = [hid for hid in hospital_ids if hid != worst_hospital_id and free_beds(hid) > 0]
    if not candidates:
        return None
    destination_id = _nearest_hospital(graph, road_network, worst_hospital_id, candidates)
    if destination_id is None:
        return None

    available = int(graph.nodes[worst_hospital_id]["asset"].attributes.get("beds_occupied", 0))
    feasible = min(available, free_beds(destination_id))
    for tier in TRANSFER_TIERS:
        if feasible >= tier:
            return (worst_hospital_id, destination_id, tier)
    return None  # feasible < smallest tier (2) — not worth a transfer at this granularity


def _pick_load_shed(graph: nx.DiGraph[str]) -> dict[str, int] | None:
    """Dev doc §5.6: "shed load tier-by-tier when a substation exceeds
    95% capacity". Escalates one tier when still overloaded after the
    current tier's shedding; de-escalates one tier once comfortably
    under a lower threshold (`DESHED_THRESHOLD` — disclosed hysteresis,
    prevents the tier flapping up and down every tick right at the
    boundary). Every substation is independently eligible, not just the
    single worst one — see `AgentAction.shed_tier`'s docstring for why
    this rule doesn't follow the "pick the one worst asset" convention
    the other three rules do."""
    updates: dict[str, int] = {}
    for asset_id in graph.nodes:
        asset = graph.nodes[asset_id]["asset"]
        if asset.asset_type != AssetType.SUBSTATION:
            continue
        attrs = asset.attributes
        capacity_mw = float(attrs.get("capacity_mw", 0.0))
        if capacity_mw <= 0:
            continue
        current_tier = int(attrs.get("shed_tier", 0))
        ratio = post_shed_ratio(float(attrs.get("load_mw", 0.0)), capacity_mw, current_tier)
        if ratio > OVERLOAD_THRESHOLD and current_tier < MAX_SHED_TIER:
            updates[asset_id] = current_tier + 1
        elif ratio < DESHED_THRESHOLD and current_tier > 0:
            updates[asset_id] = current_tier - 1
    return updates or None


class RuleBasedAgent:
    """All four of dev doc §5.6's rules — see module docstring."""

    name = "rule_based"

    def act(
        self,
        graph: nx.DiGraph[str],
        tick: int,
        road_network: RoadNetwork | None = None,
        edge_states: dict[str, EdgeRuntimeState] | None = None,
    ) -> AgentAction:
        return AgentAction(
            repair_target=_pick_repair_target(graph),
            ambulance_assignment=_pick_ambulance_dispatch(graph, road_network),
            patient_transfer=_pick_patient_transfer(graph, edge_states, road_network),
            shed_tier=_pick_load_shed(graph),
        )


class DoNothingAgent:
    """The zero-action control condition — not part of the dev doc's
    §5.6 spec, but the natural baseline for the harness to compare the
    rule-based agent against (dev doc §14 Phase 4 wants "an Experiment A
    results table"; a table with only one row can't show whether the
    rules actually helped)."""

    name = "do_nothing"

    def act(
        self,
        graph: nx.DiGraph[str],
        tick: int,
        road_network: RoadNetwork | None = None,
        edge_states: dict[str, EdgeRuntimeState] | None = None,
    ) -> AgentAction:
        return AgentAction(
            repair_target=None,
            ambulance_assignment=None,
            patient_transfer=None,
            shed_tier=None,
        )
