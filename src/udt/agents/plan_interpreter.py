"""Plan interpreter — `Plan -> AgentAction` (dev doc §7.3.1), bridges the
LLM planning layer to real, differentiated execution before a trained
goal-conditioned MAPPO policy exists at usable quality (M6b/M7 — every
row there discloses training has only run at smoke scale so far).

**Why this exists:** `llm/graph.py`'s `simulate_each` node used to pass
the same fixed `proxy_action` to `simulate()` for every candidate plan,
regardless of that plan's `goal_weights`/`priority_assets`/`directives` —
so counterfactual comparison had nothing to actually compare (found and
disclosed 2026-09-22, `MTP_Module_Planner.md`'s M9a row). Live execution
had the identical problem: `api/episode_session.py`'s `RuleBasedPolicyExecutor`
disclosed outright that it "ignores goal_weights entirely". This module
fixes both call sites with one mechanism.

**Design: nudge the baseline, don't replace it.** Every function here
wraps one of `agents/rule_based.py`'s four already-tested pick functions
and biases its choice according to a plan's declared priorities — it does
not invent new decision logic. A `Plan` with no `priority_assets`/
`directives` and default (all-1.0) `goal_weights` degenerates exactly to
`RuleBasedAgent.act`'s own decision (a free correctness check — see
`test_plan_interpreter.py`'s `test_empty_plan_matches_rule_based_baseline`).

**Disclosed dependency direction:** this module imports `Plan`/`Directive`
from `udt.llm.schemas` — a new `agents -> llm.schemas` edge that didn't
exist before. Kept intentional and narrow (schemas only, never
`llm.graph`/`llm.tools`) rather than duplicating the schema, since `Plan`
is the one real contract this module exists to interpret — duplicating it
would drift the moment the schema changed. `api/episode_session.py`
already imports from both `agents/` and `llm/` directly, so this doesn't
introduce a new coupling at the application level, only names it
explicitly at the package level.

**Closed directive vocabulary — exactly four kinds, one per `AgentAction`
field (dev doc §5.3):** a directive cannot bias a lever the twin doesn't
have. An unrecognized `kind` is not an error (a plan may carry directives
a future action-space extension would understand) — it's simply not
interpreted by anything here, same "advisory unless understood" status
dev doc §7.3 originally gave every directive.
"""

from __future__ import annotations

import networkx as nx

from udt.agents.rule_based import (
    REPAIRABLE_TYPES,
    TRANSFER_TIERS,
    _nearest_hospital,
    _pick_ambulance_dispatch,
    _pick_patient_transfer,
    _pick_repair_target,
)
from udt.common.models import AgentAction, AssetType
from udt.llm.schemas import Plan
from udt.twin.cascade import EdgeRuntimeState
from udt.twin.power import DESHED_THRESHOLD, MAX_SHED_TIER, OVERLOAD_THRESHOLD, post_shed_ratio
from udt.twin.road_network import RoadNetwork

# Disclosed, uncalibrated threshold-nudge constant (dev doc §7.3.1) — how
# far one `shed_bias` directive, or one full unit of `g_power` away from
# 1.0, shifts the shed/deshed thresholds. Small on purpose: the gap
# between `OVERLOAD_THRESHOLD` (0.95) and `DESHED_THRESHOLD` (0.80) is
# 0.15, so even a directive-plus-weight combination (worst case ~0.03)
# can't invert their order. Not fitted to any real outcome data — flag
# for the same calibration pass `scripts/calibrate_risk.py` already
# covers for this twin's other placeholder constants.
SHED_BIAS_DELTA = 0.02


def _repair_target(plan: Plan, graph: nx.DiGraph[str]) -> str | None:
    """`protect_repair` directives win first (explicit, structured);
    `priority_assets` next (looser list, dev doc §7.3's "priority target
    designations"); the baseline worst-asset rule otherwise. Both only
    fire if the named asset is actually damaged and repairable — a
    directive naming a healthy or non-repairable asset is not a license
    to repair nothing when something else needs it."""
    for directive in plan.directives:
        if directive.kind != "protect_repair":
            continue
        asset_id = directive.details.get("asset_id")
        if asset_id and asset_id in graph.nodes:
            asset = graph.nodes[asset_id]["asset"]
            if asset.asset_type in REPAIRABLE_TYPES and asset.functional_level < 1.0:
                return asset_id
    for asset_id in plan.priority_assets:
        if asset_id in graph.nodes:
            asset = graph.nodes[asset_id]["asset"]
            if asset.asset_type in REPAIRABLE_TYPES and asset.functional_level < 1.0:
                return asset_id
    return _pick_repair_target(graph)


def _free_beds(graph: nx.DiGraph[str], hospital_id: str) -> int:
    attrs = graph.nodes[hospital_id]["asset"].attributes
    return int(attrs.get("beds_total", 0)) - int(attrs.get("beds_occupied", 0))


def _transfer_from(
    graph: nx.DiGraph[str], road_network: RoadNetwork | None, hospital_id: str
) -> tuple[str, str, int] | None:
    """The feasibility half of `_pick_patient_transfer` (destination
    search, bed availability, transfer-tier sizing), reused directly —
    only the "which hospital is the source" half is replaced, since a
    `protect_transfer` directive names that explicitly instead of
    searching for the single worst hospital."""
    hospital_ids = [
        aid for aid in graph.nodes if graph.nodes[aid]["asset"].asset_type == AssetType.HOSPITAL
    ]
    if hospital_id not in hospital_ids or len(hospital_ids) < 2:
        return None
    candidates = [hid for hid in hospital_ids if hid != hospital_id and _free_beds(graph, hid) > 0]
    if not candidates:
        return None
    destination_id = _nearest_hospital(graph, road_network, hospital_id, candidates)
    if destination_id is None:
        return None
    available = int(graph.nodes[hospital_id]["asset"].attributes.get("beds_occupied", 0))
    feasible = min(available, _free_beds(graph, destination_id))
    for tier in TRANSFER_TIERS:
        if feasible >= tier:
            return (hospital_id, destination_id, tier)
    return None


def _patient_transfer(
    plan: Plan,
    graph: nx.DiGraph[str],
    edge_states: dict[str, EdgeRuntimeState] | None,
    road_network: RoadNetwork | None,
) -> tuple[str, str, int] | None:
    """A `protect_transfer` directive is a stronger signal than the
    baseline's own buffer-threshold check (dev doc §5.6) — it moves
    patients out of the named hospital preemptively as long as it's
    physically feasible (a destination with free beds exists), not only
    once that hospital's own buffer has already dropped low. Falls back
    to the baseline's threshold-triggered rule if no directive fires."""
    for directive in plan.directives:
        if directive.kind != "protect_transfer":
            continue
        hospital_id = directive.details.get("hospital_id")
        if hospital_id:
            result = _transfer_from(graph, road_network, hospital_id)
            if result is not None:
                return result
    return _pick_patient_transfer(graph, edge_states, road_network)


def _nearest_idle_ambulance(
    graph: nx.DiGraph[str],
    road_network: RoadNetwork,
    idle_ambulance_ids: list[str],
    pickup_node: object,
) -> str | None:
    best_id: str | None = None
    best_time = float("inf")
    for ambulance_id in idle_ambulance_ids:
        home_node = graph.nodes[ambulance_id]["asset"].attributes["home_node"]
        travel_time = road_network.travel_time_minutes(home_node, pickup_node)  # type: ignore[arg-type]
        if travel_time is not None and travel_time < best_time:
            best_time = travel_time
            best_id = ambulance_id
    return best_id


def _ambulance_dispatch(
    plan: Plan, graph: nx.DiGraph[str], road_network: RoadNetwork | None
) -> dict[str, str] | None:
    """A `protect_dispatch` directive re-ranks *which pending request*
    gets served first — nearest to the named hospital (by real travel
    time when a road network is available) instead of strictly
    oldest-first — then reuses the baseline's own nearest-idle-ambulance
    search for that request. Falls back to oldest-first if no directive
    fires or the named hospital isn't reachable."""
    if road_network is not None:
        for directive in plan.directives:
            if directive.kind != "protect_dispatch":
                continue
            hospital_id = directive.details.get("hospital_id")
            if not hospital_id or hospital_id not in graph.nodes:
                continue
            pending: list[dict[str, object]] = graph.graph.get("pending_requests", [])
            if not pending:
                continue
            idle_ambulance_ids = [
                asset_id
                for asset_id in graph.nodes
                if graph.nodes[asset_id]["asset"].asset_type == AssetType.AMBULANCE
                and graph.nodes[asset_id]["asset"].attributes.get("status") == "idle"
            ]
            if not idle_ambulance_ids:
                continue
            hospital_geom = graph.nodes[hospital_id]["asset"].geometry["coordinates"]
            hospital_node = road_network.nearest_node(*hospital_geom)
            best_request: dict[str, object] | None = None
            best_time = float("inf")
            for request in pending:
                pickup_node = road_network.nearest_node(*request["location"])  # type: ignore[misc]
                travel_time = road_network.travel_time_minutes(hospital_node, pickup_node)
                if travel_time is not None and travel_time < best_time:
                    best_time = travel_time
                    best_request = request
            if best_request is None:
                continue
            pickup_node = road_network.nearest_node(*best_request["location"])  # type: ignore[misc]
            best_ambulance_id = _nearest_idle_ambulance(
                graph, road_network, idle_ambulance_ids, pickup_node
            )
            if best_ambulance_id is not None:
                return {best_ambulance_id: str(best_request["request_id"])}
    return _pick_ambulance_dispatch(graph, road_network)


def _load_shed(plan: Plan, graph: nx.DiGraph[str]) -> dict[str, int] | None:
    """`shed_bias` directives (explicit) take precedence over the
    `g_power`-derived nudge (implicit) — see this module's docstring and
    dev doc §7.3.1 for why a directive is treated as the more specific
    instruction. Both only ever shift the baseline's own thresholds by a
    small amount (`SHED_BIAS_DELTA`); they never bypass the hysteresis
    band or the tier cap."""
    directive_bias = 0.0
    has_directive = False
    for directive in plan.directives:
        if directive.kind != "shed_bias":
            continue
        has_directive = True
        direction = directive.details.get("direction")
        if direction == "early":
            directive_bias -= SHED_BIAS_DELTA
        elif direction == "late":
            directive_bias += SHED_BIAS_DELTA

    bias = directive_bias if has_directive else -(plan.goal_weights.g_power - 1.0) * SHED_BIAS_DELTA

    effective_overload = OVERLOAD_THRESHOLD + bias
    effective_deshed = DESHED_THRESHOLD + bias

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
        if ratio > effective_overload and current_tier < MAX_SHED_TIER:
            updates[asset_id] = current_tier + 1
        elif ratio < effective_deshed and current_tier > 0:
            updates[asset_id] = current_tier - 1
    return updates or None


def interpret_plan(
    plan: Plan,
    graph: nx.DiGraph[str],
    *,
    road_network: RoadNetwork | None = None,
    edge_states: dict[str, EdgeRuntimeState] | None = None,
) -> AgentAction:
    """`Plan -> AgentAction` (dev doc §7.3.1). Deterministic, no LLM
    involved — every candidate plan interpreted this way against the same
    live state produces a real, potentially different `AgentAction`,
    which is the whole point: it's what lets `simulate_each` (counterfactual
    comparison) and live execution both actually depend on which plan is
    in play, instead of silently ignoring it."""
    return AgentAction(
        repair_target=_repair_target(plan, graph),
        ambulance_assignment=_ambulance_dispatch(plan, graph, road_network),
        patient_transfer=_patient_transfer(plan, graph, edge_states, road_network),
        shed_tier=_load_shed(plan, graph),
    )
