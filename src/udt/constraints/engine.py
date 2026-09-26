"""Constraint engine (dev doc §8, module M7).

Declarative rules live in `configs/constraints.yaml`, documented there in
the dev doc's own notation (`rule: "transfers_in[h] <= beds_free[h]"`,
etc.) for reference/logging. The actual check logic is a registered
Python callable per rule id — same `@register_...` decorator-registry
pattern already used for incident types (`incidents/registry.py`)
rather than a bespoke expression-language parser for those YAML strings,
since only 3 rules exist and each one's real inputs (`AgentAction`
fields, `RoadNetwork` queries) are Python objects anyway, not a flat
namespace an expression evaluator could bind against generically.

API, dev doc §8 verbatim: `check(state, joint_action) ->
ConstraintReport{passed, violations[], repaired_action}`. Repair order:
**clip magnitude -> drop sub-action -> replace with no-op**. Used three
ways per the dev doc: (1) pre-hoc action masking during constrained-MARL
training, (2) post-hoc validation of every action before execution (all
experiments, all paths), (3) a future `check_constraints` LLM tool.
Only (2) is wired up in this slice — see this module's own docstring
note below and `MTP_Module_Planner.md`'s M7 row for what's deferred to
slice 2 (pre-hoc masking + the Lagrangian soft-constraint variant, the
actual "constrained MARL" half of M7's acceptance criterion).

**Rules implemented, and why the third isn't (disclosed, not silently
dropped):**
- `bed_capacity` (`transfers_in[h] <= beds_free[h]`, `on_violation:
  clip`): checks `AgentAction.patient_transfer`'s destination hospital
  against its actual free-bed count — `twin/demand.py`'s
  `effective_free_beds` (M8: scaled by the destination's current
  `functional_level`, not just its nominal `beds_total`), the same
  function `apply_patient_transfer`'s enactment uses, so this check and
  what actually happens never disagree. Clipping to 0 free beds means
  "no transfer" — repaired to `patient_transfer=None` rather than a
  0-patient transfer tuple, since the latter is a distinction without a
  difference downstream.
- `route_flood_safety` (`route_max_flood_depth[a] < 0.4`, `on_violation:
  drop`): checks each entry of `AgentAction.ambulance_assignment`
  against `RoadNetwork.shortest_path_max_depth` for the exact route the
  dispatch would take (home node -> pickup node) — the same route
  `twin/ambulances.py`'s `dispatch_ambulance` would actually enact.
  A dispatch to an already-gone request (stale decision — the request
  was already picked up/expired by the time this check runs) is left
  as-is: `Simulator.step` already no-ops on that case on its own, and
  "the request doesn't exist" isn't a flood-safety violation to report.
- `power_balance` (`allocated_mw[s] <= capacity_mw[s] *
  functional_level[s]`, `on_violation: clip`): **NOT implemented.**
  No `allocated_mw` decision variable exists anywhere in this build —
  `AgentAction.shed_tier` is the only power-side action, and it's a
  discrete tier choice, not a continuous MW allocation the dev doc's
  rule text is written against. Registered as a real no-op check (always
  passes) rather than omitted from the registry, so `configs/
  constraints.yaml`'s three declared rules and this module's registry
  stay 1:1 — the gap is visible in the no-op's own docstring, not hidden
  by a missing entry.
"""

from __future__ import annotations

from collections.abc import Callable

import networkx as nx

from udt.common.models import AgentAction, ConstraintReport, ConstraintViolation
from udt.twin.demand import effective_free_beds
from udt.twin.road_network import RoadNetwork

BED_CAPACITY_RULE_ID = "bed_capacity"
ROUTE_FLOOD_SAFETY_RULE_ID = "route_flood_safety"
POWER_BALANCE_RULE_ID = "power_balance"

# dev doc §8 exactly: "route_max_flood_depth[a] < 0.4".
ROUTE_FLOOD_SAFETY_MAX_DEPTH_M = 0.4

# `(graph, action, road_network) -> (violations found, action with those
# violations' repairs applied)` — repairs chain: each rule receives the
# *previous* rule's repaired action, not the original, so two rules that
# both touch the same field compose correctly instead of one silently
# undoing the other.
CheckFn = Callable[
    ["nx.DiGraph[str]", AgentAction, "RoadNetwork | None"],
    tuple[list[ConstraintViolation], AgentAction],
]

# Registration order = evaluation order = `configs/constraints.yaml`'s
# declared order (dev doc §8's own listing order): bed_capacity,
# route_flood_safety, power_balance.
_REGISTRY: dict[str, CheckFn] = {}


def register_rule(rule_id: str) -> Callable[[CheckFn], CheckFn]:
    def decorator(fn: CheckFn) -> CheckFn:
        if rule_id in _REGISTRY:
            raise ValueError(f"Constraint rule {rule_id!r} already registered")
        _REGISTRY[rule_id] = fn
        return fn

    return decorator


@register_rule(BED_CAPACITY_RULE_ID)
def _check_bed_capacity(
    graph: nx.DiGraph[str], action: AgentAction, road_network: RoadNetwork | None
) -> tuple[list[ConstraintViolation], AgentAction]:
    if action.patient_transfer is None:
        return [], action
    from_id, to_id, count = action.patient_transfer
    # M8: `effective_free_beds` scales the destination's free-bed count
    # by its current functional_level (`twin/demand.py`'s own docstring)
    # — using the same function `apply_patient_transfer`'s enactment
    # does, so this check's "safe" verdict and what actually happens
    # when the repaired action executes never disagree.
    beds_free = effective_free_beds(graph.nodes[to_id]["asset"])
    if count <= beds_free:
        return [], action

    clipped = max(0, beds_free)
    repaired_transfer = (from_id, to_id, clipped) if clipped > 0 else None
    violation = ConstraintViolation(
        rule_id=BED_CAPACITY_RULE_ID,
        scope="health",
        on_violation="clip",
        detail=(
            f"patient_transfer({from_id} -> {to_id}, {count}) exceeds {beds_free} free "
            f"beds at {to_id}; clipped to {clipped}"
        ),
    )
    return [violation], action.model_copy(update={"patient_transfer": repaired_transfer})


@register_rule(ROUTE_FLOOD_SAFETY_RULE_ID)
def _check_route_flood_safety(
    graph: nx.DiGraph[str], action: AgentAction, road_network: RoadNetwork | None
) -> tuple[list[ConstraintViolation], AgentAction]:
    if not action.ambulance_assignment or road_network is None:
        return [], action

    pending: list[dict[str, object]] = graph.graph.get("pending_requests", [])
    violations: list[ConstraintViolation] = []
    safe_assignment: dict[str, str] = {}

    for ambulance_id, request_id in action.ambulance_assignment.items():
        request = next((r for r in pending if r["request_id"] == request_id), None)
        if request is None:
            # Stale decision — the request is already gone by the time
            # this check runs. Not this rule's concern (see module
            # docstring); leave it as-is, `Simulator.step` no-ops on it.
            safe_assignment[ambulance_id] = request_id
            continue

        home_node = graph.nodes[ambulance_id]["asset"].attributes["home_node"]
        pickup_node = road_network.nearest_node(*request["location"])  # type: ignore[misc]
        max_depth = road_network.shortest_path_max_depth(home_node, pickup_node)

        if max_depth is not None and max_depth < ROUTE_FLOOD_SAFETY_MAX_DEPTH_M:
            safe_assignment[ambulance_id] = request_id
            continue

        reason = "no route exists" if max_depth is None else f"route max depth {max_depth:.2f} m"
        violations.append(
            ConstraintViolation(
                rule_id=ROUTE_FLOOD_SAFETY_RULE_ID,
                scope="transport",
                on_violation="drop",
                detail=(
                    f"dispatch({ambulance_id} -> {request_id}) dropped: {reason} "
                    f"(threshold {ROUTE_FLOOD_SAFETY_MAX_DEPTH_M} m)"
                ),
            )
        )

    repaired = action.model_copy(update={"ambulance_assignment": safe_assignment or None})
    return violations, repaired


@register_rule(POWER_BALANCE_RULE_ID)
def _check_power_balance(
    graph: nx.DiGraph[str], action: AgentAction, road_network: RoadNetwork | None
) -> tuple[list[ConstraintViolation], AgentAction]:
    """Always passes — disclosed no-op, see module docstring: no
    `allocated_mw` decision variable exists in this build for the rule
    to check anything against."""
    return [], action


def check(
    graph: nx.DiGraph[str],
    action: AgentAction,
    *,
    road_network: RoadNetwork | None = None,
) -> ConstraintReport:
    """Dev doc §8's `check(state, joint_action) -> ConstraintReport`.
    `state` is `graph` (the live twin graph the caller already has —
    matches every other `Agent.act`/rule function's own signature in
    this codebase rather than introducing a separate state snapshot
    type just for this call)."""
    all_violations: list[ConstraintViolation] = []
    repaired = action
    for check_fn in _REGISTRY.values():
        violations, repaired = check_fn(graph, repaired, road_network)
        all_violations.extend(violations)
    return ConstraintReport(
        passed=not all_violations, violations=all_violations, repaired_action=repaired
    )
