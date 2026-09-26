"""Thin tool wrappers over twin/constraint-engine APIs (dev doc §7.2's
"Tools exposed"), module M9a.

Real code, no LLM involved — these are exactly the tools dev doc §7.2's
LangGraph flow calls from its non-LLM nodes (`simulate_each`,
`check_constraints`). Implements the subset the mocked graph (`graph.py`)
actually needs today: `get_twin_state_summary`, `simulate_plan`,
`check_constraints`. **Deferred to M9b** (live wiring): `get_asset`,
`get_dependencies`, `get_policy_registry` (thin reads, easy to add once
something calls them), and `get_playbook` (dev doc's own "optional RAG
[...] nice-to-have, not core").
"""

from __future__ import annotations

from udt.common.models import AgentAction, AssetType, ConstraintReport, Incident, SimulationResult
from udt.constraints.engine import check
from udt.twin.counterfactual import simulate
from udt.twin.road_network import RoadNetwork
from udt.twin.simulator import DegradationFn, Simulator

# Same set `logging/metrics.py`/`twin/counterfactual.py` use — duplicated
# rather than imported, same "avoid an unwanted cross-package dependency
# for one frozenset" rationale those modules already disclose.
CRITICAL_TYPES = frozenset({AssetType.HOSPITAL, AssetType.SUBSTATION, AssetType.WATER})


def get_twin_state_summary(sim: Simulator, incident: Incident) -> str:
    """dev doc §7.2's `get_twin_state_summary()` tool — a short, human/
    LLM-readable text summary of the current twin state: the incident's
    type/severity/age, and every critical asset's current functional
    level. Real text built from the live graph, not a placeholder
    string; this is exactly what would be interpolated into a live
    prompt in M9b."""
    lines = [
        f"Incident: {incident.type}, severity={incident.severity:.2f}, "
        f"onset_tick={incident.onset_tick}, current_tick={sim.tick}."
    ]
    for asset_id in sim.graph.nodes:
        asset = sim.graph.nodes[asset_id]["asset"]
        if asset.asset_type in CRITICAL_TYPES:
            lines.append(
                f"- {asset_id} ({asset.asset_type.value}): "
                f"functional_level={asset.functional_level:.2f}"
            )
    return "\n".join(lines)


def simulate_plan(
    sim: Simulator,
    proxy_action: AgentAction,
    *,
    degradation_fn: DegradationFn | None,
    incident: Incident | None,
    horizon_ticks: int | None = None,
    n_rollouts: int | None = None,
    base_seed: int = 0,
) -> SimulationResult:
    """dev doc §7.2's `simulate_plan(plan) -> SimulationResult` tool.

    **Disclosed simplification:** a `Plan` (dev doc §7.3) only conditions
    a goal-weighted MARL policy's reward via `goal_weights`/
    `priority_assets` — it isn't itself a concrete joint action `twin/
    counterfactual.py`'s `simulate()` can execute. The faithful
    implementation would roll a plan's `goal_weights` through a
    *trained* goal-conditioned MAPPO policy (M6b's own machinery), but
    no trained policy at usable quality exists yet (every M5-M7 row's
    own disclosure — training has only run at smoke scale). This tool
    instead simulates whatever concrete `proxy_action` its caller
    supplies — `graph.py`'s `simulate_each` node passes the rule-based
    baseline's own current decision as that stand-in, same "mechanism
    verified, not full-fidelity" status `scripts/calibrate_risk.py`
    already carries for its own use of `simulate()`. This function's own
    job is just the real `simulate()` call — picking `proxy_action` is
    the caller's responsibility, not this tool's."""
    kwargs: dict[str, object] = {"base_seed": base_seed}
    if horizon_ticks is not None:
        kwargs["horizon_ticks"] = horizon_ticks
    if n_rollouts is not None:
        kwargs["n_rollouts"] = n_rollouts
    return simulate(
        sim, proxy_action, degradation_fn=degradation_fn, incident=incident, **kwargs  # type: ignore[arg-type]
    )


def check_constraints_tool(
    sim: Simulator, action: AgentAction, *, road_network: RoadNetwork | None = None
) -> ConstraintReport:
    """dev doc §7.2's `check_constraints(actions) -> ConstraintReport`
    tool — a direct, real wrapper over `constraints/engine.py`'s
    `check()`. Meaningful for the OOD path's `ActionProposal.actions`
    (already concrete `AgentAction`s, dev doc §7.3: "same action
    vocabulary as §5.3, validated feasible") — the ID path's `Plan.
    directives` aren't concrete actions (§7.3: "advisory context [...]
    not executed directly"), so there is nothing for this tool to check
    against a bare `Plan`."""
    return check(sim.graph, action, road_network=road_network)
