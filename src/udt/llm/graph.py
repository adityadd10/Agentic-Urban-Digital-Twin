"""LangGraph planning flow (dev doc §7.2), module M9a — "LangGraph flow
running against recorded/mocked responses" (dev doc §13's "LLM
(recorded)" testing row: "CI never calls a live API").

```
perceive -> summarize_state -> route(ID/OOD from §6, passed in
                                      — the router is code, NOT the LLM)
  ID:  generate_plans(2-3) -> simulate_each(tool) ->
       rank_by_outcome(code, not LLM) -> emit best Plan
  OOD: reason_about_novel_incident -> propose_direct_actions ->
       simulate(tool) -> refine_once_if_P(failure)>0.5 ->
       emit ActionProposal (flagged mandatory-approval)
```

**What M9a builds vs. defers:**
- Every non-LLM node (`perceive`, `summarize_state`, `simulate_each`,
  `rank_by_outcome`, `check_constraints`, `simulate_proposal`) is real
  code calling real twin/constraint-engine functions (`llm/tools.py`) —
  nothing here is a placeholder.
- The two LLM-calling nodes (`generate_plans`, `reason_and_propose` —
  dev doc's "reason_about_novel_incident" and "propose_direct_actions"
  are combined into one call here, disclosed below) go through an
  injected `LLMClient` — an abstract `complete(prompt, node=...) ->
  str` interface. Every automated test in this repo still injects a
  scripted/recorded client (dev doc §13: "CI never calls a live API");
  a real, billed implementation now exists too (M9b's "live wiring") —
  `llm/anthropic_client.py`'s `AnthropicLLMClient`, typically wrapped in
  `llm/cache.py`'s `CachedLLMClient` — see `scripts/run_planner_demo.py`
  for the one place in this codebase that actually constructs one.
- `routing_path` ("ID"/"OOD") is read directly off the state dict, not
  computed inside this graph — dev doc's own wording: "the router is
  code, NOT the LLM". `routing/ood.py`'s `route(...)` (dev doc §6, M9b)
  is that code — a caller runs it first and passes the resulting
  `RoutingDecision.path` in, same as `scripts/run_planner_demo.py`
  does; the graph itself stays agnostic to how `routing_path` was
  decided.
- `reason_about_novel_incident` + `propose_direct_actions` are merged
  into one LLM call (`reason_and_propose_node`) rather than two separate
  graph nodes — the dev doc's own diagram doesn't specify what state
  `reason_about_novel_incident` returns that `propose_direct_actions`
  needs beyond a plain continuation of the same reasoning, and splitting
  them into two round-trips would double real API cost/latency for no
  structural benefit this prototype needs; disclosed rather than
  silently collapsed.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, TypedDict, TypeVar

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ValidationError

from udt.agents.plan_interpreter import interpret_plan
from udt.common.models import AgentAction, ConstraintReport, Incident, SimulationResult
from udt.llm.schemas import ActionProposal, GoalWeights, Plan
from udt.llm.tools import check_constraints_tool, get_twin_state_summary, simulate_plan
from udt.twin.road_network import RoadNetwork
from udt.twin.simulator import DegradationFn, Simulator

# dev doc §7.1 exactly: "max 3 attempts".
MAX_LLM_ATTEMPTS = 3
# dev doc §7.2: "generate_plans(2-3)".
N_PLANS_DEFAULT = 2
# dev doc §9.3's own P(failure) threshold, reused here for §7.2's
# "refine_once_if_P(failure)>0.5" — same number, same meaning, kept as
# its own named constant in this module rather than importing `risk/`'s
# (`risk/human_model.py`'s `DEFAULT_APPROVE_IF_PFAIL_LT`) since the two
# concepts are only coincidentally the same value, not the same thing.
REFINE_P_FAILURE_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# LLM client interface (M9b builds the real, Anthropic-backed one)
# ---------------------------------------------------------------------------
class LLMClient(Protocol):
    """Returns raw text (expected to be JSON) for a prompt. `node` names
    which graph node is calling — a recorded/cached client uses it to
    pick the right canned response; a live client (M9b) uses it for
    prompt-version/cache-key bookkeeping (dev doc §7.1: cache keyed on
    `(prompt_version, model, state_digest)`)."""

    def complete(self, prompt: str, *, node: str) -> str: ...


_T = TypeVar("_T", bound=BaseModel)


def _generate_structured(
    client: LLMClient, node: str, prompt: str, schema: type[_T]
) -> tuple[_T | None, bool]:
    """dev doc §7.1: "Every LLM output is structured JSON validated
    against a Pydantic schema; on validation failure retry with the
    error message appended, max 3 attempts, then fall back [...] and
    log llm_fallback=true." Returns `(parsed, fell_back)` —
    `parsed is None` iff `fell_back` is `True`."""
    error: str | None = None
    for _attempt in range(MAX_LLM_ATTEMPTS):
        current_prompt = prompt
        if error is not None:
            current_prompt = (
                f"{prompt}\n\nYour previous response was invalid for this schema: "
                f"{error}\nRespond again with ONLY valid JSON matching the schema."
            )
        raw = client.complete(current_prompt, node=node)
        try:
            data = json.loads(raw)
            return schema.model_validate(data), False
        except (json.JSONDecodeError, ValidationError) as exc:
            error = str(exc)
    return None, True


class _PlanBatch(BaseModel):
    """Internal-only wrapper so one LLM call returns dev doc §7.2's
    "generate_plans(2-3)" as a single validated response — not one of
    §7.3's own named schemas."""

    plans: list[Plan]


def _fallback_plan() -> Plan:
    """dev doc §7.1: "fall back to the rule-based baseline plan". All-`1.0`
    weights, no priority assets, no directives — `plan_interpreter.
    interpret_plan` degenerates this exactly to `RuleBasedAgent.act`'s own
    decision (dev doc §7.3.1), which is what "rule-based baseline
    behavior" now concretely means for a `Plan`, not just its
    `goal_weights` in isolation (that was true before §7.3.1 existed;
    `directives`/`priority_assets` have real executable meaning now)."""
    return Plan(
        plan_id="fallback_rule_based",
        objective="Fallback: LLM planning unavailable or invalid; default to unconditioned policy.",
        goal_weights=GoalWeights(g_health=1.0, g_power=1.0, g_transport=1.0, g_cost=1.0),
        priority_assets=[],
        directives=[],
        rationale="llm_fallback=true after exhausting retries.",
        assumptions=[],
        expected_outcome="Same behavior as an unconditioned (g=1.0) policy.",
    )


def _fallback_action_proposal() -> ActionProposal:
    """dev doc §7.1's fallback rule, applied to the OOD path: no actions
    proposed (an all-`None` `AgentAction`, `risk/gate.py`'s own
    `safe_hold_action()` shape) rather than guessing."""
    return ActionProposal(
        actions=[AgentAction()],
        rationale="llm_fallback=true after exhausting retries.",
        confidence_note="n/a",
    )


def _merge_actions(actions: list[AgentAction]) -> AgentAction:
    """`ActionProposal.actions` is a list (dev doc §7.3), but `twin/
    counterfactual.py`'s `simulate()` and `constraints/engine.py`'s
    `check()` both take one joint `AgentAction` — merged here,
    first-non-`None`-wins per field, disclosed rather than silently
    picking just `actions[0]` or raising on more than one entry."""
    merged = AgentAction()
    for action in actions:
        if merged.repair_target is None and action.repair_target is not None:
            merged.repair_target = action.repair_target
        if merged.ambulance_assignment is None and action.ambulance_assignment is not None:
            merged.ambulance_assignment = action.ambulance_assignment
        if merged.patient_transfer is None and action.patient_transfer is not None:
            merged.patient_transfer = action.patient_transfer
        if merged.shed_tier is None and action.shed_tier is not None:
            merged.shed_tier = action.shed_tier
    return merged


# ---------------------------------------------------------------------------
# Prompt building (dev doc §7.1: "All prompts live in src/udt/llm/
# prompts/*.md [...] never inline in Python")
# ---------------------------------------------------------------------------
_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text()


def _build_generate_plans_prompt(twin_summary: str, incident: Incident, n_plans: int) -> str:
    return _load_prompt("generate_plans.md").format(
        twin_summary=twin_summary,
        incident_type=incident.type,
        severity=incident.severity,
        n_plans=n_plans,
    )


def _build_propose_actions_prompt(
    twin_summary: str, incident: Incident, prior_result: SimulationResult | None
) -> str:
    feedback_section = ""
    if prior_result is not None:
        feedback_section = (
            f"\nYour previous proposal was simulated and came back with "
            f"P(failure)={prior_result.p_failure:.2f} (> {REFINE_P_FAILURE_THRESHOLD} — too "
            f"risky). Revise your proposal to reduce failure risk.\n"
        )
    return _load_prompt("propose_direct_actions.md").format(
        twin_summary=twin_summary,
        incident_type=incident.type,
        severity=incident.severity,
        feedback_section=feedback_section,
    )


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------
# `StateGraph(PlannerState)` resolves every annotation below via
# `typing.get_type_hints`, which *fully evaluates* every forward
# reference — unlike everywhere else in this codebase, where `from
# __future__ import annotations` lets `twin/simulator.py`'s `DegradationFn
# = Callable[[int, "nx.DiGraph[str]"], ...]` keep its `nx.DiGraph[str]`
# forward ref forever unevaluated (that module's own docstring explains
# why: `nx.DiGraph` isn't actually subscriptable at runtime). Using
# `DegradationFn` as a field's annotation here would make langgraph choke
# on exactly that. `degradation_fn` below is deliberately typed as a
# self-contained `Callable[[int, Any], ...]` instead — functionally the
# same callable `run_planner`'s own signature (which nothing calls
# `get_type_hints` on) still types precisely as `DegradationFn`.
class PlannerState(TypedDict, total=False):
    llm_client: LLMClient
    sim: Simulator
    incident: Incident
    routing_path: Literal["ID", "OOD"]
    degradation_fn: Callable[[int, Any], dict[str, float]] | None
    road_network: RoadNetwork | None

    twin_summary: str
    plans: list[Plan]
    simulation_results: dict[str, SimulationResult]
    selected_plan: Plan | None
    action_proposal: ActionProposal | None
    constraint_report: ConstraintReport | None
    checked_action: AgentAction
    """The OOD path's `action_proposal` after `check_constraints`
    repaired it — what `simulate_proposal` actually simulates and (in a
    full deployment) what would actually execute."""
    llm_fallback: bool
    refine_count: int


@dataclass
class PlannerOutput:
    """What a caller actually wants out of one graph run — pulled out of
    the raw `PlannerState` dict at the end, so callers don't have to know
    the state's internal key names."""

    routing_path: Literal["ID", "OOD"]
    plans: list[Plan] = field(default_factory=list)
    """Every ID-path candidate `generate_plans` produced (dev doc §10's
    `plans[{plan, simulation_summary}]` wants all of them, not just the
    winner) — always empty on the OOD path, which never has candidate
    `Plan`s to begin with."""
    selected_plan: Plan | None = None
    action_proposal: ActionProposal | None = None
    constraint_report: ConstraintReport | None = None
    llm_fallback: bool = False
    simulation_results: dict[str, SimulationResult] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def _perceive(state: PlannerState) -> dict[str, Any]:
    assert "sim" in state and "incident" in state, "PlannerState needs sim + incident to start"
    return {}


def _summarize_state(state: PlannerState) -> dict[str, Any]:
    return {"twin_summary": get_twin_state_summary(state["sim"], state["incident"])}


def _route(state: PlannerState) -> Literal["ID", "OOD"]:
    return state["routing_path"]


def _plan_diversity_signature(plan: Plan) -> tuple[frozenset[str], frozenset[str]]:
    """dev doc §7.3's diversity requirement: two plans count as the same
    candidate if `plan_interpreter.interpret_plan` would be equally
    likely to treat them the same — i.e. same `priority_assets` set and
    same set of directive `kind`s — regardless of `goal_weights`
    magnitude or `directives[].details`/`plan_id`/prose fields, none of
    which alone changes which lever the interpreter pulls."""
    return (
        frozenset(plan.priority_assets),
        frozenset(directive.kind for directive in plan.directives),
    )


def _plans_are_diverse(plans: list[Plan]) -> bool:
    if len(plans) < 2:
        return True
    signatures = {_plan_diversity_signature(p) for p in plans}
    return len(signatures) == len(plans)


def _generate_plans(state: PlannerState) -> dict[str, Any]:
    """dev doc §7.3's candidate-diversity requirement: retry generation
    (same `MAX_LLM_ATTEMPTS` budget §7.1 already uses for schema
    validity — up to `MAX_LLM_ATTEMPTS` regenerations, each internally
    retried up to `MAX_LLM_ATTEMPTS` times for schema validity by
    `_generate_structured`, so bounded at `MAX_LLM_ATTEMPTS**2` LLM calls
    worst case) if the candidates aren't structurally distinct — two
    plans that would interpret to the same action have nothing for
    `simulate_each`/`rank_by_outcome` to actually compare. Exhausting
    retries without diverse candidates falls back the same way an
    unparseable response does: one plan, nothing to compare, logged."""
    prompt = _build_generate_plans_prompt(state["twin_summary"], state["incident"], N_PLANS_DEFAULT)
    for _attempt in range(MAX_LLM_ATTEMPTS):
        batch, fell_back = _generate_structured(
            state["llm_client"], "generate_plans", prompt, _PlanBatch
        )
        if fell_back or batch is None or not batch.plans:
            return {"plans": [_fallback_plan()], "llm_fallback": True}
        plans = batch.plans[:3]
        if _plans_are_diverse(plans):
            return {"plans": plans, "llm_fallback": False}
    return {"plans": [_fallback_plan()], "llm_fallback": True}


def _simulate_each(state: PlannerState) -> dict[str, Any]:
    """Each candidate is interpreted (`plan_interpreter.interpret_plan`,
    dev doc §7.3.1) against the *same* live `sim` state before being
    simulated — this is what makes the comparison real: two candidates
    that differ in `priority_assets`/`directives` now generally produce
    different concrete `AgentAction`s and therefore different simulated
    outcomes, instead of every candidate silently testing an identical
    stand-in action (the bug this fixes, `MTP_Module_Planner.md`'s
    2026-09-22 M9a entry)."""
    sim = state["sim"]
    road_network = state.get("road_network")
    results: dict[str, SimulationResult] = {}
    for plan in state["plans"]:
        action = interpret_plan(
            plan, sim.graph, road_network=road_network, edge_states=sim.edge_states
        )
        results[plan.plan_id] = simulate_plan(
            sim,
            action,
            degradation_fn=state.get("degradation_fn"),
            incident=state.get("incident"),
        )
    return {"simulation_results": results}


def _rank_by_outcome(state: PlannerState) -> dict[str, Any]:
    results = state["simulation_results"]
    plans_by_id = {p.plan_id: p for p in state["plans"]}
    # dev doc: "rank_by_outcome(code, not LLM)" - lowest P(failure) first,
    # mean outcome (unmet_patient_hours) as the tiebreaker.
    best_id = min(results, key=lambda pid: (results[pid].p_failure, results[pid].mean_outcome))
    return {"selected_plan": plans_by_id[best_id]}


def _reason_and_propose(state: PlannerState) -> dict[str, Any]:
    prior_result = state.get("simulation_results", {}).get("proposal")
    prompt = _build_propose_actions_prompt(state["twin_summary"], state["incident"], prior_result)
    proposal, fell_back = _generate_structured(
        state["llm_client"], "propose_direct_actions", prompt, ActionProposal
    )
    refine_count = state.get("refine_count", 0) + (1 if prior_result is not None else 0)
    if fell_back or proposal is None:
        return {
            "action_proposal": _fallback_action_proposal(),
            "llm_fallback": True,
            "refine_count": refine_count,
        }
    return {"action_proposal": proposal, "llm_fallback": False, "refine_count": refine_count}


def _check_constraints(state: PlannerState) -> dict[str, Any]:
    proposal = state["action_proposal"]
    assert proposal is not None
    merged = _merge_actions(proposal.actions)
    report = check_constraints_tool(state["sim"], merged, road_network=state.get("road_network"))
    return {"constraint_report": report, "checked_action": report.repaired_action}


def _simulate_proposal(state: PlannerState) -> dict[str, Any]:
    action = state["checked_action"]
    result = simulate_plan(
        state["sim"],
        action,
        degradation_fn=state.get("degradation_fn"),
        incident=state.get("incident"),
    )
    existing = dict(state.get("simulation_results", {}))
    existing["proposal"] = result
    return {"simulation_results": existing}


def _refine_or_done(state: PlannerState) -> Literal["refine", "done"]:
    result = state["simulation_results"].get("proposal")
    if (
        result is not None
        and result.p_failure > REFINE_P_FAILURE_THRESHOLD
        and state.get("refine_count", 0) < 1
    ):
        return "refine"
    return "done"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def build_planner_graph() -> Any:
    """Compiles dev doc §7.2's flow. The `llm_client` (and `sim`/
    `incident`/`routing_path`/...) are supplied per-invocation via the
    initial state dict passed to `.invoke(...)`, not baked into the
    graph at build time — the graph itself is stateless/reusable across
    many decisions."""
    graph: StateGraph[PlannerState] = StateGraph(PlannerState)

    graph.add_node("perceive", _perceive)
    graph.add_node("summarize_state", _summarize_state)
    graph.add_node("generate_plans", _generate_plans)
    graph.add_node("simulate_each", _simulate_each)
    graph.add_node("rank_by_outcome", _rank_by_outcome)
    graph.add_node("reason_and_propose", _reason_and_propose)
    graph.add_node("check_constraints", _check_constraints)
    graph.add_node("simulate_proposal", _simulate_proposal)

    graph.add_edge(START, "perceive")
    graph.add_edge("perceive", "summarize_state")
    graph.add_conditional_edges(
        "summarize_state", _route, {"ID": "generate_plans", "OOD": "reason_and_propose"}
    )

    graph.add_edge("generate_plans", "simulate_each")
    graph.add_edge("simulate_each", "rank_by_outcome")
    graph.add_edge("rank_by_outcome", END)

    graph.add_edge("reason_and_propose", "check_constraints")
    graph.add_edge("check_constraints", "simulate_proposal")
    graph.add_conditional_edges(
        "simulate_proposal", _refine_or_done, {"refine": "reason_and_propose", "done": END}
    )

    return graph.compile()


def run_planner(
    compiled_graph: Any,
    *,
    llm_client: LLMClient,
    sim: Simulator,
    incident: Incident,
    routing_path: Literal["ID", "OOD"],
    degradation_fn: DegradationFn | None = None,
    road_network: RoadNetwork | None = None,
) -> PlannerOutput:
    """Convenience entry point: builds the initial `PlannerState`, runs
    the compiled graph, and unpacks the result into a `PlannerOutput`.

    No `proxy_action` parameter (removed 2026-09-22, dev doc §7.3.1): the
    ID path now derives each candidate's simulated action from the plan
    itself (`_simulate_each` -> `plan_interpreter.interpret_plan`), and
    the OOD path never used a proxy action to begin with (`checked_action`
    comes from the LLM's own `ActionProposal`, post-constraint-repair)."""
    initial_state: PlannerState = {
        "llm_client": llm_client,
        "sim": sim,
        "incident": incident,
        "routing_path": routing_path,
        "degradation_fn": degradation_fn,
        "road_network": road_network,
    }
    final_state = compiled_graph.invoke(initial_state)
    return PlannerOutput(
        routing_path=routing_path,
        plans=final_state.get("plans", []),
        selected_plan=final_state.get("selected_plan"),
        action_proposal=final_state.get("action_proposal"),
        constraint_report=final_state.get("constraint_report"),
        llm_fallback=final_state.get("llm_fallback", False),
        simulation_results=final_state.get("simulation_results", {}),
    )
