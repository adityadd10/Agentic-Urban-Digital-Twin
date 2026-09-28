"""Stateful, incrementally-steppable episode session (dev doc §10/§11),
module M10a/M10b.

`EpisodeSession` is the real engine behind both `experiments/
decision_loop.py`'s headless whole-episode runner (M10a) and M10b's
FastAPI `POST /episodes/{id}/step` endpoint — the same "route -> plan ->
execute -> constraint-check -> risk-assess -> gate -> log" loop
`experiments/decision_loop.py` originally implemented as one big
function now lives here, factored into a class with a `step(n_ticks)`
method, so a live API server can advance an episode a few ticks at a
time between requests instead of only ever running a whole episode in
one blocking call.

See `experiments/decision_loop.py`'s own (much shorter) module
docstring for the disclosed scope this shares: `PolicyExecutor`'s one
real implementation (`RuleBasedPolicyExecutor`, ignores goal weights),
`critic_returns` as a caller-supplied stand-in (no 5 trained MAPPO
seeds exist), and the same tick-physics ordering requirement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from udt.agents.plan_interpreter import interpret_plan
from udt.agents.rule_based import RuleBasedAgent
from udt.common.models import (
    AgentAction,
    AssetType,
    DependencyGraph,
    Incident,
    PolicyRegistryEntry,
    RiskCalibration,
    RouterCalibration,
    SimulationResult,
    TwinState,
)
from udt.constraints.engine import check
from udt.llm.graph import LLMClient, build_planner_graph, run_planner
from udt.llm.schemas import Plan
from udt.logging.decision_log import (
    ApprovalSummary,
    DecisionLogWriter,
    DecisionRecord,
    LLMCallSummary,
    MarlActionSummary,
    PlanSummary,
    RiskSummary,
    build_decision_id,
    twin_state_digest,
)
from udt.risk.engine import assess
from udt.risk.gate import decide_tier, safe_hold_action
from udt.risk.human_model import HumanModel
from udt.routing.ood import route
from udt.twin.ambulances import generate_requests
from udt.twin.cascade import EdgeRuntimeState
from udt.twin.counterfactual import HORIZON_TICKS_DEFAULT
from udt.twin.power import update_substation_load
from udt.twin.road_network import RoadNetwork
from udt.twin.simulator import DegradationFn, Simulator

# dev doc §5.1 exactly - matches every other harness's own constant of
# the same name/value.
DECISION_INTERVAL_TICKS = 3

# dev doc §7.1 exactly: the LLM (route + generate_plans/reason_and_propose)
# is called "only at incident onset and at re-plan triggers (a plan's
# simulated expectation diverges from reality by >30%, or every 2
# simulated hours) — never per tick." 2 simulated hours at dt=5min = 24
# ticks. Added 2026-09-22 — before this, `_make_decision` called the full
# planner every `DECISION_INTERVAL_TICKS`, which silently collapsed the
# strategic (LLM) and tactical (policy-executor) timescales into the same
# cadence and meant a "selected plan" never persisted across more than one
# tactical decision (found during a project-review session, not by a
# test failing — see `MTP_Module_Planner.md`'s M9a/M10a entries).
REPLAN_INTERVAL_TICKS = 24
# dev doc §7.1's own "> 30%" divergence trigger.
DIVERGENCE_THRESHOLD = 0.30
# Disclosed absolute floor (patient-hours) for the divergence check —
# `DIVERGENCE_THRESHOLD` alone is a *proportional* test and is meaningless
# when a plan predicted ~0 unmet_patient_hours (any realized value would
# "diverge" by an undefined/infinite percentage); this floor gives the
# "predicted safe, isn't" case a way to still trigger a replan. Not
# calibrated against real outcome data — same disclosure tier as
# `plan_interpreter.SHED_BIAS_DELTA`.
DIVERGENCE_ABS_FLOOR_HOURS = 2.0


@dataclass
class PolicyDecision:
    action: AgentAction
    policy_id: str
    masked_actions_count: int = 0


class PolicyExecutor(Protocol):
    def decide(
        self,
        sim: Simulator,
        road_network: RoadNetwork | None,
        plan: Plan | None,
        edge_states: dict[str, EdgeRuntimeState] | None,
    ) -> PolicyDecision: ...


@dataclass
class RuleBasedPolicyExecutor:
    """The one real, tested `PolicyExecutor` this module ships. Until
    2026-09-22 this **ignored `plan` (then `goal_weights`) entirely** —
    disclosed as a real gap, not a permanent design choice, since
    `RuleBasedAgent` (dev doc §5.6) isn't goal-conditioned (that's a
    MARL-specific mechanism, M6b, not trained to usable quality yet).
    Now: a given `plan` is interpreted (`agents/plan_interpreter.py`, dev
    doc §7.3.1) into a real, plan-biased `AgentAction` instead of being
    dropped — the honest bridge until a `mappo_constrained`-backed
    executor can be plugged in for real (none exists at usable quality
    yet, every M5-M7 row's own disclosure)."""

    agent: RuleBasedAgent = field(default_factory=RuleBasedAgent)
    policy_id: str = "rule_based"

    def decide(
        self,
        sim: Simulator,
        road_network: RoadNetwork | None,
        plan: Plan | None,
        edge_states: dict[str, EdgeRuntimeState] | None,
    ) -> PolicyDecision:
        if plan is not None:
            action = interpret_plan(
                plan, sim.graph, road_network=road_network, edge_states=edge_states
            )
        else:
            action = self.agent.act(
                sim.graph, sim.tick, road_network=road_network, edge_states=edge_states
            )
        return PolicyDecision(action=action, policy_id=self.policy_id, masked_actions_count=0)


def _merge_actions(actions: list[AgentAction]) -> AgentAction:
    """Same first-non-`None`-wins merge `llm/graph.py`'s `_check_
    constraints` node already uses for `ActionProposal.actions` — needed
    again here since this session calls `constraints/engine.py`'s
    `check()` itself, on whichever action (ID or OOD) is about to
    execute, rather than relying on the planner graph's own internal
    repair (dev doc §8 usage (2): "all experiments, all paths")."""
    merged = AgentAction()
    for a in actions:
        if merged.repair_target is None and a.repair_target is not None:
            merged.repair_target = a.repair_target
        if merged.ambulance_assignment is None and a.ambulance_assignment is not None:
            merged.ambulance_assignment = a.ambulance_assignment
        if merged.patient_transfer is None and a.patient_transfer is not None:
            merged.patient_transfer = a.patient_transfer
        if merged.shed_tier is None and a.shed_tier is not None:
            merged.shed_tier = a.shed_tier
    return merged


class EpisodeSession:
    """One running episode's full state — construct once, call `step()`
    as many times as needed (once per tick or a batch at a time)."""

    def __init__(
        self,
        *,
        episode_id: str,
        sim: Simulator,
        incident: Incident,
        dep_graph: DependencyGraph,
        degradation_fn: DegradationFn | None,
        road_network: RoadNetwork | None,
        ward_polygon: object | None,
        llm_client: LLMClient,
        policy_executor: PolicyExecutor,
        env_version: str,
        registry_entries: list[PolicyRegistryEntry],
        router_calibration: RouterCalibration | None,
        risk_calibration: RiskCalibration,
        critic_returns: list[float],
        human_model: HumanModel,
        decision_log_path: str | Path,
        n_ticks_max: int = 288,
    ) -> None:
        self.episode_id = episode_id
        self.sim = sim
        self.incident = incident
        self.dep_graph = dep_graph
        self.degradation_fn = degradation_fn
        self.road_network = road_network
        self.ward_polygon = ward_polygon
        self.llm_client = llm_client
        self.policy_executor = policy_executor
        self.env_version = env_version
        self.registry_entries = registry_entries
        self.router_calibration = router_calibration
        self.risk_calibration = risk_calibration
        self.critic_returns = critic_returns
        self.human_model = human_model
        self.n_ticks_max = n_ticks_max  # dev doc §3.1: T_max = 24h at dt=5min -> 288

        self.trace: list[TwinState] = []
        self.decisions: list[DecisionRecord] = []
        self._writer = DecisionLogWriter(decision_log_path)
        self._planner_graph = build_planner_graph()
        self._current_action: AgentAction | None = None

        # Persisted strategic state (2026-09-22 addition, dev doc §7.1/
        # §7.3.1) — what a "selected plan" governs between replans, and
        # what the divergence check compares reality against. `None`/-1
        # until the first decision tick forces a replan (dev doc §7.1:
        # "at incident onset").
        self._routing_path: str | None = None
        self._current_plan: Plan | None = None
        """ID path only — the persisted, currently-governing `Plan`."""
        self._current_sim_result: SimulationResult | None = None
        """Whichever path is active: the selected plan's (ID) or the
        last OOD proposal's `SimulationResult` — reused for risk
        assessment and the divergence check between replans, rather than
        re-running a full Monte-Carlo `simulate()` every tactical tick."""
        self._llm_fallback: bool = False
        self._replan_tick: int = -1
        self._unmet_patient_hours_since_replan: float = 0.0

    @property
    def ended(self) -> bool:
        return self.sim.tick >= self.n_ticks_max

    def inject_incident(self, incident: Incident, degradation_fn: DegradationFn | None) -> None:
        """dev doc §11.2's `POST /incidents`: "inject incident (demo:
        'what if X happens now')" — replaces the incident/degradation_fn
        driving this session's ongoing simulation, from the next `step()`
        call forward. Does not rewind or replay past ticks."""
        self.incident = incident
        self.degradation_fn = degradation_fn

    @property
    def latest_state(self) -> TwinState | None:
        return self.trace[-1] if self.trace else None

    def step(self, n_ticks: int = 1) -> list[TwinState]:
        """Advances the episode by `n_ticks` raw simulator ticks (dev
        doc §11.2's own `POST /episodes/{id}/step?n=1` naming) —
        deciding a fresh *tactical* action every `DECISION_INTERVAL_
        TICKS`-th tick, same as every other harness in this codebase.
        **Not** the same as replanning: `_make_decision` only re-invokes
        the LLM planner every `REPLAN_INTERVAL_TICKS` (or sooner, on
        divergence) — most decision ticks reuse the persisted strategic
        plan and just re-derive a concrete tactical action against the
        current state (dev doc §7.1/§7.3.1). Returns just the snapshots
        produced by *this* call, not the whole trace so far. Silently
        capped so the episode never advances past `n_ticks_max` —
        calling `step()` again once `self.ended` is `True` is a
        harmless no-op (returns `[]`), not an error."""
        n_ticks = max(0, min(n_ticks, self.n_ticks_max - self.sim.tick))
        new_snapshots: list[TwinState] = []
        for _ in range(n_ticks):
            update_substation_load(
                self.sim.graph, self.sim.tick, self.incident, self.sim.dt_minutes
            )
            if self.road_network is not None and self.ward_polygon is not None:
                self.road_network.update_for_tick(self.incident, self.sim.tick, self.sim.dt_minutes)
                generate_requests(
                    self.sim.graph,
                    self.sim.tick,
                    self.sim.dt_hours,
                    self.incident,
                    self.ward_polygon,
                    self.sim.rng,
                )

            is_decision_tick = self.sim.tick % DECISION_INTERVAL_TICKS == 0
            if is_decision_tick:
                record, self._current_action = self._make_decision()
                self._writer.append(record)
                self.decisions.append(record)

            assert self._current_action is not None  # tick 0 is always a decision tick
            snapshot = self.sim.step(
                degradation_fn=self.degradation_fn,
                repair_target=self._current_action.repair_target if is_decision_tick else None,
                ambulance_assignment=(
                    self._current_action.ambulance_assignment if is_decision_tick else None
                ),
                ambulance_destination=(
                    self._current_action.ambulance_destination if is_decision_tick else None
                ),
                patient_transfer=(
                    self._current_action.patient_transfer if is_decision_tick else None
                ),
                shed_tier=self._current_action.shed_tier if is_decision_tick else None,
            )
            self.trace.append(snapshot)
            new_snapshots.append(snapshot)

            # Realized `unmet_patient_hours` this tick, accumulated since
            # the last replan — the divergence check's "reality" side.
            # Same per-tick computation `twin/counterfactual.py`'s
            # `_run_one_rollout` uses for the "prediction" side, so the
            # two are actually comparable.
            queued_this_tick = sum(
                int(asset.attributes.get("patient_queue", 0))
                for asset in snapshot.assets
                if asset.asset_type == AssetType.HOSPITAL
            )
            self._unmet_patient_hours_since_replan += queued_this_tick * self.sim.dt_hours
        return new_snapshots

    def _is_replan_due(self, routing_path: str) -> bool:
        """dev doc §7.1: replan "at incident onset and at re-plan
        triggers (>30% divergence, or every 2 simulated hours)". A
        routing-path change (e.g. the incident got worse and now fails
        the OOD router's support check) also forces a replan — reusing
        an ID-path `Plan` after routing has flipped to OOD (or vice
        versa) would silently execute a strategy chosen for a situation
        the router no longer thinks it's in."""
        if self._replan_tick < 0 or routing_path != self._routing_path:
            return True
        if self.sim.tick - self._replan_tick >= REPLAN_INTERVAL_TICKS:
            return True
        return self._divergence_exceeded()

    def _divergence_exceeded(self) -> bool:
        """Disclosed, simplified proxy for dev doc §7.1's "a plan's
        simulated expectation diverges from reality by >30%": compares
        realized `unmet_patient_hours` since the last replan against the
        selected plan/proposal's own predicted `mean_outcome`, prorated
        by how much of its `HORIZON_TICKS_DEFAULT` rollout window has
        elapsed. `DIVERGENCE_ABS_FLOOR_HOURS` covers the case a plan
        predicted ~0 and reality isn't, which a pure percentage test
        can't express."""
        if self._current_sim_result is None:
            return False
        ticks_elapsed = self.sim.tick - self._replan_tick
        if ticks_elapsed <= 0:
            return False
        expected_by_now = self._current_sim_result.mean_outcome * min(
            1.0, ticks_elapsed / HORIZON_TICKS_DEFAULT
        )
        threshold = max(expected_by_now * (1.0 + DIVERGENCE_THRESHOLD), DIVERGENCE_ABS_FLOOR_HOURS)
        return self._unmet_patient_hours_since_replan > threshold

    def _make_decision(self) -> tuple[DecisionRecord, AgentAction]:
        sim = self.sim
        decision_id = build_decision_id(self.episode_id, sim.tick)

        # Routing is cheap, non-LLM code (dev doc §6) - re-checked every
        # decision tick regardless of replan cadence, since an injected
        # incident update can change ID/OOD eligibility mid-episode.
        routing_decision = route(
            self.incident,
            self.dep_graph,
            env_version=self.env_version,
            registry_entries=self.registry_entries,
            calibration=self.router_calibration,
            onset_hour_of_day=sim.onset_hour_of_day,
        )

        plans_summary: list[PlanSummary] = []
        action_proposal = None
        if self._is_replan_due(routing_decision.path):
            # The expensive, LLM-calling step (dev doc §7.1: "only at
            # incident onset and at re-plan triggers [...] never per
            # tick") - everything below this block runs every decision
            # tick regardless; only this part is gated.
            planner_output = run_planner(
                self._planner_graph,
                llm_client=self.llm_client,
                sim=sim,
                incident=self.incident,
                routing_path=routing_decision.path,
                degradation_fn=self.degradation_fn,
                road_network=self.road_network,
            )
            plans_summary = [
                PlanSummary(
                    plan=plan, simulation_summary=planner_output.simulation_results[plan.plan_id]
                )
                for plan in planner_output.plans
                if plan.plan_id in planner_output.simulation_results
            ]
            self._routing_path = routing_decision.path
            self._llm_fallback = planner_output.llm_fallback
            self._replan_tick = sim.tick
            self._unmet_patient_hours_since_replan = 0.0
            if routing_decision.path == "ID":
                self._current_plan = planner_output.selected_plan
                self._current_sim_result = (
                    planner_output.simulation_results.get(planner_output.selected_plan.plan_id)
                    if planner_output.selected_plan
                    else None
                )
            else:
                self._current_plan = None
                self._current_sim_result = planner_output.simulation_results.get("proposal")
                # An OOD `ActionProposal` is this replan tick's concrete
                # action only (dev doc §7.1) - it doesn't get replayed on
                # later, non-replan ticks the way an ID `Plan` persists,
                # since replaying a stale concrete action (rather than a
                # re-interpretable strategy) isn't meaningfully safer
                # than the rule-based default. See the `else` branch
                # below.
                action_proposal = planner_output.action_proposal

        if routing_decision.path == "ID":
            marl_decision = self.policy_executor.decide(
                sim, self.road_network, self._current_plan, sim.edge_states
            )
            candidate_action = marl_decision.action
            selected_plan_id = self._current_plan.plan_id if self._current_plan else None
        else:
            marl_decision = PolicyDecision(
                action=AgentAction(), policy_id="none", masked_actions_count=0
            )
            if action_proposal is not None:
                candidate_action = _merge_actions(action_proposal.actions)
            else:
                # Not this tick's replan - no fresh OOD reasoning and no
                # persisted concrete action to fall back to either;
                # OOD-by-definition means no trained/interpretable policy
                # is trustworthy here, so the safe default is the plain
                # rule-based decision, not silently repeating stale LLM
                # output.
                candidate_action = self.policy_executor.decide(
                    sim, self.road_network, None, sim.edge_states
                ).action
            selected_plan_id = None
        sim_result = self._current_sim_result

        # M7: post-hoc validation on whatever's about to execute (dev
        # doc §8 usage (2): "all experiments, all paths") - regardless
        # of which path produced `candidate_action`.
        constraint_report = check(sim.graph, candidate_action, road_network=self.road_network)
        checked_action = constraint_report.repaired_action

        if sim_result is None:
            # No SimulationResult to assess (e.g. the LLM fell back
            # with nothing simulated) - a disclosed neutral default,
            # not a claim of zero risk.
            sim_result = SimulationResult(
                rollouts=[], p_failure=0.0, mean_outcome=0.0, std_outcome=0.0
            )
        risk_assessment = assess(sim_result, self.critic_returns, self.risk_calibration)

        tier = decide_tier(risk_assessment.risk, risk_assessment.confidence, routing_decision.path)
        approval_required = tier.value == "HUMAN_APPROVAL"
        if approval_required:
            approved = self.human_model.decide(risk_assessment.p_failure)
            final_action = checked_action if approved else safe_hold_action()
            approval_response = "approve" if approved else "reject"
        else:
            final_action = checked_action
            approval_response = None

        record = DecisionRecord(
            decision_id=decision_id,
            episode_id=self.episode_id,
            tick=sim.tick,
            twin_state_digest=twin_state_digest(sim),
            incident_id=self.incident.incident_id,
            routing=routing_decision,
            llm=LLMCallSummary(
                prompt_version=None,
                model=None,
                tokens=0,
                cache_hit=False,
                fallback=self._llm_fallback,
            ),
            plans=plans_summary,
            selected_plan_id=selected_plan_id,
            marl=MarlActionSummary(
                policy_id=marl_decision.policy_id,
                action=candidate_action.model_dump(),
                masked_actions_count=marl_decision.masked_actions_count,
            ),
            constraint_report=constraint_report,
            risk=RiskSummary(
                p_failure=risk_assessment.p_failure,
                consequence=risk_assessment.consequence,
                risk=risk_assessment.risk,
                confidence=risk_assessment.confidence,
                tier=tier,
            ),
            approval=ApprovalSummary(
                required=approval_required,
                response=approval_response,
                latency=0.0,
                timeout_triggered=False,
            ),
            executed_action=final_action,
            realized_outcome_next_interval=None,
        )
        return record, final_action
