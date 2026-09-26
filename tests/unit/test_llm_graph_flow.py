"""Phase 9 acceptance tests: the full compiled LangGraph flow
(`llm.graph.build_planner_graph`/`run_planner`), dev doc §7.2, module
M9a — "LangGraph flow running against recorded/mocked responses" (dev
doc §13's own testing-strategy row; this file IS that test).

Hand-built graphs (same convention as `test_counterfactual.py`), not the
real Kurla data — the graph's own wiring/branching is what's under test
here, not the twin's numbers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))
from fake_llm_client import FakeLLMClient  # noqa: E402

from udt.common.models import (  # noqa: E402
    Asset,
    AssetType,
    DependencyGraph,
    Incident,
)
from udt.llm.graph import _rank_by_outcome, build_planner_graph, run_planner  # noqa: E402
from udt.twin.simulator import Simulator  # noqa: E402

POINT = {"type": "Point", "coordinates": [72.88, 19.07]}
INCIDENT = Incident(
    incident_id="test",
    type="flood",
    location=POINT,
    onset_tick=0,
    severity=0.8,
    directly_affected_assets=[],
)


def _healthy_hospital_sim() -> Simulator:
    hospital = Asset(
        asset_id="H1",
        asset_type=AssetType.HOSPITAL,
        geometry=POINT,
        attributes={"beds_total": 100, "beds_occupied": 10},
    )
    return Simulator(DependencyGraph(assets=[hospital], edges=[]))


def _already_failed_hospital_sim() -> Simulator:
    hospital = Asset(
        asset_id="H1",
        asset_type=AssetType.HOSPITAL,
        geometry=POINT,
        intrinsic_level=0.1,
        functional_level=0.1,
        attributes={"beds_total": 100, "beds_occupied": 10},
    )
    return Simulator(DependencyGraph(assets=[hospital], edges=[]))


def _no_degradation(tick: int, graph: object) -> dict[str, float]:
    return {}


def _valid_plan_batch_json(plan_ids: list[str], *, diverse: bool = False) -> str:
    """`diverse=True` gives each plan a distinct `priority_assets` set
    (plan index 0 empty, others `["H1"]`) so the batch passes
    `llm/graph.py`'s candidate-diversity check (dev doc §7.3) — tests
    that need >1 plan to actually survive `_generate_plans` (rather than
    exhaust retries into `_fallback_plan`) must pass this."""
    return json.dumps(
        {
            "plans": [
                {
                    "plan_id": pid,
                    "objective": "Test objective.",
                    "goal_weights": {
                        "g_health": 1.0,
                        "g_power": 1.0,
                        "g_transport": 1.0,
                        "g_cost": 1.0,
                    },
                    "priority_assets": ["H1"] if (not diverse or i > 0) else [],
                    "directives": [],
                    "rationale": "Test rationale.",
                    "assumptions": [],
                    "expected_outcome": "Test outcome.",
                }
                for i, pid in enumerate(plan_ids)
            ]
        }
    )


def _valid_action_proposal_json(repair_target: str | None = None) -> str:
    return json.dumps(
        {
            "actions": [
                {
                    "repair_target": repair_target,
                    "ambulance_assignment": None,
                    "patient_transfer": None,
                    "shed_tier": None,
                }
            ],
            "rationale": "Test rationale.",
            "confidence_note": "n/a",
        }
    )


@pytest.mark.phase9
def test_id_path_runs_end_to_end_and_selects_a_proposed_plan() -> None:
    sim = _healthy_hospital_sim()
    client = FakeLLMClient({"generate_plans": [_valid_plan_batch_json(["p1", "p2"], diverse=True)]})
    graph = build_planner_graph()

    output = run_planner(
        graph,
        llm_client=client,
        sim=sim,
        incident=INCIDENT,
        routing_path="ID",
        degradation_fn=_no_degradation,
    )

    assert output.routing_path == "ID"
    assert output.llm_fallback is False
    assert output.selected_plan is not None
    assert output.selected_plan.plan_id in ("p1", "p2")
    assert set(output.simulation_results.keys()) == {"p1", "p2"}
    assert output.action_proposal is None  # ID path never touches this


@pytest.mark.phase9
def test_id_path_falls_back_to_rule_based_plan_when_llm_always_invalid() -> None:
    from udt.llm.graph import MAX_LLM_ATTEMPTS

    sim = _healthy_hospital_sim()
    client = FakeLLMClient({"generate_plans": ["not json"] * MAX_LLM_ATTEMPTS})
    graph = build_planner_graph()

    output = run_planner(
        graph,
        llm_client=client,
        sim=sim,
        incident=INCIDENT,
        routing_path="ID",
        degradation_fn=_no_degradation,
    )

    assert output.llm_fallback is True
    assert output.selected_plan is not None
    assert output.selected_plan.plan_id == "fallback_rule_based"


@pytest.mark.phase9
def test_id_path_falls_back_when_candidates_are_never_diverse() -> None:
    """dev doc §7.3's diversity requirement: candidates with identical
    `priority_assets`/`directives` (differing only in `plan_id`/prose)
    are not genuinely different — `_generate_plans` must retry
    (`MAX_LLM_ATTEMPTS` times) and, if every attempt comes back
    non-diverse, fall back the same way an invalid response does. Every
    scripted batch here is non-diverse on purpose."""
    from udt.llm.graph import MAX_LLM_ATTEMPTS

    sim = _healthy_hospital_sim()
    client = FakeLLMClient(
        {"generate_plans": [_valid_plan_batch_json(["p1", "p2"], diverse=False)] * MAX_LLM_ATTEMPTS}
    )
    graph = build_planner_graph()

    output = run_planner(
        graph,
        llm_client=client,
        sim=sim,
        incident=INCIDENT,
        routing_path="ID",
        degradation_fn=_no_degradation,
    )

    assert output.llm_fallback is True
    assert output.selected_plan is not None
    assert output.selected_plan.plan_id == "fallback_rule_based"
    assert len(client.calls) == MAX_LLM_ATTEMPTS  # every outer attempt used exactly one call


@pytest.mark.phase9
def test_simulate_each_produces_different_outcomes_for_diverse_plans() -> None:
    """The bug this fixes (`MTP_Module_Planner.md`'s 2026-09-22 M9a
    entry): before `plan_interpreter.interpret_plan` was wired in,
    `_simulate_each` simulated the same stand-in action for every
    candidate, so two genuinely different plans always simulated
    *identically*, no matter how different. Two damaged hospitals, where
    the baseline rule would repair the worse one (H1) but a directive
    redirects repair to the other (H2), is a case where the two plans'
    simulated outcomes must differ if the interpreter is actually being
    consulted per-plan."""
    from udt.common.models import Asset, AssetType, DependencyGraph
    from udt.llm.graph import _simulate_each
    from udt.llm.schemas import Directive, GoalWeights, Plan
    from udt.twin.simulator import Simulator

    def _hospital(asset_id: str, level: float) -> Asset:
        return Asset(
            asset_id=asset_id,
            asset_type=AssetType.HOSPITAL,
            geometry=POINT,
            intrinsic_level=level,
            functional_level=level,
            attributes={"beds_total": 100, "beds_occupied": 10},
        )

    h1_worse = _hospital("H1", 0.3)
    h2_better = _hospital("H2", 0.7)
    sim = Simulator(DependencyGraph(assets=[h1_worse, h2_better], edges=[]))
    weights = GoalWeights(g_health=1.0, g_power=1.0, g_transport=1.0, g_cost=1.0)
    plan_baseline = Plan(
        plan_id="baseline",
        objective="x",
        goal_weights=weights,
        priority_assets=[],
        directives=[],
        rationale="x",
        expected_outcome="x",
    )
    plan_redirect = Plan(
        plan_id="redirect",
        objective="x",
        goal_weights=weights,
        priority_assets=[],
        directives=[Directive(kind="protect_repair", details={"asset_id": "H2"})],
        rationale="x",
        expected_outcome="x",
    )
    state = {
        "plans": [plan_baseline, plan_redirect],
        "sim": sim,
        "road_network": None,
        "degradation_fn": None,
        "incident": None,
    }
    result = _simulate_each(state)  # type: ignore[arg-type]
    baseline_result = result["simulation_results"]["baseline"]
    redirect_result = result["simulation_results"]["redirect"]
    # Repairing the worse hospital (baseline) vs. the better one
    # (redirected) changes which hospital's functional_level rises this
    # rollout - the two results must not be identical.
    assert baseline_result.mean_outcome != redirect_result.mean_outcome or (
        baseline_result.rollouts[0].min_hospital_functional_level
        != redirect_result.rollouts[0].min_hospital_functional_level
    )


@pytest.mark.phase9
def test_ood_path_runs_end_to_end_without_refining_when_outcome_is_safe() -> None:
    sim = _healthy_hospital_sim()
    client = FakeLLMClient({"propose_direct_actions": [_valid_action_proposal_json()]})
    graph = build_planner_graph()

    output = run_planner(
        graph,
        llm_client=client,
        sim=sim,
        incident=INCIDENT,
        routing_path="OOD",
        degradation_fn=_no_degradation,
    )

    assert output.routing_path == "OOD"
    assert output.action_proposal is not None
    assert output.constraint_report is not None
    assert output.constraint_report.passed  # no-op action, nothing to violate
    assert "proposal" in output.simulation_results
    assert output.selected_plan is None  # OOD path never touches this
    # Healthy hospital -> P(failure) should be low -> exactly one LLM call.
    assert len(client.calls) == 1


@pytest.mark.phase9
def test_ood_path_refines_exactly_once_when_p_failure_is_high() -> None:
    """dev doc §7.2: "refine_once_if_P(failure)>0.5" — an already-failed
    hospital guarantees P(failure)=1.0 on the first simulate, forcing
    exactly one refine loop, then stopping regardless of the second
    result (dev doc's own "once", not "until safe")."""
    sim = _already_failed_hospital_sim()
    client = FakeLLMClient(
        {
            "propose_direct_actions": [
                _valid_action_proposal_json(),
                _valid_action_proposal_json(repair_target="H1"),
            ]
        }
    )
    graph = build_planner_graph()

    output = run_planner(
        graph,
        llm_client=client,
        sim=sim,
        incident=INCIDENT,
        routing_path="OOD",
        degradation_fn=_no_degradation,
    )

    assert len(client.calls) == 2  # exactly one refinement, not a retry loop
    # The refinement prompt must actually carry the P(failure) feedback.
    assert "P(failure)" in client.calls[1][1]
    assert output.action_proposal is not None
    assert output.action_proposal.actions[0].repair_target == "H1"


@pytest.mark.phase9
def test_ood_path_checks_constraints_before_simulating() -> None:
    """A patient_transfer that violates `bed_capacity` must come back
    repaired in `constraint_report`, and it's the *repaired* action that
    gets simulated - not the raw LLM proposal."""
    hospital_a = Asset(
        asset_id="H1",
        asset_type=AssetType.HOSPITAL,
        geometry=POINT,
        attributes={"beds_total": 100, "beds_occupied": 10},
    )
    hospital_b = Asset(
        asset_id="H2",
        asset_type=AssetType.HOSPITAL,
        geometry=POINT,
        attributes={"beds_total": 50, "beds_occupied": 48},  # only 2 free
    )
    sim = Simulator(DependencyGraph(assets=[hospital_a, hospital_b], edges=[]))

    raw = json.dumps(
        {
            "actions": [
                {
                    "repair_target": None,
                    "ambulance_assignment": None,
                    "patient_transfer": ["H1", "H2", 20],  # exceeds H2's 2 free beds
                    "shed_tier": None,
                }
            ],
            "rationale": "Move patients to H2.",
            "confidence_note": "n/a",
        }
    )
    client = FakeLLMClient({"propose_direct_actions": [raw]})
    graph = build_planner_graph()

    output = run_planner(
        graph,
        llm_client=client,
        sim=sim,
        incident=INCIDENT,
        routing_path="OOD",
        degradation_fn=_no_degradation,
    )

    assert output.constraint_report is not None
    assert not output.constraint_report.passed
    assert output.constraint_report.repaired_action.patient_transfer == ("H1", "H2", 2)


@pytest.mark.phase9
def test_rank_by_outcome_picks_lowest_p_failure_then_lowest_mean_outcome() -> None:
    from udt.common.models import RolloutOutcome, SimulationResult
    from udt.llm.schemas import GoalWeights, Plan

    def _plan(plan_id: str) -> Plan:
        return Plan(
            plan_id=plan_id,
            objective="x",
            goal_weights=GoalWeights(g_health=1.0, g_power=1.0, g_transport=1.0, g_cost=1.0),
            priority_assets=[],
            directives=[],
            rationale="x",
            expected_outcome="x",
        )

    def _result(p_failure: float, mean_outcome: float) -> SimulationResult:
        return SimulationResult(
            rollouts=[
                RolloutOutcome(
                    min_hospital_functional_level=1.0 - p_failure,
                    min_critical_functional_level=1.0 - p_failure,
                    unmet_patient_hours=mean_outcome,
                    new_cascading_failures=0,
                )
            ],
            p_failure=p_failure,
            mean_outcome=mean_outcome,
            std_outcome=0.0,
        )

    state = {
        "plans": [_plan("bad"), _plan("best"), _plan("tied_but_worse_outcome")],
        "simulation_results": {
            "bad": _result(p_failure=0.5, mean_outcome=1.0),
            "best": _result(p_failure=0.1, mean_outcome=10.0),  # lowest p_failure wins outright
            "tied_but_worse_outcome": _result(p_failure=0.1, mean_outcome=20.0),
        },
    }
    result = _rank_by_outcome(state)  # type: ignore[arg-type]
    assert result["selected_plan"].plan_id == "best"
