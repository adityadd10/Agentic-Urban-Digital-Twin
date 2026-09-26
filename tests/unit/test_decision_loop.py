"""Phase 10 acceptance tests: `experiments/decision_loop.py` (dev doc
§10/§11's Experiment G, module M10a — "Experiment G config running
end-to-end without UI"), the full headless decision loop.

Hand-built graphs (same convention as `test_llm_graph_flow.py`), not
real Kurla data — this loop's own wiring/branching is what's under
test."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT / "experiments", REPO_ROOT / "tests" / "fixtures"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fake_llm_client import FakeLLMClient  # noqa: E402

from decision_loop import RuleBasedPolicyExecutor, run_decision_loop_episode  # noqa: E402
from udt.common.models import (  # noqa: E402
    AgentAction,
    Asset,
    AssetType,
    AutonomyTier,
    DependencyGraph,
    Incident,
    PolicyRegistryEntry,
    RiskCalibration,
)
from udt.logging.decision_log import load_decision_log  # noqa: E402
from udt.risk.human_model import HumanModel  # noqa: E402
from udt.routing.ood import featurize_incident  # noqa: E402
from udt.twin.simulator import Simulator  # noqa: E402

POINT = {"type": "Point", "coordinates": [72.88, 19.07]}
ZERO_CALIBRATION = RiskCalibration(
    p95_risk_raw=0.0, p95_disagreement_raw=0.0, conformal_threshold=0.0, p95_interval_width=0.0
)
REGISTRY_ENTRY = PolicyRegistryEntry(
    incident_type="flood",
    policy_path="run/model.pt",
    env_version="udt_multi_env_v0",
    suite_version="none-n1-scenario",
    val_score=-1.0,
    trained_date="2026-01-01T00:00:00+00:00",
)


def _incident(severity: float = 0.7) -> Incident:
    return Incident(
        incident_id="test_inc",
        type="flood",
        location=POINT,
        onset_tick=0,
        severity=severity,
        directly_affected_assets=[],
    )


def _healthy_sim_and_graph() -> tuple[Simulator, DependencyGraph]:
    hospital = Asset(
        asset_id="H1",
        asset_type=AssetType.HOSPITAL,
        geometry=POINT,
        attributes={"beds_total": 100, "beds_occupied": 10},
    )
    dep_graph = DependencyGraph(assets=[hospital], edges=[])
    return Simulator(dep_graph.model_copy(deep=True)), dep_graph


def _already_failed_sim_and_graph() -> tuple[Simulator, DependencyGraph]:
    hospital = Asset(
        asset_id="H1",
        asset_type=AssetType.HOSPITAL,
        geometry=POINT,
        intrinsic_level=0.1,
        functional_level=0.1,
        attributes={"beds_total": 100, "beds_occupied": 10},
    )
    dep_graph = DependencyGraph(assets=[hospital], edges=[])
    return Simulator(dep_graph.model_copy(deep=True)), dep_graph


def _no_degradation(tick: int, graph: object) -> dict[str, float]:
    return {}


def _in_distribution_router_calibration(incident: Incident, dep_graph: DependencyGraph) -> object:
    from udt.common.models import RouterCalibration

    x = featurize_incident(incident, dep_graph).tolist()
    return RouterCalibration(
        incident_type="flood",
        env_version="udt_multi_env_v0",
        training_features=[x] * 10,
        calibration_scores=[0.0] * 10,
        k=5,
    )


def _plan_batch_json(plan_ids: list[str], *, diverse: bool = False) -> str:
    """`diverse=True` gives plan index 0 empty `priority_assets` and
    every other plan `["H1"]`, so a >1-plan batch passes `llm/graph.py`'s
    candidate-diversity check (dev doc §7.3) instead of exhausting
    retries into `_fallback_plan`."""
    return json.dumps(
        {
            "plans": [
                {
                    "plan_id": pid,
                    "objective": "x",
                    "goal_weights": {
                        "g_health": 1.0,
                        "g_power": 1.0,
                        "g_transport": 1.0,
                        "g_cost": 1.0,
                    },
                    "priority_assets": ["H1"] if (not diverse or i > 0) else [],
                    "directives": [],
                    "rationale": "x",
                    "assumptions": [],
                    "expected_outcome": "x",
                }
                for i, pid in enumerate(plan_ids)
            ]
        }
    )


def _action_proposal_json() -> str:
    return json.dumps(
        {
            "actions": [
                {
                    "repair_target": None,
                    "ambulance_assignment": None,
                    "patient_transfer": None,
                    "shed_tier": None,
                }
            ],
            "rationale": "x",
            "confidence_note": "n/a",
        }
    )


@pytest.mark.phase10
def test_id_path_runs_to_completion_and_logs_every_decision(tmp_path: Path) -> None:
    sim, dep_graph = _healthy_sim_and_graph()
    incident = _incident()
    # Only ONE scripted response: with `REPLAN_INTERVAL_TICKS=24` and
    # this test's 9 ticks (3 decision ticks), the planner should only
    # actually be re-invoked once (tick 0, dev doc §7.1's "at incident
    # onset") - the other two decision ticks reuse the persisted plan,
    # deciding a fresh tactical action against it but never calling the
    # LLM again. A second scripted response existing would silently mask
    # a regression back to "replans every decision tick".
    client = FakeLLMClient({"generate_plans": [_plan_batch_json(["p1", "p2"], diverse=True)]})
    log_path = tmp_path / "decisions.jsonl"

    result = run_decision_loop_episode(
        sim,
        incident,
        dep_graph,
        degradation_fn=_no_degradation,
        road_network=None,
        ward_polygon=None,
        llm_client=client,
        policy_executor=RuleBasedPolicyExecutor(),
        env_version="udt_multi_env_v0",
        registry_entries=[REGISTRY_ENTRY],
        router_calibration=_in_distribution_router_calibration(incident, dep_graph),
        risk_calibration=ZERO_CALIBRATION,
        critic_returns=[1.0, 1.0, 1.0, 1.0, 1.0],
        human_model=HumanModel(),
        n_ticks=9,
        episode_id="test_ep",
        decision_log_path=log_path,
    )

    assert len(result.trace) == 9
    assert len(result.decisions) == 3  # ticks 0, 3, 6 (9 ticks / 3-tick interval)
    for record in result.decisions:
        assert record.routing.path == "ID"
        assert record.episode_id == "test_ep"
    # Same selected plan governs all three tactical decisions (no
    # replan trigger fires inside 9 ticks) - the whole point of
    # `REPLAN_INTERVAL_TICKS`/persistence existing.
    assert len({record.selected_plan_id for record in result.decisions}) == 1
    assert sum(1 for node, _ in client.calls if node == "generate_plans") == 1

    logged = load_decision_log(log_path)
    assert len(logged) == 3
    assert logged == result.decisions


@pytest.mark.phase10
def test_id_path_low_risk_is_autonomous_no_approval_needed(tmp_path: Path) -> None:
    sim, dep_graph = _healthy_sim_and_graph()
    incident = _incident()
    client = FakeLLMClient({"generate_plans": [_plan_batch_json(["p1"])]})

    result = run_decision_loop_episode(
        sim,
        incident,
        dep_graph,
        degradation_fn=_no_degradation,
        road_network=None,
        ward_polygon=None,
        llm_client=client,
        policy_executor=RuleBasedPolicyExecutor(),
        env_version="udt_multi_env_v0",
        registry_entries=[REGISTRY_ENTRY],
        router_calibration=_in_distribution_router_calibration(incident, dep_graph),
        risk_calibration=ZERO_CALIBRATION,
        critic_returns=[1.0] * 5,
        human_model=HumanModel(),
        n_ticks=1,
        episode_id="test_ep",
        decision_log_path=tmp_path / "decisions.jsonl",
    )
    record = result.decisions[0]
    assert record.risk.tier == AutonomyTier.AUTONOMOUS
    assert record.approval.required is False
    assert record.approval.response is None


@pytest.mark.phase10
def test_ood_path_always_requires_approval_regardless_of_risk(tmp_path: Path) -> None:
    """dev doc §9.3: OOD forces HUMAN_APPROVAL unconditionally - even
    when the underlying risk is objectively low."""
    sim, dep_graph = _healthy_sim_and_graph()
    incident = _incident()
    client = FakeLLMClient({"propose_direct_actions": [_action_proposal_json()]})

    result = run_decision_loop_episode(
        sim,
        incident,
        dep_graph,
        degradation_fn=_no_degradation,
        road_network=None,
        ward_polygon=None,
        llm_client=client,
        policy_executor=RuleBasedPolicyExecutor(),
        env_version="udt_multi_env_v0",
        registry_entries=[],  # empty registry -> registry_hit=False -> forces OOD
        router_calibration=_in_distribution_router_calibration(incident, dep_graph),
        risk_calibration=ZERO_CALIBRATION,
        critic_returns=[1.0] * 5,
        human_model=HumanModel(),
        n_ticks=1,
        episode_id="test_ep",
        decision_log_path=tmp_path / "decisions.jsonl",
    )
    record = result.decisions[0]
    assert record.routing.path == "OOD"
    assert record.risk.tier == AutonomyTier.HUMAN_APPROVAL
    assert record.approval.required is True
    # Low P(failure) on a healthy hospital -> the headless HumanModel approves.
    assert record.approval.response == "approve"


@pytest.mark.phase10
def test_ood_path_rejection_executes_the_safe_hold_action(tmp_path: Path) -> None:
    """A high-P(failure) situation must be rejected by the headless
    HumanModel and fall back to the safe hold action (no commitments),
    not whatever the LLM proposed."""
    sim, dep_graph = _already_failed_sim_and_graph()
    incident = _incident()
    repair_proposal = json.dumps(
        {
            "actions": [
                {
                    "repair_target": "H1",
                    "ambulance_assignment": None,
                    "patient_transfer": None,
                    "shed_tier": None,
                }
            ],
            "rationale": "x",
            "confidence_note": "n/a",
        }
    )
    # This hospital is already below the failure threshold, so
    # `llm/graph.py`'s own refine-once loop fires regardless of what's
    # proposed - two scripted responses needed, not one.
    client = FakeLLMClient({"propose_direct_actions": [repair_proposal, repair_proposal]})

    result = run_decision_loop_episode(
        sim,
        incident,
        dep_graph,
        degradation_fn=_no_degradation,
        road_network=None,
        ward_polygon=None,
        llm_client=client,
        policy_executor=RuleBasedPolicyExecutor(),
        env_version="udt_multi_env_v0",
        registry_entries=[],
        router_calibration=_in_distribution_router_calibration(incident, dep_graph),
        risk_calibration=ZERO_CALIBRATION,
        critic_returns=[1.0] * 5,
        human_model=HumanModel(),  # default approve_if_pfail_lt=0.5
        n_ticks=1,
        episode_id="test_ep",
        decision_log_path=tmp_path / "decisions.jsonl",
    )
    record = result.decisions[0]
    assert record.risk.p_failure == 1.0  # H1 starts already below the 0.3 failure threshold
    assert record.approval.response == "reject"
    assert record.executed_action == AgentAction()  # safe hold - no commitments, not the repair


@pytest.mark.phase10
def test_llm_fallback_is_recorded_in_the_decision_log(tmp_path: Path) -> None:
    sim, dep_graph = _healthy_sim_and_graph()
    incident = _incident()
    client = FakeLLMClient({"generate_plans": ["not valid json"] * 3})

    result = run_decision_loop_episode(
        sim,
        incident,
        dep_graph,
        degradation_fn=_no_degradation,
        road_network=None,
        ward_polygon=None,
        llm_client=client,
        policy_executor=RuleBasedPolicyExecutor(),
        env_version="udt_multi_env_v0",
        registry_entries=[REGISTRY_ENTRY],
        router_calibration=_in_distribution_router_calibration(incident, dep_graph),
        risk_calibration=ZERO_CALIBRATION,
        critic_returns=[1.0] * 5,
        human_model=HumanModel(),
        n_ticks=1,
        episode_id="test_ep",
        decision_log_path=tmp_path / "decisions.jsonl",
    )
    assert result.decisions[0].llm.fallback is True
