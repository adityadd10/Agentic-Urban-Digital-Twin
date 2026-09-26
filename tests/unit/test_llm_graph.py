"""Phase 9 acceptance tests: `llm.graph`'s pure-logic pieces (dev doc
§7.1's validation-retry-then-fallback rule, and the action-merge/prompt-
building helpers), module M9a — exercised directly, without running the
full LangGraph flow (see `tests/integration/test_llm_planner_flow.py`
for that)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))
from fake_llm_client import FakeLLMClient  # noqa: E402

from udt.common.models import AgentAction, Incident  # noqa: E402
from udt.llm.graph import (  # noqa: E402
    MAX_LLM_ATTEMPTS,
    REFINE_P_FAILURE_THRESHOLD,
    _build_generate_plans_prompt,
    _build_propose_actions_prompt,
    _fallback_action_proposal,
    _fallback_plan,
    _generate_structured,
    _merge_actions,
)
from udt.llm.schemas import GoalWeights  # noqa: E402

INCIDENT = Incident(
    incident_id="test",
    type="flood",
    location={"type": "Point", "coordinates": [72.88, 19.07]},
    onset_tick=0,
    severity=0.8,
    directly_affected_assets=[],
)


class _Echo(BaseModel):
    value: int


@pytest.mark.phase9
def test_generate_structured_parses_a_valid_first_response() -> None:
    client = FakeLLMClient({"n": ['{"value": 42}']})
    parsed, fell_back = _generate_structured(client, "n", "prompt", _Echo)
    assert not fell_back
    assert parsed == _Echo(value=42)
    assert len(client.calls) == 1


@pytest.mark.phase9
def test_generate_structured_retries_on_invalid_json_then_succeeds() -> None:
    """dev doc §7.1: "on validation failure retry with the error message
    appended"."""
    client = FakeLLMClient({"n": ["not json at all", '{"value": 7}']})
    parsed, fell_back = _generate_structured(client, "n", "prompt", _Echo)
    assert not fell_back
    assert parsed == _Echo(value=7)
    assert len(client.calls) == 2
    # The retry prompt must actually include the previous error, per
    # dev doc §7.1's own wording - not just a bare repeat of the prompt.
    assert "invalid" in client.calls[1][1].lower()


@pytest.mark.phase9
def test_generate_structured_retries_on_schema_validation_failure() -> None:
    client = FakeLLMClient({"n": ['{"value": "not an int... actually invalid"}', '{"value": 1}']})
    parsed, fell_back = _generate_structured(client, "n", "prompt", _Echo)
    assert not fell_back
    assert parsed == _Echo(value=1)


@pytest.mark.phase9
def test_generate_structured_falls_back_after_max_attempts() -> None:
    """dev doc §7.1: "max 3 attempts, then fall back [...] and log
    llm_fallback=true"."""
    client = FakeLLMClient({"n": ["bad"] * MAX_LLM_ATTEMPTS})
    parsed, fell_back = _generate_structured(client, "n", "prompt", _Echo)
    assert parsed is None
    assert fell_back is True
    assert len(client.calls) == MAX_LLM_ATTEMPTS


@pytest.mark.phase9
def test_fallback_plan_has_neutral_goal_weights() -> None:
    """dev doc §7.1: "fall back to the rule-based baseline plan" - a
    `Plan`'s only real effect is its goal_weights (directives are
    advisory only), so "rule-based baseline" means all-1.0 weights, the
    same default M6a's own unconditioned policy used."""
    plan = _fallback_plan()
    assert plan.goal_weights == GoalWeights(g_health=1.0, g_power=1.0, g_transport=1.0, g_cost=1.0)
    assert plan.directives == []


@pytest.mark.phase9
def test_fallback_action_proposal_commits_nothing() -> None:
    proposal = _fallback_action_proposal()
    assert proposal.actions == [AgentAction()]


@pytest.mark.phase9
def test_merge_actions_takes_first_non_none_per_field() -> None:
    merged = _merge_actions(
        [
            AgentAction(repair_target="S1"),
            AgentAction(repair_target="S2", shed_tier={"S1": 2}),
        ]
    )
    assert merged.repair_target == "S1"  # first one wins
    assert merged.shed_tier == {"S1": 2}  # only the second action had one


@pytest.mark.phase9
def test_merge_actions_of_empty_list_is_a_no_op() -> None:
    assert _merge_actions([]) == AgentAction()


@pytest.mark.phase9
def test_build_generate_plans_prompt_interpolates_real_incident_data() -> None:
    prompt = _build_generate_plans_prompt("SUMMARY_TEXT", INCIDENT, n_plans=2)
    assert "SUMMARY_TEXT" in prompt
    assert "flood" in prompt
    assert "0.80" in prompt


@pytest.mark.phase9
def test_build_propose_actions_prompt_includes_feedback_only_on_refinement() -> None:
    from udt.common.models import RolloutOutcome, SimulationResult

    fresh_prompt = _build_propose_actions_prompt("SUMMARY", INCIDENT, prior_result=None)
    assert "previous proposal" not in fresh_prompt

    prior = SimulationResult(
        rollouts=[
            RolloutOutcome(
                min_hospital_functional_level=0.1,
                min_critical_functional_level=0.1,
                unmet_patient_hours=5.0,
                new_cascading_failures=0,
            )
        ],
        p_failure=0.9,
        mean_outcome=5.0,
        std_outcome=0.0,
    )
    refine_prompt = _build_propose_actions_prompt("SUMMARY", INCIDENT, prior_result=prior)
    assert "previous proposal" in refine_prompt
    assert f"{REFINE_P_FAILURE_THRESHOLD}" in refine_prompt
    assert "0.90" in refine_prompt
