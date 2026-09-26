"""Phase 9 acceptance tests: `llm.schemas` (dev doc §7.3's LLM->MARL/
LLM->twin contract), module M9a."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from udt.common.models import AgentAction
from udt.llm.schemas import ActionProposal, Directive, GoalWeights, Plan


def _goal_weights(**overrides: float) -> dict[str, float]:
    base = {"g_health": 1.0, "g_power": 1.0, "g_transport": 1.0, "g_cost": 1.0}
    base.update(overrides)
    return base


@pytest.mark.phase9
def test_goal_weights_accepts_the_dev_docs_own_bounds() -> None:
    GoalWeights(**_goal_weights(g_health=0.5))
    GoalWeights(**_goal_weights(g_health=2.0))


@pytest.mark.phase9
def test_goal_weights_rejects_out_of_bounds() -> None:
    with pytest.raises(ValidationError):
        GoalWeights(**_goal_weights(g_health=0.49))
    with pytest.raises(ValidationError):
        GoalWeights(**_goal_weights(g_power=2.01))


@pytest.mark.phase9
def test_directive_matches_dev_docs_own_example_shape() -> None:
    d = Directive(kind="prepare_transfer", details={"from": "H2", "to": "H1"})
    assert d.kind == "prepare_transfer"
    assert d.details == {"from": "H2", "to": "H1"}


@pytest.mark.phase9
def test_plan_round_trips_through_json() -> None:
    plan = Plan(
        plan_id="p1",
        objective="Protect H1's power supply.",
        goal_weights=GoalWeights(**_goal_weights(g_power=1.8)),
        priority_assets=["H1", "S1"],
        directives=[Directive(kind="prepare_transfer", details={"from": "H2", "to": "H1"})],
        rationale="S1 is at risk of overload.",
        assumptions=["Flood severity stays constant."],
        expected_outcome="H1 stays above 80% functional.",
    )
    restored = Plan.model_validate_json(plan.model_dump_json())
    assert restored == plan


@pytest.mark.phase9
def test_plan_rejects_more_than_five_priority_assets() -> None:
    with pytest.raises(ValidationError):
        Plan(
            plan_id="p1",
            objective="x",
            goal_weights=GoalWeights(**_goal_weights()),
            priority_assets=["A", "B", "C", "D", "E", "F"],
            directives=[],
            rationale="x",
            expected_outcome="x",
        )


@pytest.mark.phase9
def test_plan_rejects_more_than_six_directives() -> None:
    with pytest.raises(ValidationError):
        Plan(
            plan_id="p1",
            objective="x",
            goal_weights=GoalWeights(**_goal_weights()),
            priority_assets=[],
            directives=[Directive(kind="k") for _ in range(7)],
            rationale="x",
            expected_outcome="x",
        )


@pytest.mark.phase9
def test_action_proposal_uses_the_real_agent_action_vocabulary() -> None:
    proposal = ActionProposal(
        actions=[AgentAction(repair_target="S1"), AgentAction(shed_tier={"S1": 2})],
        rationale="Repair S1 and shed load while it's under repair.",
        confidence_note="moderately confident",
    )
    assert len(proposal.actions) == 2
    assert proposal.actions[0].repair_target == "S1"
    # dev doc §7.3: confidence_note is free text, never coerced/parsed.
    assert isinstance(proposal.confidence_note, str)
