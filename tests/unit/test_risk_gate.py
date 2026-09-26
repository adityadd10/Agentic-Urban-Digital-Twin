"""Phase 8 acceptance tests: `risk.gate` (dev doc §9.3's autonomy gate),
module M8. Includes the dev doc's own named acceptance criteria: "gate
unit tests incl. timeout-default; OOD => approval always"."""

from __future__ import annotations

from pathlib import Path

import pytest

from udt.common.models import AgentAction, AutonomyTier
from udt.risk.gate import (
    TIMEOUT_TICKS_DEFAULT,
    decide_tier,
    decide_tier_from_config,
    load_autonomy_config,
    safe_hold_action,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
AUTONOMY_CONFIG_PATH = REPO_ROOT / "configs" / "autonomy.yaml"


@pytest.mark.phase8
def test_ood_routing_always_forces_human_approval() -> None:
    """dev doc §9.3: "if routing.path == OOD: tier = HUMAN_APPROVAL
    (unconditional)" — even the best-looking risk/confidence numbers
    can't override it."""
    assert decide_tier(risk=0.0, confidence=1.0, routing_path="OOD") == AutonomyTier.HUMAN_APPROVAL


@pytest.mark.phase8
def test_low_risk_high_confidence_is_autonomous() -> None:
    assert decide_tier(risk=0.1, confidence=0.9, routing_path="ID") == AutonomyTier.AUTONOMOUS


@pytest.mark.phase8
def test_moderate_risk_moderate_confidence_is_execute_and_flag() -> None:
    assert (
        decide_tier(risk=0.3, confidence=0.6, routing_path="ID") == AutonomyTier.EXECUTE_AND_FLAG
    )


@pytest.mark.phase8
def test_high_risk_or_low_confidence_requires_human_approval() -> None:
    assert decide_tier(risk=0.9, confidence=0.9, routing_path="ID") == AutonomyTier.HUMAN_APPROVAL
    assert decide_tier(risk=0.1, confidence=0.1, routing_path="ID") == AutonomyTier.HUMAN_APPROVAL


@pytest.mark.phase8
def test_thresholds_are_strict_inequalities_at_the_exact_boundary() -> None:
    """dev doc §9.3's own notation (`<`, `>`) is strict — exactly at a
    boundary must NOT qualify for the more-autonomous tier."""
    assert decide_tier(risk=0.2, confidence=0.9, routing_path="ID") != AutonomyTier.AUTONOMOUS
    assert decide_tier(risk=0.1, confidence=0.8, routing_path="ID") != AutonomyTier.AUTONOMOUS
    assert (
        decide_tier(risk=0.5, confidence=0.9, routing_path="ID") != AutonomyTier.EXECUTE_AND_FLAG
    )
    assert (
        decide_tier(risk=0.3, confidence=0.5, routing_path="ID") != AutonomyTier.EXECUTE_AND_FLAG
    )


@pytest.mark.phase8
def test_timeout_default_matches_dev_docs_ten_simulated_minutes() -> None:
    """dev doc §9.3: "Timeout (default 10 simulated min)" at dt=5min/tick
    -> 2 ticks."""
    assert TIMEOUT_TICKS_DEFAULT == 2


@pytest.mark.phase8
def test_safe_hold_action_commits_nothing_new() -> None:
    """dev doc §9.3: "no new commitments" — every field must be the
    do-nothing default."""
    action = safe_hold_action()
    assert action == AgentAction()
    assert action.repair_target is None
    assert action.ambulance_assignment is None
    assert action.patient_transfer is None
    assert action.shed_tier is None


@pytest.mark.phase8
def test_load_autonomy_config_matches_decide_tiers_own_defaults() -> None:
    """`configs/autonomy.yaml`'s real thresholds must agree with
    `decide_tier`'s keyword defaults - if someone edits one and forgets
    the other, this test catches the drift."""
    config = load_autonomy_config(AUTONOMY_CONFIG_PATH)
    assert decide_tier_from_config(0.1, 0.9, "ID", config) == decide_tier(0.1, 0.9, "ID")
    assert decide_tier_from_config(0.3, 0.6, "ID", config) == decide_tier(0.3, 0.6, "ID")
    assert decide_tier_from_config(0.9, 0.1, "ID", config) == decide_tier(0.9, 0.1, "ID")
    assert config["approval_timeout_ticks"] == TIMEOUT_TICKS_DEFAULT
