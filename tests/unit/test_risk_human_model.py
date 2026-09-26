"""Phase 8 acceptance tests: `risk.human_model.HumanModel` (dev doc
§9.3's scripted approve/reject stand-in), module M8."""

from __future__ import annotations

import pytest

from udt.risk.human_model import DEFAULT_APPROVE_IF_PFAIL_LT, HumanModel


@pytest.mark.phase8
def test_default_matches_dev_doc_exactly() -> None:
    assert DEFAULT_APPROVE_IF_PFAIL_LT == 0.5


@pytest.mark.phase8
def test_approves_when_p_failure_is_below_threshold() -> None:
    assert HumanModel().decide(p_failure=0.3) is True


@pytest.mark.phase8
def test_rejects_when_p_failure_is_above_threshold() -> None:
    assert HumanModel().decide(p_failure=0.7) is False


@pytest.mark.phase8
def test_rejects_exactly_at_the_threshold() -> None:
    """Strict less-than (dev doc: "approve-if-P_failure<0.5") — exactly
    0.5 must NOT approve."""
    assert HumanModel().decide(p_failure=0.5) is False


@pytest.mark.phase8
def test_threshold_is_configurable() -> None:
    """dev doc §9.3: "its behavior is config" —
    `configs/experiments/*.yaml`'s `human_model.approve_if_pfail_lt`
    feeds this directly."""
    lenient = HumanModel(approve_if_pfail_lt=0.9)
    assert lenient.decide(p_failure=0.7) is True
