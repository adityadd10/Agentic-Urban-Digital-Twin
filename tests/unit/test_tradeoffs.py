"""Physical trade-off features for the twin-v3 suite (`udt.scenarios.tradeoffs`)."""

from __future__ import annotations

import pytest

from udt.scenarios.tradeoffs import destination_tradeoff, severity_band

CHAINS = {"H1": {"H1", "S2", "W0", "S0"}, "H2": {"H2", "S0", "W1"}, "H3": {"H3", "S1", "W2"}}
DRY = {
    ("H1", "H2"): 13.0,
    ("H1", "H3"): 14.0,
    ("H2", "H1"): 13.0,
    ("H2", "H3"): 15.0,
    ("H3", "H1"): 14.0,
    ("H3", "H2"): 15.0,
}


def _p(**threatened: float) -> dict[str, float]:
    base = dict.fromkeys(["H1", "H2", "H3", "S0", "S1", "S2", "W0", "W1", "W2"], 0.0)
    base.update(threatened)
    return base


@pytest.mark.phase3
def test_near_threatened_far_safe_is_a_destination_tradeoff() -> None:
    # S0 flooded threatens H1 (via its pump W0) and H2 (power); H3 stays safe.
    assert destination_tradeoff(CHAINS, DRY, _p(S0=0.9)) is True


@pytest.mark.phase3
def test_near_safe_alternative_is_not_a_tradeoff() -> None:
    # only H1 threatened (its own substation S2): nearest alternative H2 is fine.
    assert destination_tradeoff(CHAINS, DRY, _p(S2=0.9)) is False


@pytest.mark.phase3
def test_all_threatened_or_none_is_not_a_tradeoff() -> None:
    assert destination_tradeoff(CHAINS, DRY, _p()) is False
    assert destination_tradeoff(CHAINS, DRY, _p(S0=0.9, S1=0.9, S2=0.9)) is False


@pytest.mark.phase3
def test_severity_bands_are_equal_thirds() -> None:
    assert severity_band(0.2) == "mild" and severity_band(0.46) == "mild"
    assert severity_band(0.5) == "moderate" and severity_band(0.9) == "severe"
    assert severity_band(1.0) == "severe"
