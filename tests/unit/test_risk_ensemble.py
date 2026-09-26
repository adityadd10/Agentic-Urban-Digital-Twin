"""Phase 8 acceptance tests: `risk.ensemble.compute_disagreement` (dev
doc §9.2), module M8."""

from __future__ import annotations

import numpy as np
import pytest

from udt.risk.ensemble import EPSILON, compute_disagreement


@pytest.mark.phase8
def test_disagreement_is_zero_when_all_critics_agree_exactly() -> None:
    assert compute_disagreement([5.0, 5.0, 5.0, 5.0, 5.0]) == pytest.approx(0.0)


@pytest.mark.phase8
def test_disagreement_is_zero_with_a_single_critic() -> None:
    """A lone critic can't disagree with itself — a documented special
    case, not a numerical accident."""
    assert compute_disagreement([3.7]) == 0.0


@pytest.mark.phase8
def test_disagreement_matches_hand_computed_formula() -> None:
    returns = [1.0, 2.0, 3.0, 4.0, 5.0]
    expected = np.std(returns) / (abs(np.mean(returns)) + EPSILON)
    assert compute_disagreement(returns) == pytest.approx(expected)


@pytest.mark.phase8
def test_disagreement_raises_on_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        compute_disagreement([])


@pytest.mark.phase8
def test_disagreement_stays_finite_when_mean_is_near_zero() -> None:
    """dev doc §9.2's own "+eps" term exists exactly for this case —
    critics disagreeing about the *sign* of a near-zero-mean return
    shouldn't blow up to infinity."""
    result = compute_disagreement([-1.0, 1.0, -1.0, 1.0, 0.0])
    assert result == result  # not NaN
    assert result < float("inf")
