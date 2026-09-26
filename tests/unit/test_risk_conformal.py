"""Phase 8 acceptance tests: `risk.conformal` (dev doc §9.2's split-
conformal calibration), module M8.

Includes the exact acceptance criterion the dev doc's own Phase 8 row
names: "conformal coverage on val within [85%, 95%] for the 90%
interval" — checked here against synthetic data with a known
distribution, not real val episodes (which don't exist yet, see
`risk/engine.py`'s module docstring).
"""

from __future__ import annotations

import numpy as np
import pytest

from udt.risk.conformal import calibrate, interval_width


@pytest.mark.phase8
def test_calibrate_matches_hand_computed_order_statistic() -> None:
    """n=9 scores, coverage=0.9 -> ceil(10*0.9)=9th of 9 -> the max."""
    scores = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    assert calibrate(scores, coverage=0.9) == 9.0


@pytest.mark.phase8
def test_calibrate_uses_the_correct_rank_not_a_plain_percentile() -> None:
    """The split-conformal rank (ceil((n+1) x coverage)) is deliberately
    NOT the same as numpy's own 90th-percentile of the same data — this
    test would pass with a naive `np.percentile(scores, 90)`
    implementation by coincidence on some inputs, so it's chosen to
    disagree with that on purpose."""
    scores = list(range(1, 11))  # 1..10, n=10
    # ceil(11 * 0.9) = 10 -> the 10th (largest) of 10.
    assert calibrate(scores, coverage=0.9) == 10.0
    # A plain 90th percentile of 1..10 (linear interpolation) is 9.1,
    # not 10 - confirms this isn't accidentally equivalent.
    assert np.percentile(scores, 90) != calibrate(scores, coverage=0.9)


@pytest.mark.phase8
def test_calibrate_caps_rank_at_n_when_coverage_asks_for_more() -> None:
    scores = [1.0, 2.0, 3.0]
    assert calibrate(scores, coverage=1.0) == 3.0  # capped at the largest, not IndexError


@pytest.mark.phase8
def test_calibrate_raises_on_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        calibrate([])


@pytest.mark.phase8
def test_interval_width_is_twice_the_threshold() -> None:
    assert interval_width(3.5) == pytest.approx(7.0)


@pytest.mark.phase8
def test_conformal_coverage_lands_in_85_to_95_percent_on_synthetic_data() -> None:
    """dev doc's own Phase 8 acceptance criterion, verbatim: calibrate
    on one batch of nonconformity scores drawn from a known
    distribution, then check what fraction of a *fresh* batch from the
    same distribution falls within the calibrated threshold — should
    land close to the nominal 90% coverage, specifically within the
    dev doc's own [85%, 95%] tolerance band."""
    rng = np.random.default_rng(0)
    calibration_scores = np.abs(rng.normal(loc=0.0, scale=1.0, size=2000)).tolist()
    threshold = calibrate(calibration_scores, coverage=0.9)

    test_scores = np.abs(rng.normal(loc=0.0, scale=1.0, size=5000))
    empirical_coverage = float(np.mean(test_scores <= threshold))

    assert 0.85 <= empirical_coverage <= 0.95
