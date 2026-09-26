"""Phase 8 acceptance tests: `risk.engine` (dev doc §9.1/§9.2's risk +
confidence formulas), module M8."""

from __future__ import annotations

import pytest

from udt.common.models import RiskCalibration, RolloutOutcome, SimulationResult
from udt.risk.engine import (
    CASCADE_WEIGHT,
    assess,
    calibrate_from_samples,
    compute_confidence,
    compute_risk,
    normalize_to_p95,
)
from udt.risk.ensemble import compute_disagreement

ZERO_CALIBRATION = RiskCalibration(
    p95_risk_raw=0.0, p95_disagreement_raw=0.0, conformal_threshold=0.0, p95_interval_width=0.0
)


def _rollout(min_hospital: float, unmet: float, cascades: int) -> RolloutOutcome:
    return RolloutOutcome(
        min_hospital_functional_level=min_hospital,
        min_critical_functional_level=min_hospital,
        unmet_patient_hours=unmet,
        new_cascading_failures=cascades,
    )


@pytest.mark.phase8
def test_normalize_to_p95_clips_at_one() -> None:
    assert normalize_to_p95(raw=10.0, p95_reference=5.0) == 1.0


@pytest.mark.phase8
def test_normalize_to_p95_scales_linearly_below_the_reference() -> None:
    assert normalize_to_p95(raw=2.5, p95_reference=5.0) == pytest.approx(0.5)


@pytest.mark.phase8
def test_normalize_to_p95_is_zero_with_no_reference_data() -> None:
    assert normalize_to_p95(raw=100.0, p95_reference=0.0) == 0.0


@pytest.mark.phase8
def test_compute_risk_is_zero_when_no_rollout_fails() -> None:
    result = SimulationResult(
        rollouts=[_rollout(0.9, 5.0, 0), _rollout(0.8, 3.0, 0)],
        p_failure=0.0,
        mean_outcome=4.0,
        std_outcome=1.0,
    )
    p_failure, consequence, risk_raw, risk = compute_risk(result, ZERO_CALIBRATION)
    assert p_failure == 0.0
    assert consequence == 0.0
    assert risk_raw == 0.0
    assert risk == 0.0


@pytest.mark.phase8
def test_compute_risk_matches_hand_computed_formula() -> None:
    """dev doc §9.1 exactly: Consequence = mean over FAILING rollouts of
    (unmet demand + 5*cascades); Risk = P_failure x Consequence. Two
    failing rollouts, one healthy one mixed in to confirm the healthy
    one is excluded from the Consequence mean."""
    rollouts = [
        _rollout(0.1, unmet=10.0, cascades=1),  # failing: 10 + 5*1 = 15
        _rollout(0.2, unmet=20.0, cascades=2),  # failing: 20 + 5*2 = 30
        _rollout(0.9, unmet=100.0, cascades=5),  # healthy - excluded
    ]
    result = SimulationResult(
        rollouts=rollouts, p_failure=2 / 3, mean_outcome=43.3, std_outcome=1.0
    )
    p_failure, consequence, risk_raw, risk = compute_risk(result, ZERO_CALIBRATION)

    expected_consequence = ((10.0 + CASCADE_WEIGHT * 1) + (20.0 + CASCADE_WEIGHT * 2)) / 2
    assert p_failure == pytest.approx(2 / 3)
    assert consequence == pytest.approx(expected_consequence)
    assert risk_raw == pytest.approx((2 / 3) * expected_consequence)
    assert risk == 0.0  # ZERO_CALIBRATION's p95_risk_raw=0 -> normalize_to_p95 returns 0.0


@pytest.mark.phase8
def test_compute_risk_normalizes_against_the_calibrated_p95() -> None:
    rollouts = [_rollout(0.1, unmet=10.0, cascades=0)]
    result = SimulationResult(rollouts=rollouts, p_failure=1.0, mean_outcome=10.0, std_outcome=0.0)
    calibration = RiskCalibration(
        p95_risk_raw=5.0, p95_disagreement_raw=1.0, conformal_threshold=1.0, p95_interval_width=1.0
    )
    _, _, risk_raw, risk = compute_risk(result, calibration)
    assert risk_raw == pytest.approx(10.0)  # 1.0 x 10.0
    assert risk == 1.0  # clipped: 10.0 / 5.0 > 1.0


@pytest.mark.phase8
def test_compute_confidence_matches_hand_computed_formula() -> None:
    """dev doc §9.2: confidence = 1 - max(disagreement_norm,
    interval_width_norm)."""
    calibration = RiskCalibration(
        p95_risk_raw=1.0, p95_disagreement_raw=2.0, conformal_threshold=3.0, p95_interval_width=10.0
    )
    critic_returns = [1.0, 2.0, 3.0, 4.0, 5.0]
    disagreement_raw = compute_disagreement(critic_returns)
    disagreement_norm = min(1.0, disagreement_raw / 2.0)
    width_norm = min(1.0, (2.0 * 3.0) / 10.0)  # interval_width(3.0) = 6.0
    expected = 1.0 - max(disagreement_norm, width_norm)

    assert compute_confidence(critic_returns, calibration) == pytest.approx(expected)


@pytest.mark.phase8
def test_assess_bundles_risk_and_confidence_into_one_report() -> None:
    result = SimulationResult(
        rollouts=[_rollout(0.1, 10.0, 1)], p_failure=1.0, mean_outcome=10.0, std_outcome=0.0
    )
    calibration = RiskCalibration(
        p95_risk_raw=15.0, p95_disagreement_raw=1.0, conformal_threshold=0.5, p95_interval_width=2.0
    )
    report = assess(result, [1.0, 1.0, 1.0, 1.0, 1.0], calibration)
    assert report.p_failure == 1.0
    assert report.consequence == pytest.approx(15.0)  # 10 + 5*1
    assert 0.0 <= report.confidence <= 1.0
    assert 0.0 <= report.risk <= 1.0


@pytest.mark.phase8
def test_calibrate_from_samples_produces_real_p95_values() -> None:
    risk_samples = list(range(1, 101))  # 1..100
    disagreement_samples = [x / 10.0 for x in range(1, 101)]
    nonconformity = [abs(x) for x in range(-50, 50)]

    calibration = calibrate_from_samples(risk_samples, disagreement_samples, nonconformity)

    assert calibration.p95_risk_raw == pytest.approx(95.05, abs=1.0)
    assert calibration.p95_disagreement_raw == pytest.approx(9.505, abs=0.1)
    assert calibration.conformal_threshold > 0.0
    assert calibration.p95_interval_width > 0.0


@pytest.mark.phase8
def test_calibrate_from_samples_handles_empty_input_gracefully() -> None:
    calibration = calibrate_from_samples([], [], [])
    assert calibration.p95_risk_raw == 0.0
    assert calibration.p95_disagreement_raw == 0.0
    assert calibration.conformal_threshold == 0.0
    assert calibration.p95_interval_width == 0.0


@pytest.mark.phase8
def test_calibrate_from_samples_is_deterministic_given_the_same_seed() -> None:
    nonconformity = [0.1, 0.5, 0.3, 0.9, 0.2, 0.7, 0.4, 0.6, 0.8, 1.0]
    a = calibrate_from_samples([1.0], [1.0], nonconformity, seed=42)
    b = calibrate_from_samples([1.0], [1.0], nonconformity, seed=42)
    assert a == b
