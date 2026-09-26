"""Risk engine (dev doc §9.1/§9.2, module M8).

Ties `twin/counterfactual.py`'s `simulate()` output (§9.1's P_failure/
Consequence ingredients) together with `risk/ensemble.py` (§9.2's
ensemble disagreement) and `risk/conformal.py` (§9.2's conformal
interval) into one `RiskAssessment` per decision — `risk/gate.py` then
turns that into an autonomy tier.

**Two disclosed prerequisite gaps, both real and both already flagged
elsewhere this session, not new here:**
1. §9.2's "K=5 critics from the 5 training seeds" needs 5 separately
   trained MAPPO runs, which don't exist (no training has run at the
   real budget yet). `compute_confidence` is generic over however many
   critic returns a caller supplies.
2. §9.1's "95th percentile on val scenarios" and §9.2's "val episodes"
   both need a frozen train/val/test scenario suite (M3's own
   deferral). `calibrate_from_samples` below computes a real
   `RiskCalibration` from whatever samples a caller supplies — real
   held-out scenarios once M3's suite exists, or (this module's own
   tests, and `scripts/calibrate_risk.py`) a handful of freshly
   generated ones today — flagged the same way M6b's `val_score_is_
   true_holdout=False` already is, not silently implied as more rigorous
   than it is.
"""

from __future__ import annotations

import numpy as np

from udt.common.models import RiskAssessment, RiskCalibration, SimulationResult
from udt.risk.conformal import calibrate, interval_width
from udt.risk.ensemble import compute_disagreement

# dev doc §9.1 exactly: "criticality-weighted unmet demand + 5·cascades".
CASCADE_WEIGHT = 5.0

# dev doc §9.2 default coverage: "90% prediction interval".
DEFAULT_COVERAGE = 0.9

# Bootstrap resamples for estimating a *reference scale* for interval
# width (see `calibrate_from_samples`'s docstring — a single calibration
# pass only produces one threshold, not a distribution to percentile).
DEFAULT_N_BOOTSTRAP = 200


def normalize_to_p95(raw: float, p95_reference: float) -> float:
    """Shared normalization dev doc §9.1 states explicitly for risk
    ("normalized to [0,1] by the ... 95th percentile") and this module
    applies consistently to §9.2's disagreement/interval-width signals
    too (see `RiskCalibration.p95_disagreement_raw`'s docstring for why
    that's a disclosed, not dev-doc-literal, extension). A reference of
    0 (no calibration data) maps everything to 0.0 rather than raising
    or dividing by zero — a disclosed neutral default, not a claim that
    0 risk/disagreement is actually known."""
    if p95_reference <= 0:
        return 0.0
    return max(0.0, min(1.0, raw / p95_reference))


def compute_risk(
    result: SimulationResult, calibration: RiskCalibration
) -> tuple[float, float, float, float]:
    """dev doc §9.1's exact formula. Returns `(p_failure, consequence,
    risk_raw, risk_normalized)`.

    "Criticality-weighted unmet demand" (dev doc §9.1's own wording) is
    read here as `unmet_patient_hours` unweighted: no per-hospital
    criticality weight exists anywhere else in this codebase either
    (`DependencyEdge.criticality` is a per-*edge* property feeding the
    cascade formula, not a per-hospital output-side weight `simulate()`'s
    summary could read) — a disclosed simplification, not a silent
    substitution of a different quantity."""
    failing = [r for r in result.rollouts if r.min_hospital_functional_level < 0.3]
    if not failing:
        return result.p_failure, 0.0, 0.0, 0.0
    consequence = sum(
        r.unmet_patient_hours + CASCADE_WEIGHT * r.new_cascading_failures for r in failing
    ) / len(failing)
    risk_raw = result.p_failure * consequence
    risk = normalize_to_p95(risk_raw, calibration.p95_risk_raw)
    return result.p_failure, consequence, risk_raw, risk


def compute_confidence(critic_returns: list[float], calibration: RiskCalibration) -> float:
    """dev doc §9.2: `confidence = 1 - max(disagreement_norm,
    interval_width_norm)`."""
    disagreement_raw = compute_disagreement(critic_returns)
    disagreement_norm = normalize_to_p95(disagreement_raw, calibration.p95_disagreement_raw)
    width = interval_width(calibration.conformal_threshold)
    width_norm = normalize_to_p95(width, calibration.p95_interval_width)
    return 1.0 - max(disagreement_norm, width_norm)


def assess(
    result: SimulationResult,
    critic_returns: list[float],
    calibration: RiskCalibration,
) -> RiskAssessment:
    """One call per decision — the top-level entry point `risk/gate.py`
    (via `decide_tier`) and dev doc §10's decision log both consume."""
    p_failure, consequence, risk_raw, risk = compute_risk(result, calibration)
    confidence = compute_confidence(critic_returns, calibration)
    return RiskAssessment(
        p_failure=p_failure,
        consequence=consequence,
        risk_raw=risk_raw,
        risk=risk,
        confidence=confidence,
    )


def calibrate_from_samples(
    risk_raw_samples: list[float],
    disagreement_raw_samples: list[float],
    nonconformity_scores: list[float],
    *,
    coverage: float = DEFAULT_COVERAGE,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = 0,
) -> RiskCalibration:
    """Builds a real `RiskCalibration` from calibration-set samples —
    see module docstring for what this session substitutes for the dev
    doc's true held-out "val scenarios"/"val episodes".

    `p95_interval_width` needs its own reference *distribution* of
    interval widths, but a single conformal calibration pass produces
    exactly one threshold, not a distribution — resolved by bootstrap-
    resampling `nonconformity_scores` `n_bootstrap` times (with
    replacement, dev doc §3.6's own explicit-seeded-rng convention) and
    calibrating each resample independently, then taking the 95th
    percentile of *those* widths as the reference scale for "how wide
    counts as wide"."""
    rng = np.random.default_rng(seed)
    p95_risk = float(np.percentile(risk_raw_samples, 95)) if risk_raw_samples else 0.0
    p95_disagreement = (
        float(np.percentile(disagreement_raw_samples, 95)) if disagreement_raw_samples else 0.0
    )

    if nonconformity_scores:
        threshold = calibrate(nonconformity_scores, coverage)
        scores_arr = np.asarray(nonconformity_scores, dtype=float)
        bootstrap_widths = [
            interval_width(
                calibrate(rng.choice(scores_arr, size=scores_arr.size).tolist(), coverage)
            )
            for _ in range(n_bootstrap)
        ]
        p95_width = float(np.percentile(bootstrap_widths, 95))
    else:
        threshold = 0.0
        p95_width = 0.0

    return RiskCalibration(
        p95_risk_raw=p95_risk,
        p95_disagreement_raw=p95_disagreement,
        conformal_threshold=threshold,
        p95_interval_width=p95_width,
    )
