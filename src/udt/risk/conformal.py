"""Conformal prediction interval (dev doc §9.2, module M8).

Dev doc §9.2: "Conformal interval: nonconformity = |simulated mean
outcome - realized outcome| on val episodes -> 90% prediction interval
on the outcome metric; wide interval => low confidence."

Standard split-conformal calibration (Vovk, Gammerman & Shafer 2005,
"Algorithmic Learning in a Random World"; the regression-interval form
used here follows Lei, G'Sell, Rinaldo, Tibshirani & Wasserman 2018,
"Distribution-Free Predictive Inference for Regression", JASA) — the
calibration threshold for a `coverage`-level interval is the
`ceil((n+1)*coverage)`-th order statistic of the nonconformity scores,
not a plain percentile; using that exact (slightly-more-than-`coverage`)
rank is what gives split conformal its finite-sample coverage guarantee
regardless of the underlying error distribution.

**Disclosed prerequisite gap:** the "val episodes" dev doc §9.2 asks the
nonconformity scores to come from need a frozen train/val/test scenario
suite (M3's own deferral, never built this session). `calibrate` below
is generic over whatever nonconformity scores a caller supplies —
`risk/engine.py`'s `calibrate_from_samples` and this module's own tests
generate a real small batch on the fly instead of drawing from a true
held-out split, flagged the same way M6b's `val_score_is_true_
holdout=False` already is.
"""

from __future__ import annotations

import math


def calibrate(nonconformity_scores: list[float], coverage: float = 0.9) -> float:
    """Returns the calibrated threshold (the interval's half-width) for
    a `coverage`-level prediction interval, dev doc §9.2's default
    `coverage=0.9`."""
    if not nonconformity_scores:
        raise ValueError("calibrate needs at least one nonconformity score")
    n = len(nonconformity_scores)
    sorted_scores = sorted(nonconformity_scores)
    # ceil((n+1) x coverage)-th order statistic, 1-indexed; capped at n
    # since coverage close to 1.0 can ask for the (n+1)-th of only n
    # scores (the textbook fix: treat that case as "use the largest
    # score", i.e. an infinite/most-conservative interval given so few
    # calibration points).
    rank = min(math.ceil((n + 1) * coverage), n)
    return sorted_scores[rank - 1]


def interval_width(threshold: float) -> float:
    """A symmetric residual-based interval's full width — `calibrate`'s
    threshold IS the half-width (nonconformity = an absolute residual),
    so the full width is twice that."""
    return 2.0 * threshold
