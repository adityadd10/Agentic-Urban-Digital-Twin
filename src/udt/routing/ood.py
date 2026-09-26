"""ID/OOD router (dev doc §6), module M9b.

**The open-set vs. closed-set decision that blocked this module is now
resolved** (explicit user instruction): the system is being built for
the *open-set* case — it must genuinely detect incidents statistically
unlike anything it was trained on, not just check membership in a fixed
list — even though, today, only one incident type ("flood") exists to
train or route against at all. Every function below is written
generically for that open-set goal; nothing here is hardcoded to
"flood" specifically.

```
1. Registry check: incident.type present in policies.yaml with matching
   env_version. Type "unknown" => instant OOD.
2. Support check: featurize incident -> x = [severity, footprint_area_km2,
   n_affected_assets, onset_hour_sin, onset_hour_cos,
   mean_criticality_of_affected]. Novelty score = k-NN distance (k=5) to
   the training-scenario feature set of that type; conformal p-value
   from val-set distances. p < 0.05 => OOD.

Both must pass for ID.
```

**Disclosed limitation of today's actual data, not a bug in this
module's logic:** `scenarios/generator.py`'s `generate_flood_
scenario` (M3) currently produces scenarios where 4 of the 6 feature
dimensions are constant across every call — `footprint_area_km2` (always
the same ward polygon), `n_affected_assets` (flood has no point
footprint, so `Incident.directly_affected_assets` is always `[]`, per
that module's own docstring), and `onset_hour_sin`/`onset_hour_cos`
(nothing in this codebase varies onset-time-of-day yet). Only `severity`
(and, weakly, `mean_criticality_of_affected` once `directly_affected_
assets` is ever populated) actually varies today. The k-NN/conformal
machinery below handles all 6 dimensions correctly and will
discriminate properly the moment scenario generation produces real
variation along those other axes (multiple wards, varying onset times,
additional incident types) — this is a real gap in M3's scope, not
something to silently paper over here by inventing fake variation.

**No live scenario suite exists** (M3's own deferral, unresolved all
session) — `scripts/calibrate_router.py` builds a real `RouterCalibration`
from freshly-generated scenarios instead of a true held-out train/val
split, same disclosed status as `risk/engine.py`'s own calibration.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from shapely.geometry import shape

from udt.common.models import (
    DependencyGraph,
    Incident,
    PolicyRegistryEntry,
    RouterCalibration,
    RoutingDecision,
)

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

# dev doc §6 exact.
DEFAULT_K = 5
P_VALUE_OOD_THRESHOLD = 0.05

# Degrees-of-latitude -> km, standard approximation (Mumbai sits at
# ~19N, small enough a ward polygon that a flat approximation — not a
# true geodesic area — is adequate; disclosed, same status as this
# codebase's other geometry simplifications, e.g. `agents/rule_based.
# py`'s straight-line-distance fallback).
_KM_PER_DEG_LAT = 111.32


def _footprint_area_km2(geometry: dict[str, object]) -> float:
    polygon = shape(geometry)
    if polygon.geom_type not in ("Polygon", "MultiPolygon"):
        return 0.0  # a bare point incident has no footprint
    km_per_deg_lon = _KM_PER_DEG_LAT * math.cos(math.radians(polygon.centroid.y))
    return float(polygon.area * _KM_PER_DEG_LAT * km_per_deg_lon)


def featurize_incident(
    incident: Incident,
    dep_graph: DependencyGraph,
    *,
    onset_hour_of_day: float = 0.0,
) -> FloatArray:
    """dev doc §6's `x = [severity, footprint_area_km2, n_affected_assets,
    onset_hour_sin, onset_hour_cos, mean_criticality_of_affected]`.

    `onset_hour_of_day` isn't a field on `Incident`/`Scenario` anywhere
    in this codebase (only `Simulator.__init__`'s own constructor
    param) — passed explicitly by the caller, which already has to know
    it to build the `Simulator` in the first place, rather than
    inventing a new field on `Incident` this session's data model
    doesn't otherwise need.

    `mean_criticality_of_affected` falls back to the mean criticality
    across *every* edge in the graph when `directly_affected_assets` is
    empty (true for every flood scenario this codebase currently
    generates, see module docstring) — restricted to edges touching
    those specific assets when it isn't."""
    footprint_area_km2 = _footprint_area_km2(incident.location)
    n_affected_assets = len(incident.directly_affected_assets)
    hour_angle = 2.0 * math.pi * onset_hour_of_day / 24.0
    onset_hour_sin = math.sin(hour_angle)
    onset_hour_cos = math.cos(hour_angle)

    if incident.directly_affected_assets:
        affected = set(incident.directly_affected_assets)
        criticalities = [
            e.criticality
            for e in dep_graph.edges
            if e.supplier in affected or e.consumer in affected
        ]
    else:
        criticalities = [e.criticality for e in dep_graph.edges]
    mean_criticality = float(np.mean(criticalities)) if criticalities else 0.0

    return np.array(
        [
            incident.severity,
            footprint_area_km2,
            float(n_affected_assets),
            onset_hour_sin,
            onset_hour_cos,
            mean_criticality,
        ],
        dtype=np.float64,
    )


def registry_check(
    incident_type: str, env_version: str, registry_entries: list[PolicyRegistryEntry]
) -> bool:
    """dev doc §6: "incident.type present in policies.yaml with matching
    env_version"."""
    return any(
        e.incident_type == incident_type and e.env_version == env_version for e in registry_entries
    )


def knn_novelty_score(x: FloatArray, training_features: FloatArray, k: int = DEFAULT_K) -> float:
    """dev doc §6: "Novelty score = k-NN distance (k=5) to the
    training-scenario feature set of that type" — the mean Euclidean
    distance to the `k` nearest training points (the dev doc doesn't
    pin "k-NN distance" more precisely than that; mean-of-k-nearest is
    the standard reading, not just the single k-th distance).

    Returns `+inf` when there's no training data at all — a genuinely
    novel type with zero history is maximally novel by construction,
    not an error to raise."""
    if training_features.shape[0] == 0:
        return float("inf")
    k = min(k, training_features.shape[0])
    distances = np.linalg.norm(training_features - x, axis=1)
    nearest = np.sort(distances)[:k]
    return float(np.mean(nearest))


def conformal_p_value(score: float, calibration_scores: list[float]) -> float:
    """dev doc §6: "conformal p-value from val-set distances" — the
    standard split-conformal p-value: `(1 + #{calibration scores >=
    score}) / (1 + n)`, the same "+1 smoothing" convention `risk/
    conformal.py`'s calibration already uses elsewhere in this codebase
    (it guarantees the p-value is never exactly 0 from finite data, and
    is uniformly distributed under the null that `x` really is drawn
    from the training distribution — the standard conformal guarantee).

    Returns `0.0` (maximally novel) with no calibration data at all."""
    if not calibration_scores:
        return 0.0
    n = len(calibration_scores)
    count_at_least_as_extreme = sum(1 for s in calibration_scores if s >= score)
    return (1 + count_at_least_as_extreme) / (1 + n)


def route(
    incident: Incident,
    dep_graph: DependencyGraph,
    *,
    env_version: str,
    registry_entries: list[PolicyRegistryEntry],
    calibration: RouterCalibration | None,
    onset_hour_of_day: float = 0.0,
) -> RoutingDecision:
    """dev doc §6: both the registry check and the support check must
    pass for `path="ID"`; either failing (or `incident.type ==
    "unknown"`, or no calibration data existing at all for this
    incident type) routes to `"OOD"`. Every decision's `reasons` records
    which check(s) failed, for dev doc §10's decision log."""
    reasons: list[str] = []

    if incident.type == "unknown":
        return RoutingDecision(
            path="OOD",
            registry_hit=False,
            support_score=0.0,
            reasons=["incident type is 'unknown' -> instant OOD (dev doc §6)"],
        )

    registry_hit = registry_check(incident.type, env_version, registry_entries)
    if not registry_hit:
        reasons.append(
            f"incident type {incident.type!r} not found in registry "
            f"for env_version {env_version!r}"
        )

    if calibration is None or calibration.incident_type != incident.type:
        reasons.append(f"no router calibration available for incident type {incident.type!r}")
        return RoutingDecision(
            path="OOD", registry_hit=registry_hit, support_score=0.0, reasons=reasons
        )

    x = featurize_incident(incident, dep_graph, onset_hour_of_day=onset_hour_of_day)
    training_features = np.array(calibration.training_features, dtype=np.float64)
    score = knn_novelty_score(x, training_features, k=calibration.k)
    p_value = conformal_p_value(score, calibration.calibration_scores)
    support_ok = p_value >= P_VALUE_OOD_THRESHOLD
    if not support_ok:
        reasons.append(
            f"support p-value {p_value:.3f} < {P_VALUE_OOD_THRESHOLD} -> "
            "novel/out-of-distribution incident"
        )

    path = "ID" if (registry_hit and support_ok) else "OOD"
    if path == "ID":
        reasons.append("registry hit and within support -> ID")
    return RoutingDecision(
        path=path, registry_hit=registry_hit, support_score=p_value, reasons=reasons
    )
