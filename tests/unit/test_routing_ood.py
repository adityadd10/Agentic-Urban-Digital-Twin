"""Phase 9 acceptance tests: `routing.ood` (dev doc §6's ID/OOD router),
module M9b — built for the *open-set* case per explicit user
instruction, even though only "flood" exists to route against today."""

from __future__ import annotations

import numpy as np
import pytest

from udt.common.models import (
    Asset,
    AssetType,
    DependencyEdge,
    DependencyGraph,
    Incident,
    PolicyRegistryEntry,
    RouterCalibration,
)
from udt.routing.ood import (
    P_VALUE_OOD_THRESHOLD,
    conformal_p_value,
    featurize_incident,
    knn_novelty_score,
    registry_check,
    route,
)

POINT = {"type": "Point", "coordinates": [72.88, 19.07]}
POLYGON = {
    "type": "Polygon",
    "coordinates": [
        [[72.86, 19.05], [72.91, 19.05], [72.91, 19.10], [72.86, 19.10], [72.86, 19.05]]
    ],
}


def _incident(**overrides: object) -> Incident:
    base = dict(
        incident_id="test",
        type="flood",
        location=POLYGON,
        onset_tick=0,
        severity=0.7,
        directly_affected_assets=[],
    )
    base.update(overrides)
    return Incident(**base)  # type: ignore[arg-type]


def _dep_graph() -> DependencyGraph:
    s1 = Asset(asset_id="S1", asset_type=AssetType.SUBSTATION, geometry=POINT, attributes={})
    h1 = Asset(asset_id="H1", asset_type=AssetType.HOSPITAL, geometry=POINT, attributes={})
    edge = DependencyEdge(
        edge_id="E1",
        supplier="S1",
        consumer="H1",
        kind="power",
        demand=1.0,
        criticality=0.8,
        buffer_hours=4.0,
        floor=0.2,
    )
    return DependencyGraph(assets=[s1, h1], edges=[edge])


REGISTRY_ENTRY = PolicyRegistryEntry(
    incident_type="flood",
    policy_path="run/model.pt",
    env_version="udt_multi_env_v0",
    suite_version="none-n1-scenario",
    val_score=-1.0,
    trained_date="2026-01-01T00:00:00+00:00",
)


# ---------------------------------------------------------------------------
# featurize_incident
# ---------------------------------------------------------------------------
@pytest.mark.phase9
def test_featurize_incident_has_six_dimensions_per_dev_doc() -> None:
    x = featurize_incident(_incident(), _dep_graph())
    assert x.shape == (6,)


@pytest.mark.phase9
def test_featurize_incident_first_dimension_is_severity() -> None:
    x = featurize_incident(_incident(severity=0.42), _dep_graph())
    assert x[0] == pytest.approx(0.42)


@pytest.mark.phase9
def test_featurize_incident_footprint_area_is_zero_for_a_point() -> None:
    x = featurize_incident(_incident(location=POINT), _dep_graph())
    assert x[1] == pytest.approx(0.0)


@pytest.mark.phase9
def test_featurize_incident_footprint_area_is_positive_for_a_polygon() -> None:
    x = featurize_incident(_incident(location=POLYGON), _dep_graph())
    assert x[1] > 0.0


@pytest.mark.phase9
def test_featurize_incident_uses_mean_criticality_of_all_edges_when_none_directly_affected() -> (
    None
):
    x = featurize_incident(_incident(directly_affected_assets=[]), _dep_graph())
    assert x[5] == pytest.approx(0.8)  # the one edge's own criticality


@pytest.mark.phase9
def test_featurize_incident_restricts_criticality_to_affected_assets() -> None:
    graph = _dep_graph()
    graph.assets.append(
        Asset(asset_id="H2", asset_type=AssetType.HOSPITAL, geometry=POINT, attributes={})
    )
    graph.edges.append(
        DependencyEdge(
            edge_id="E2",
            supplier="S1",
            consumer="H2",
            kind="power",
            demand=1.0,
            criticality=0.2,
            buffer_hours=4.0,
            floor=0.2,
        )
    )
    x = featurize_incident(_incident(directly_affected_assets=["H1"]), graph)
    assert x[5] == pytest.approx(0.8)  # only E1 (touches H1), not E2's 0.2


@pytest.mark.phase9
def test_featurize_incident_onset_hour_sin_cos_at_midnight_and_noon() -> None:
    midnight = featurize_incident(_incident(), _dep_graph(), onset_hour_of_day=0.0)
    noon = featurize_incident(_incident(), _dep_graph(), onset_hour_of_day=12.0)
    assert midnight[3] == pytest.approx(0.0, abs=1e-9)  # sin(0)
    assert midnight[4] == pytest.approx(1.0)  # cos(0)
    assert noon[3] == pytest.approx(0.0, abs=1e-9)  # sin(2pi*0.5)
    assert noon[4] == pytest.approx(-1.0)  # cos(2pi*0.5)


# ---------------------------------------------------------------------------
# registry_check
# ---------------------------------------------------------------------------
@pytest.mark.phase9
def test_registry_check_hits_on_matching_type_and_env_version() -> None:
    assert registry_check("flood", "udt_multi_env_v0", [REGISTRY_ENTRY]) is True


@pytest.mark.phase9
def test_registry_check_misses_on_wrong_incident_type() -> None:
    assert registry_check("substation_failure", "udt_multi_env_v0", [REGISTRY_ENTRY]) is False


@pytest.mark.phase9
def test_registry_check_misses_on_wrong_env_version() -> None:
    assert registry_check("flood", "udt_multi_env_v1", [REGISTRY_ENTRY]) is False


@pytest.mark.phase9
def test_registry_check_misses_on_empty_registry() -> None:
    assert registry_check("flood", "udt_multi_env_v0", []) is False


# ---------------------------------------------------------------------------
# knn_novelty_score / conformal_p_value
# ---------------------------------------------------------------------------
@pytest.mark.phase9
def test_knn_novelty_score_is_zero_for_a_point_identical_to_training_data() -> None:
    training = np.array([[0.5, 1.0, 0.0, 0.0, 1.0, 0.5]] * 5)
    x = np.array([0.5, 1.0, 0.0, 0.0, 1.0, 0.5])
    assert knn_novelty_score(x, training, k=5) == pytest.approx(0.0)


@pytest.mark.phase9
def test_knn_novelty_score_matches_hand_computed_mean_distance() -> None:
    training = np.array([[0.0] * 6, [1.0] * 6, [2.0] * 6])
    x = np.array([0.0] * 6)
    # distances: 0, sqrt(6), sqrt(24) - k=2 nearest: 0 and sqrt(6)
    expected = float(np.mean([0.0, np.sqrt(6)]))
    assert knn_novelty_score(x, training, k=2) == pytest.approx(expected)


@pytest.mark.phase9
def test_knn_novelty_score_is_infinite_with_no_training_data() -> None:
    x = np.zeros(6)
    assert knn_novelty_score(x, np.empty((0, 6)), k=5) == float("inf")


@pytest.mark.phase9
def test_conformal_p_value_is_one_when_score_is_the_most_extreme_low() -> None:
    # score lower than every calibration score -> every one counts as
    # ">= score" -> p = (1+n)/(1+n) = 1.0
    assert conformal_p_value(0.0, [1.0, 2.0, 3.0]) == pytest.approx(1.0)


@pytest.mark.phase9
def test_conformal_p_value_is_smallest_when_score_exceeds_everything() -> None:
    # score bigger than every calibration score -> p = (1+0)/(1+n)
    assert conformal_p_value(100.0, [1.0, 2.0, 3.0]) == pytest.approx(1 / 4)


@pytest.mark.phase9
def test_conformal_p_value_is_zero_with_no_calibration_data() -> None:
    assert conformal_p_value(1.0, []) == 0.0


# ---------------------------------------------------------------------------
# route() - end to end
# ---------------------------------------------------------------------------
def _calibration(
    training_features: list[list[float]], calibration_scores: list[float]
) -> RouterCalibration:
    return RouterCalibration(
        incident_type="flood",
        env_version="udt_multi_env_v0",
        training_features=training_features,
        calibration_scores=calibration_scores,
        k=5,
    )


@pytest.mark.phase9
def test_route_is_ood_when_incident_type_is_unknown() -> None:
    decision = route(
        _incident(type="unknown"),
        _dep_graph(),
        env_version="udt_multi_env_v0",
        registry_entries=[REGISTRY_ENTRY],
        calibration=None,
    )
    assert decision.path == "OOD"
    assert "unknown" in decision.reasons[0]


@pytest.mark.phase9
def test_route_is_ood_when_type_not_in_registry() -> None:
    decision = route(
        _incident(type="substation_failure"),
        _dep_graph(),
        env_version="udt_multi_env_v0",
        registry_entries=[REGISTRY_ENTRY],
        calibration=None,
    )
    assert decision.path == "OOD"
    assert decision.registry_hit is False


@pytest.mark.phase9
def test_route_is_ood_with_no_calibration_even_if_registry_hits() -> None:
    decision = route(
        _incident(),
        _dep_graph(),
        env_version="udt_multi_env_v0",
        registry_entries=[REGISTRY_ENTRY],
        calibration=None,
    )
    assert decision.path == "OOD"
    assert decision.registry_hit is True  # registry passed, support check couldn't run


@pytest.mark.phase9
def test_route_is_id_when_registry_hits_and_incident_matches_training_distribution() -> None:
    incident = _incident(severity=0.7)
    x = featurize_incident(incident, _dep_graph())
    training = [x.tolist()] * 10  # incident is dead-center of its own training set
    calibration_scores = [0.0] * 10  # every calibration scenario also matched perfectly
    decision = route(
        incident,
        _dep_graph(),
        env_version="udt_multi_env_v0",
        registry_entries=[REGISTRY_ENTRY],
        calibration=_calibration(training, calibration_scores),
    )
    assert decision.path == "ID"
    assert decision.support_score >= P_VALUE_OOD_THRESHOLD


@pytest.mark.phase9
def test_route_is_ood_when_incident_is_statistically_novel() -> None:
    """The open-set case this module was explicitly built for: same
    known incident *type* (registry hits), but a severity far outside
    anything the training set ever saw."""
    training = [[0.5, 10.0, 0.0, 0.0, 1.0, 0.5]] * 20
    calibration_scores = [
        knn_novelty_score(np.array(row), np.array(training), k=5)
        for row in [[0.5 + d, 10.0, 0.0, 0.0, 1.0, 0.5] for d in np.linspace(-0.05, 0.05, 20)]
    ]
    novel_incident = _incident(severity=0.999999)  # nowhere near training severities (~0.5)
    decision = route(
        novel_incident,
        _dep_graph(),
        env_version="udt_multi_env_v0",
        registry_entries=[REGISTRY_ENTRY],
        calibration=_calibration(training, calibration_scores),
    )
    assert decision.path == "OOD"
    assert decision.registry_hit is True  # the *type* is known - it's the *distribution* that's off
    assert decision.support_score < P_VALUE_OOD_THRESHOLD


@pytest.mark.phase9
def test_route_reasons_are_a_real_audit_trail_not_empty() -> None:
    decision = route(
        _incident(type="unknown"),
        _dep_graph(),
        env_version="udt_multi_env_v0",
        registry_entries=[],
        calibration=None,
    )
    assert len(decision.reasons) > 0
