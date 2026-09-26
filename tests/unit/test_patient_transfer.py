"""Phase 4 acceptance tests: `twin.demand.apply_patient_transfer`
(enactment) and `agents.rule_based._pick_patient_transfer` (decision) —
dev doc §5.6's patient-transfer rule."""

from __future__ import annotations

import pytest

from udt.agents.rule_based import PREDICTED_LOSS_THRESHOLD_HOURS, _pick_patient_transfer
from udt.common.models import Asset, AssetType, DependencyEdge, DependencyGraph
from udt.twin.cascade import EdgeRuntimeState
from udt.twin.demand import apply_patient_transfer
from udt.twin.graph import build_networkx_graph

POINT_A = {"type": "Point", "coordinates": [72.87, 19.07]}
POINT_B = {"type": "Point", "coordinates": [72.90, 19.07]}  # further from the substation


def _hospital(asset_id: str, geometry: dict, beds_total: int, beds_occupied: int) -> Asset:
    return Asset(
        asset_id=asset_id,
        asset_type=AssetType.HOSPITAL,
        geometry=geometry,
        attributes={"beds_total": beds_total, "beds_occupied": beds_occupied},
    )


@pytest.mark.phase4
def test_apply_patient_transfer_moves_up_to_destination_capacity() -> None:
    graph_model = DependencyGraph(
        assets=[
            _hospital("H1", POINT_A, beds_total=100, beds_occupied=50),
            _hospital("H2", POINT_B, beds_total=100, beds_occupied=95),  # only 5 free
        ],
        edges=[],
    )
    g = build_networkx_graph(graph_model)
    moved = apply_patient_transfer(g, "H1", "H2", 10)
    assert moved == 5  # capped by destination's free beds, not the requested 10
    assert g.nodes["H1"]["asset"].attributes["beds_occupied"] == 45
    assert g.nodes["H2"]["asset"].attributes["beds_occupied"] == 100


@pytest.mark.phase4
def test_apply_patient_transfer_capped_by_source_occupancy() -> None:
    graph_model = DependencyGraph(
        assets=[
            _hospital("H1", POINT_A, beds_total=100, beds_occupied=3),
            _hospital("H2", POINT_B, beds_total=100, beds_occupied=0),
        ],
        edges=[],
    )
    g = build_networkx_graph(graph_model)
    moved = apply_patient_transfer(g, "H1", "H2", 10)
    assert moved == 3
    assert g.nodes["H1"]["asset"].attributes["beds_occupied"] == 0
    assert g.nodes["H2"]["asset"].attributes["beds_occupied"] == 3


def _graph_with_power_edge(
    h1_beds_occupied: int, h2_beds_occupied: int, remaining_hours: float
) -> tuple:
    substation = Asset(
        asset_id="S1",
        asset_type=AssetType.SUBSTATION,
        geometry=POINT_A,
        attributes={"capacity_mw": 30.0},
    )
    h1 = _hospital("H1", POINT_A, beds_total=100, beds_occupied=h1_beds_occupied)
    h2 = _hospital("H2", POINT_B, beds_total=100, beds_occupied=h2_beds_occupied)
    edge = DependencyEdge(
        edge_id="E1_power",
        supplier="S1",
        consumer="H1",
        kind="power",
        demand=0.1,
        criticality=0.9,
        buffer_hours=8.0,
        floor=0.3,
    )
    graph_model = DependencyGraph(assets=[substation, h1, h2], edges=[edge])
    g = build_networkx_graph(graph_model)
    edge_states = {
        "E1_power": EdgeRuntimeState(capacity_hours=8.0, remaining_hours=remaining_hours)
    }
    return g, edge_states


@pytest.mark.phase4
def test_pick_patient_transfer_triggers_when_buffer_low() -> None:
    g, edge_states = _graph_with_power_edge(
        h1_beds_occupied=20, h2_beds_occupied=0, remaining_hours=0.5
    )
    assert PREDICTED_LOSS_THRESHOLD_HOURS > 0.5
    decision = _pick_patient_transfer(g, edge_states, road_network=None)
    assert decision is not None
    from_id, to_id, count = decision
    assert from_id == "H1"
    assert to_id == "H2"
    assert count in (2, 5, 10)


@pytest.mark.phase4
def test_pick_patient_transfer_does_nothing_when_buffer_healthy() -> None:
    g, edge_states = _graph_with_power_edge(
        h1_beds_occupied=20, h2_beds_occupied=0, remaining_hours=8.0
    )
    assert _pick_patient_transfer(g, edge_states, road_network=None) is None


@pytest.mark.phase4
def test_pick_patient_transfer_none_without_edge_states() -> None:
    g, _ = _graph_with_power_edge(h1_beds_occupied=20, h2_beds_occupied=0, remaining_hours=0.1)
    assert _pick_patient_transfer(g, edge_states=None, road_network=None) is None


@pytest.mark.phase4
def test_pick_patient_transfer_none_when_no_destination_has_free_beds() -> None:
    g, edge_states = _graph_with_power_edge(
        h1_beds_occupied=20, h2_beds_occupied=100, remaining_hours=0.1
    )  # H2 is full -> nowhere to send patients
    assert _pick_patient_transfer(g, edge_states, road_network=None) is None
