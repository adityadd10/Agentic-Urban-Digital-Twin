"""Phase 1 acceptance tests (dev doc §13 "Unit — cascade" row): hand-
computed small scenarios asserting the functional_level formula, buffer
drain/refill, floor clipping, and fixed-point convergence."""

from __future__ import annotations

import pytest

from udt.common.models import Asset, AssetType, DependencyEdge, DependencyGraph
from udt.twin.cascade import EdgeRuntimeState, resolve_functional_levels, sat, update_buffers
from udt.twin.graph import build_networkx_graph

POINT = {"type": "Point", "coordinates": [72.88, 19.07]}


def _power_edge_graph(buffer_hours: float = 4.0, floor: float = 0.3, demand: float = 0.5) -> tuple:
    """S1 --power--> H1, the minimal case the whole formula hangs on."""
    assets = [
        Asset(
            asset_id="S1",
            asset_type=AssetType.SUBSTATION,
            geometry=POINT,
            intrinsic_level=1.0,
            functional_level=1.0,
            attributes={"capacity_mw": 10.0},
        ),
        Asset(
            asset_id="H1",
            asset_type=AssetType.HOSPITAL,
            geometry=POINT,
            intrinsic_level=1.0,
            functional_level=1.0,
            attributes={},
        ),
    ]
    edge = DependencyEdge(
        edge_id="E1",
        supplier="S1",
        consumer="H1",
        kind="power",
        demand=demand,
        criticality=0.9,
        buffer_hours=buffer_hours,
        floor=floor,
    )
    dep_graph = DependencyGraph(assets=assets, edges=[edge])
    g = build_networkx_graph(dep_graph)
    edge_states = {edge.edge_id: EdgeRuntimeState.initial(edge)}
    return g, edge_states, edge


@pytest.mark.phase1
def test_functional_level_full_health_is_one() -> None:
    g, edge_states, _ = _power_edge_graph()
    levels = resolve_functional_levels(g, edge_states)
    assert levels["S1"] == pytest.approx(1.0)
    assert levels["H1"] == pytest.approx(1.0)


@pytest.mark.phase1
def test_supplier_collapse_propagates_once_buffer_exhausted() -> None:
    """Substation drops to 0; with no buffer remaining, hospital's sat()
    hits its floor exactly (dev doc §3.3's clip formula)."""
    g, edge_states, edge = _power_edge_graph(buffer_hours=0.0, floor=0.3, demand=0.5)
    g.nodes["S1"]["asset"].intrinsic_level = 0.0
    g.nodes["S1"]["asset"].functional_level = 0.0
    levels = resolve_functional_levels(g, edge_states)
    assert levels["S1"] == pytest.approx(0.0)
    # supply=0, demand=0.5 -> ratio=0 -> clipped to floor=0.3
    assert levels["H1"] == pytest.approx(0.3)


@pytest.mark.phase1
def test_buffer_masks_supplier_collapse_while_remaining() -> None:
    """With buffer still remaining, sat()=1.0 regardless of supply — the
    exact mechanism dev doc §3.3 says makes cascades unfold over hours."""
    g, edge_states, edge = _power_edge_graph(buffer_hours=4.0, floor=0.3, demand=0.5)
    g.nodes["S1"]["asset"].intrinsic_level = 0.0
    g.nodes["S1"]["asset"].functional_level = 0.0
    levels = resolve_functional_levels(g, edge_states)
    assert levels["H1"] == pytest.approx(1.0)  # buffer still full -> masked


@pytest.mark.phase1
def test_buffer_drains_when_supply_insufficient_and_refills_at_half_rate() -> None:
    g, edge_states, edge = _power_edge_graph(buffer_hours=4.0, floor=0.3, demand=0.5)
    state = edge_states[edge.edge_id]
    g.nodes["S1"]["asset"].intrinsic_level = 0.0
    g.nodes["S1"]["asset"].functional_level = 0.0  # supply=0 < demand=0.5 -> insufficient

    update_buffers(g, edge_states, dt_hours=1.0)
    assert state.remaining_hours == pytest.approx(3.0)  # drained by dt
    update_buffers(g, edge_states, dt_hours=1.0)
    assert state.remaining_hours == pytest.approx(2.0)

    # supply restored -> refills at HALF the drain rate
    g.nodes["S1"]["asset"].functional_level = 1.0  # supply=1.0 >= demand=0.5 -> sufficient
    update_buffers(g, edge_states, dt_hours=1.0)
    assert state.remaining_hours == pytest.approx(2.5)  # +0.5, not +1.0

    # refill caps at the edge's original capacity, never exceeds it
    for _ in range(10):
        update_buffers(g, edge_states, dt_hours=1.0)
    assert state.remaining_hours == pytest.approx(4.0)


@pytest.mark.phase1
def test_buffer_never_drains_below_zero() -> None:
    g, edge_states, edge = _power_edge_graph(buffer_hours=1.0, floor=0.3, demand=0.5)
    state = edge_states[edge.edge_id]
    g.nodes["S1"]["asset"].functional_level = 0.0
    for _ in range(10):
        update_buffers(g, edge_states, dt_hours=1.0)
    assert state.remaining_hours == pytest.approx(0.0)


@pytest.mark.phase1
def test_sat_uses_floor_not_zero_when_ratio_below_floor() -> None:
    edge = DependencyEdge(
        edge_id="E",
        supplier="S",
        consumer="C",
        kind="power",
        demand=1.0,
        criticality=0.9,
        buffer_hours=0.0,
        floor=0.3,
    )
    assert sat(edge, supply=0.0, buffer_remaining=0.0) == pytest.approx(0.3)
    assert sat(edge, supply=0.5, buffer_remaining=0.0) == pytest.approx(
        0.5
    )  # between floor and 1.0
    assert sat(edge, supply=2.0, buffer_remaining=0.0) == pytest.approx(
        1.0
    )  # clipped at 1.0, not 2.0


@pytest.mark.phase1
def test_access_edges_use_max_not_product_across_multiple_roads() -> None:
    """cascade.py's module docstring point 1: a hospital with two nearby
    roads, one blocked, should still have full access via the clear one —
    not be multiplicatively penalized by the blocked road."""
    assets = [
        Asset(
            asset_id="H1",
            asset_type=AssetType.HOSPITAL,
            geometry=POINT,
            intrinsic_level=1.0,
            functional_level=1.0,
            attributes={},
        ),
        Asset(
            asset_id="R_blocked",
            asset_type=AssetType.ROAD,
            geometry=POINT,
            intrinsic_level=1.0,
            functional_level=1.0,
            attributes={"blockage": 1.0},
        ),
        Asset(
            asset_id="R_clear",
            asset_type=AssetType.ROAD,
            geometry=POINT,
            intrinsic_level=1.0,
            functional_level=1.0,
            attributes={"blockage": 0.0},
        ),
    ]
    edges = [
        DependencyEdge(
            edge_id="Eb",
            supplier="R_blocked",
            consumer="H1",
            kind="access",
            demand=1.0,
            criticality=0.6,
            buffer_hours=0.0,
            floor=0.7,
        ),
        DependencyEdge(
            edge_id="Ec",
            supplier="R_clear",
            consumer="H1",
            kind="access",
            demand=1.0,
            criticality=0.6,
            buffer_hours=0.0,
            floor=0.7,
        ),
    ]
    dep_graph = DependencyGraph(assets=assets, edges=edges)
    g = build_networkx_graph(dep_graph)
    edge_states = {e.edge_id: EdgeRuntimeState.initial(e) for e in edges}
    levels = resolve_functional_levels(g, edge_states)
    assert levels["H1"] == pytest.approx(
        1.0
    )  # clear road wins, not multiplied with the blocked one


@pytest.mark.phase1
def test_road_functional_level_reflects_blockage_not_intrinsic() -> None:
    assets = [
        Asset(
            asset_id="R1",
            asset_type=AssetType.ROAD,
            geometry=POINT,
            intrinsic_level=1.0,
            functional_level=1.0,
            attributes={"blockage": 0.4},
        )
    ]
    dep_graph = DependencyGraph(assets=assets, edges=[])
    g = build_networkx_graph(dep_graph)
    levels = resolve_functional_levels(g, {})
    assert levels["R1"] == pytest.approx(0.6)  # 1.0 * (1 - 0.4)


@pytest.mark.phase1
def test_cascade_chain_substation_water_hospital() -> None:
    """The dev doc's headline cascade chain (§2.2): substation -> water ->
    hospital. Water's power collapses -> water collapses (floor=0.0,
    'pumps just stop') -> hospital's water-sat hits its floor too, once
    both buffers are exhausted."""
    assets = [
        Asset(
            asset_id="S1",
            asset_type=AssetType.SUBSTATION,
            geometry=POINT,
            intrinsic_level=0.0,
            functional_level=0.0,
            attributes={},
        ),
        Asset(
            asset_id="W1",
            asset_type=AssetType.WATER,
            geometry=POINT,
            intrinsic_level=1.0,
            functional_level=1.0,
            attributes={},
        ),
        Asset(
            asset_id="H1",
            asset_type=AssetType.HOSPITAL,
            geometry=POINT,
            intrinsic_level=1.0,
            functional_level=1.0,
            attributes={},
        ),
    ]
    edges = [
        DependencyEdge(
            edge_id="Epw",
            supplier="S1",
            consumer="W1",
            kind="power",
            demand=0.5,
            criticality=0.9,
            buffer_hours=0.0,
            floor=0.0,
        ),
        DependencyEdge(
            edge_id="Ewh",
            supplier="W1",
            consumer="H1",
            kind="water",
            demand=0.1,
            criticality=0.7,
            buffer_hours=0.0,
            floor=0.5,
        ),
    ]
    dep_graph = DependencyGraph(assets=assets, edges=edges)
    g = build_networkx_graph(dep_graph)
    edge_states = {e.edge_id: EdgeRuntimeState.initial(e) for e in edges}
    levels = resolve_functional_levels(g, edge_states)
    assert levels["S1"] == pytest.approx(0.0)
    assert levels["W1"] == pytest.approx(0.0)  # power->water floor=0.0, no buffer
    assert levels["H1"] == pytest.approx(0.5)  # water->hospital floor=0.5, supply=0 < demand
