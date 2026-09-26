"""Phase 1 tests for `udt.twin.graph`: DependencyGraph -> NetworkX
construction and its validation."""

from __future__ import annotations

import pytest

from udt.common.models import Asset, AssetType, DependencyEdge, DependencyGraph
from udt.twin.graph import build_networkx_graph, dependency_edges_of, get_asset

POINT = {"type": "Point", "coordinates": [72.88, 19.07]}


def _asset(asset_id: str, asset_type: AssetType = AssetType.HOSPITAL) -> Asset:
    return Asset(asset_id=asset_id, asset_type=asset_type, geometry=POINT)


@pytest.mark.phase1
def test_build_networkx_graph_round_trips_assets_and_edges() -> None:
    edge = DependencyEdge(
        edge_id="E1",
        supplier="S1",
        consumer="H1",
        kind="power",
        demand=0.1,
        criticality=0.9,
        buffer_hours=1.0,
        floor=0.3,
    )
    dep_graph = DependencyGraph(
        assets=[_asset("S1", AssetType.SUBSTATION), _asset("H1")], edges=[edge]
    )
    g = build_networkx_graph(dep_graph)

    assert set(g.nodes) == {"S1", "H1"}
    assert get_asset(g, "H1").asset_id == "H1"
    assert dependency_edges_of(g, "H1") == [edge]
    assert dependency_edges_of(g, "S1") == []  # S1 is a supplier, not a consumer here


@pytest.mark.phase1
def test_build_networkx_graph_rejects_unknown_supplier() -> None:
    edge = DependencyEdge(
        edge_id="E1",
        supplier="GHOST",
        consumer="H1",
        kind="power",
        demand=0.1,
        criticality=0.9,
        buffer_hours=1.0,
        floor=0.3,
    )
    dep_graph = DependencyGraph(assets=[_asset("H1")], edges=[edge])
    with pytest.raises(ValueError, match="unknown supplier"):
        build_networkx_graph(dep_graph)


@pytest.mark.phase1
def test_build_networkx_graph_rejects_duplicate_supplier_consumer_pair() -> None:
    e1 = DependencyEdge(
        edge_id="E1",
        supplier="S1",
        consumer="H1",
        kind="power",
        demand=0.1,
        criticality=0.9,
        buffer_hours=1.0,
        floor=0.3,
    )
    e2 = DependencyEdge(
        edge_id="E2",
        supplier="S1",
        consumer="H1",
        kind="power",
        demand=0.2,
        criticality=0.9,
        buffer_hours=1.0,
        floor=0.3,
    )
    dep_graph = DependencyGraph(
        assets=[_asset("S1", AssetType.SUBSTATION), _asset("H1")], edges=[e1, e2]
    )
    with pytest.raises(ValueError, match="Duplicate"):
        build_networkx_graph(dep_graph)
