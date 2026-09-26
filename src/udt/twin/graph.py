"""Dependency graph loading (dev doc §3.4, module M1).

Wraps a `DependencyGraph` (the Pydantic model in `common/models.py`, the
twin's canonical input per the dev doc) in a NetworkX `DiGraph` for the
cascade engine (`cascade.py`) to traverse — edges point supplier -> consumer,
matching `DependencyEdge.supplier`/`.consumer`.
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from udt.common.models import Asset, DependencyEdge, DependencyGraph


def load_dependency_graph(path: str | Path) -> DependencyGraph:
    """Load and validate `dependency_graph.json` (dev doc §3.4 schema) —
    the file written by `scripts/06_build_dependency_graph.py`."""
    with open(path) as f:
        data = json.load(f)
    return DependencyGraph.model_validate(data)


def build_networkx_graph(dep_graph: DependencyGraph) -> nx.DiGraph[str]:
    """Build a NetworkX `DiGraph` from a validated `DependencyGraph`.

    Node attribute `asset` holds the `Asset`; edge attribute `edge` holds
    the `DependencyEdge`. A consumer can have multiple edges from the same
    supplier of different `kind`s (e.g. a hospital's power and water edges
    from different suppliers) — `DiGraph` allows only one edge per
    (u, v) pair, so this asserts that assumption holds for the loaded
    graph rather than silently dropping a duplicate.
    """
    g: nx.DiGraph[str] = nx.DiGraph()  # subscript is type-only — DiGraph isn't runtime-generic
    for asset in dep_graph.assets:
        g.add_node(asset.asset_id, asset=asset)

    seen_pairs: set[tuple[str, str]] = set()
    for edge in dep_graph.edges:
        if edge.supplier not in g:
            raise ValueError(f"Edge {edge.edge_id!r} references unknown supplier {edge.supplier!r}")
        if edge.consumer not in g:
            raise ValueError(f"Edge {edge.edge_id!r} references unknown consumer {edge.consumer!r}")
        pair = (edge.supplier, edge.consumer)
        if pair in seen_pairs:
            raise ValueError(
                f"Duplicate supplier->consumer pair {pair} (edge {edge.edge_id!r}) — "
                "build_networkx_graph assumes at most one edge per ordered pair"
            )
        seen_pairs.add(pair)
        g.add_edge(edge.supplier, edge.consumer, edge=edge)
    return g


def get_asset(g: nx.DiGraph[str], asset_id: str) -> Asset:
    return g.nodes[asset_id]["asset"]  # type: ignore[no-any-return]


def dependency_edges_of(g: nx.DiGraph[str], asset_id: str) -> list[DependencyEdge]:
    """All edges where `asset_id` is the consumer — `D(a)` in dev doc §3.3."""
    return [g.edges[supplier, asset_id]["edge"] for supplier in g.predecessors(asset_id)]
