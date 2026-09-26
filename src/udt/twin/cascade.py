"""Cascade propagation (dev doc §3.3, §3.5 steps 2-3, module M1).

Implements the one formula the dev doc says everything hangs on:

    functional_level(a) = intrinsic_level(a) x PI_{d in D(a)} sat(d)

    supply(d)  = functional_level(supplier)
    buffer(d)  = remaining buffer hours (mutable per-tick state, separate
                 from the DependencyEdge's static `buffer_hours`, which is
                 each edge's *capacity* — see `EdgeRuntimeState`)
    sat(d)     = 1.0                                    if buffer(d) > 0
               = clip(supply(d) / demand(d), floor_d, 1.0)   otherwise

Two interpretive choices this module makes, disclosed here because the dev
doc doesn't pin them precisely enough to implement directly:

1. **Access edges and multiple nearby roads.** §3.3 says access supply is
   "the best available route... access degrades only when all routes
   degrade." A hospital can have more than one road within the pipeline's
   100m buffer (dev doc §2.2), which would make the general
   Pi-over-all-edges product wrong for access specifically — one flooded
   adjacent road would multiplicatively penalize the hospital even with a
   clear road right next to it, contradicting "route around it." So this
   module groups a consumer's access edges together and takes their *max*
   sat() as a single factor in the product, instead of multiplying every
   access edge in individually. Power/water edges (normally exactly one
   supplier per kind, per `06_build_dependency_graph.py`'s nearest-
   neighbor rule) are unaffected — they still each contribute their own
   factor.
2. **Road functional_level.** Roads have no incoming dependency edges of
   their own, so by the general formula their functional_level would just
   equal intrinsic_level — but the flood degradation function (§4.2)
   writes to a road's `blockage` attribute, not its intrinsic_level. This
   module defines `functional_level(road) = intrinsic_level(road) * (1 -
   blockage)` so blockage actually flows into upstream access edges
   through the same general mechanism, rather than a separate special-case
   read of `blockage` wherever access supply is needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from udt.common.models import Asset, AssetType, DependencyEdge
from udt.twin.graph import dependency_edges_of

MAX_FIXED_POINT_ITERATIONS = 20
CONVERGENCE_TOLERANCE = 1e-6


@dataclass
class EdgeRuntimeState:
    """Mutable per-tick state for one edge — `buffer_hours` on
    `DependencyEdge` is the edge's buffer *capacity* (its max); this is
    the current remaining amount, which drains/refills tick by tick."""

    capacity_hours: float
    remaining_hours: float

    @classmethod
    def initial(cls, edge: DependencyEdge) -> EdgeRuntimeState:
        return cls(capacity_hours=edge.buffer_hours, remaining_hours=edge.buffer_hours)


def _road_functional_level(asset: Asset) -> float:
    blockage = float(asset.attributes.get("blockage", 0.0))
    return asset.intrinsic_level * (1.0 - blockage)


def compute_supply(g: nx.DiGraph[str], supplier_id: str) -> float:
    """`supply(d) = functional_level(supplier)` (§3.3), with the road
    special-case from the module docstring."""
    asset = g.nodes[supplier_id]["asset"]
    if asset.asset_type == AssetType.ROAD:
        return _road_functional_level(asset)
    return asset.functional_level  # type: ignore[no-any-return]


def sat(edge: DependencyEdge, supply: float, buffer_remaining: float) -> float:
    """`sat(d)` per §3.3."""
    if buffer_remaining > 0:
        return 1.0
    ratio = supply / edge.demand if edge.demand > 0 else 1.0
    return max(edge.floor, min(1.0, ratio))


def update_buffers(
    g: nx.DiGraph[str],
    edge_states: dict[str, EdgeRuntimeState],
    dt_hours: float,
) -> None:
    """§3.5 step 2: "if supply < demand share -> buffer -= dt; if supply
    restored -> buffer refills at half drain rate." Uses each supplier's
    functional_level as of the *start* of this tick (before this tick's
    fixed-point resolution), matching the algorithm's step ordering."""
    for supplier_id, _consumer_id, data in g.edges(data=True):
        edge: DependencyEdge = data["edge"]
        state = edge_states[edge.edge_id]
        supply = compute_supply(g, supplier_id)
        sufficient = supply >= edge.demand
        if sufficient:
            state.remaining_hours = min(
                state.capacity_hours, state.remaining_hours + dt_hours / 2.0
            )
        else:
            state.remaining_hours = max(0.0, state.remaining_hours - dt_hours)


def resolve_functional_levels(
    g: nx.DiGraph[str],
    edge_states: dict[str, EdgeRuntimeState],
) -> dict[str, float]:
    """§3.5 step 3: fixed-point iteration of functional_level for every
    asset, capped at `MAX_FIXED_POINT_ITERATIONS` — raising (not warning)
    if it doesn't converge, per the dev doc: "that is a bug, not a
    warning." Returns the resolved {asset_id: functional_level} map; does
    not mutate the graph's Asset objects (callers apply the result, e.g.
    the simulator's tick loop, so this function stays a pure computation)."""
    levels: dict[str, float] = {
        asset_id: g.nodes[asset_id]["asset"].functional_level for asset_id in g.nodes
    }

    for _iteration in range(1, MAX_FIXED_POINT_ITERATIONS + 1):
        max_delta = 0.0
        new_levels = dict(levels)

        for asset_id in g.nodes:
            asset: Asset = g.nodes[asset_id]["asset"]
            if asset.asset_type == AssetType.ROAD:
                new_levels[asset_id] = _road_functional_level(asset)
                continue

            deps = dependency_edges_of(g, asset_id)
            if not deps:
                new_levels[asset_id] = asset.intrinsic_level
                continue

            access_edges = [e for e in deps if e.kind == "access"]
            other_edges = [e for e in deps if e.kind != "access"]

            # Note: reads supply from the in-progress `levels` dict (this
            # iteration's working values), not `compute_supply`'s
            # `asset.functional_level` (last tick's stable value) — that
            # distinction is what makes this a fixed-point iteration.
            product = 1.0
            for edge in other_edges:
                supply = (
                    levels[edge.supplier]
                    if g.nodes[edge.supplier]["asset"].asset_type != AssetType.ROAD
                    else _road_functional_level(g.nodes[edge.supplier]["asset"])
                )
                product *= sat(edge, supply, edge_states[edge.edge_id].remaining_hours)

            if access_edges:
                access_sats = [
                    sat(
                        e,
                        _road_functional_level(g.nodes[e.supplier]["asset"]),
                        edge_states[e.edge_id].remaining_hours,
                    )
                    for e in access_edges
                ]
                product *= max(access_sats)  # best available route, see module docstring

            new_levels[asset_id] = asset.intrinsic_level * product

        max_delta = max(abs(new_levels[a] - levels[a]) for a in levels)
        levels = new_levels
        if max_delta <= CONVERGENCE_TOLERANCE:
            return levels

    raise RuntimeError(
        f"Fixed-point iteration did not converge within {MAX_FIXED_POINT_ITERATIONS} "
        "iterations (dev doc §3.5: 'that is a bug, not a warning') — check the "
        "dependency graph for an unstable cycle."
    )
