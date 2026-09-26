"""Real road-network routing (dev doc §3.3's "recomputed by Dijkstra",
never actually implemented before this; §5.3's ambulance dispatch:
"routing is NOT learned — chosen assignment routes via Dijkstra on
current travel times", module M4).

Deliberately separate from the twin's own dependency graph
(`dependency_graph.json`, ~159 assets) — that graph exists to drive the
cascade formula and was deliberately kept small (see the M2 fix that
replaced the degree-capped road sample with a facility-proximity search
against the full network). Routing needs the *complete* real road
network instead (`roads_full.graphml`, ~12k nodes / ~23k edges), so this
module loads that file directly and is used only for on-demand
shortest-path queries — never fed into the cascade's fixed-point
iteration, so it doesn't reopen the graph-size concern that fix resolved.

**Performance note, disclosed:** each edge's flood *susceptibility*
(static — sampled from the DEM-derived raster) is computed once at load
time and cached directly on the edge's data dict, because only severity/
temporal-envelope change tick to tick, not susceptibility. Re-sampling
the raster for all ~23k edges on every routing query would be needlessly
slow. This keeps a single dispatch decision's routing query cheap (an
O(edges) scalar multiply, then Dijkstra) — it hasn't been load-tested
against an RL training loop's call volume, since no RL agent exists yet
to generate that volume; revisit if/when one does.

**2026-09-26 (dev doc §3.8 / §4.2):** routing now uses the same depth field
as `degradations/flood.py`, including the incident's rainfall footprint and
its own grow/hold/recede durations. Each edge's footprint weight is static
within one incident, so it's computed once per incident (the first
`update_for_tick` call that sees a new `incident_id`) and cached on the edge
as `data["footprint"]`, like `susceptibility`. Caveat: one `RoadNetwork` is
shared across concurrently running API episodes, so two concurrent
incidents would overwrite each other's cache. The API is demo-only; the
training/evaluation harnesses run one incident at a time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from udt.common.models import Incident
from udt.incidents.degradations.flood import (
    MAX_DEPTH_AT_SEVERITY_1_M,
    SusceptibilityRaster,
    footprint_weight,
    incident_envelope,
)

# Same constants `06_build_dependency_graph.py`/`flood.py` already use for
# roads — kept consistent rather than re-deriving a different assumption.
ASSUMED_SPEED_KMPH = 30.0
ROAD_BLOCKAGE_DEPTH_SCALE_M = 0.6
IMPASSABLE_BLOCKAGE = 0.95  # this blocked or worse -> routing treats it as closed


def _edge_depth(scale: float, data: dict[str, Any]) -> float:
    """Flood depth (m) on one edge: the tick's scale x the edge's cached
    susceptibility x this incident's footprint weight."""
    return scale * float(data.get("susceptibility", 0.0)) * float(data.get("footprint", 1.0))


class RoadNetwork:
    """Wraps `roads_full.graphml` for shortest-path queries. Load once per
    episode (`RoadNetwork.load(...)`), reuse for every dispatch decision
    in that run — the per-edge susceptibility cache is what makes reuse
    cheap."""

    def __init__(
        self, graph: Any, node_ids: list[str], node_xy: np.ndarray[Any, np.dtype[np.float64]]
    ) -> None:
        self.graph = graph
        self._node_ids = node_ids
        self._node_xy = node_xy  # shape (n, 2): columns [lon, lat]
        # Set once per tick via `update_for_tick`, read by every
        # `travel_time_minutes` call until the next update — see module
        # docstring's performance note.
        self._current_depth_scale = 0.0
        self._footprint_incident_id: str | None = None

    @classmethod
    def load(cls, path: str | Path, raster: SusceptibilityRaster) -> RoadNetwork:
        g = nx.read_graphml(path)
        for u, _v, data in g.edges(data=True):
            lon, lat = float(g.nodes[u]["x"]), float(g.nodes[u]["y"])
            data["susceptibility"] = raster.value_at(lon, lat)
            data["lon"], data["lat"] = lon, lat
            data["footprint"] = 1.0

        node_ids = list(g.nodes)
        node_xy = np.array([[float(g.nodes[n]["x"]), float(g.nodes[n]["y"])] for n in node_ids])
        return cls(g, node_ids, node_xy)

    def nearest_node(self, lon: float, lat: float) -> str:
        """Nearest node by Euclidean lon/lat distance — a vectorized
        argmin over ~12k nodes, fine without a spatial index at this
        network's size."""
        d2 = (self._node_xy[:, 0] - lon) ** 2 + (self._node_xy[:, 1] - lat) ** 2
        return self._node_ids[int(np.argmin(d2))]

    def update_for_tick(self, incident: Incident, tick: int, dt_minutes: float) -> None:
        """Call once per tick, *before* any `travel_time_minutes` calls
        for that tick — including before the deciding agent's own call,
        which happens outside `Simulator.step` (dev doc §5.3: the
        dispatch *decision* itself routes via Dijkstra, not just its
        enactment). Caches the scalar `severity x temporal_envelope x
        max_depth` term so `travel_time_minutes` doesn't need to know
        about `Incident` at all — keeps this class's public query
        surface to pure graph queries."""
        if incident.incident_id != self._footprint_incident_id:
            for _u, _v, data in self.graph.edges(data=True):
                # Hand-built test graphs have no cached lon/lat: weight 1.
                has_xy = "lon" in data and "lat" in data
                data["footprint"] = (
                    footprint_weight(incident, data["lon"], data["lat"]) if has_xy else 1.0
                )
            self._footprint_incident_id = incident.incident_id
        hours_since_onset = (tick - incident.onset_tick) * dt_minutes / 60.0
        envelope = incident_envelope(incident, hours_since_onset)
        self._current_depth_scale = incident.severity * envelope * MAX_DEPTH_AT_SEVERITY_1_M

    def travel_time_minutes(self, source: str, target: str) -> float | None:
        """Dijkstra shortest travel time (minutes) from `source` to
        `target`, using the depth scale set by the most recent
        `update_for_tick` call. Each edge's current blockage (from the
        *same* flood-depth formula `degradations/flood.py` uses for road
        assets) raises its effective travel time — a flooded road can be
        routed around, exactly as dev doc §3.3 describes. Returns `None`
        if no route exists (fully cut off by blockage or genuine
        disconnection)."""
        scale = self._current_depth_scale

        def weight(u: str, v: str, data: dict[str, Any]) -> float:
            depth = _edge_depth(scale, data)
            blockage = max(0.0, min(1.0, depth / ROAD_BLOCKAGE_DEPTH_SCALE_M))
            if blockage >= IMPASSABLE_BLOCKAGE:
                return float("inf")
            length_m = float(data.get("length", 0.0))
            base_travel_min = length_m / 1000.0 / ASSUMED_SPEED_KMPH * 60.0
            return base_travel_min / max(0.05, 1.0 - blockage)

        try:
            travel_time = float(nx.dijkstra_path_length(self.graph, source, target, weight=weight))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
        # A path can exist topologically but be entirely made of
        # impassable (`inf`-weight) edges — dijkstra_path_length returns
        # `inf` rather than raising in that case, so check explicitly.
        return None if not np.isfinite(travel_time) else travel_time

    def shortest_path_max_depth(self, source: str, target: str) -> float | None:
        """Max flood depth (metres) along the *same* shortest-time route
        `travel_time_minutes` would choose, for module M7's
        `route_flood_safety` constraint (dev doc §8: "route_max_flood_
        depth[a] < 0.4"). Depth per edge uses the same `scale x
        susceptibility` term as `travel_time_minutes`'s own blockage
        calculation — just reported directly in metres rather than
        converted to a blockage fraction, since the constraint's
        threshold (0.4 m) is itself a depth, not a blockage ratio.

        Returns `None` if no route exists — same convention as
        `travel_time_minutes`, and the caller (`constraints/engine.py`)
        treats "no route" as unsafe (fails the constraint) rather than
        vacuously safe."""
        scale = self._current_depth_scale

        def weight(u: str, v: str, data: dict[str, Any]) -> float:
            depth = _edge_depth(scale, data)
            blockage = max(0.0, min(1.0, depth / ROAD_BLOCKAGE_DEPTH_SCALE_M))
            if blockage >= IMPASSABLE_BLOCKAGE:
                return float("inf")
            length_m = float(data.get("length", 0.0))
            base_travel_min = length_m / 1000.0 / ASSUMED_SPEED_KMPH * 60.0
            return base_travel_min / max(0.05, 1.0 - blockage)

        try:
            path = nx.dijkstra_path(self.graph, source, target, weight=weight)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

        max_depth = 0.0
        for u, v in zip(path[:-1], path[1:], strict=True):
            data = self.graph[u][v]
            depth = _edge_depth(scale, data)
            # Same quirk `travel_time_minutes` already guards against:
            # `dijkstra_path` can still return a path built entirely of
            # infinite-weight (impassable) edges when that's the only
            # topological route — not a real route, so treat it the same
            # as "no path" rather than reporting a finite depth for a
            # route nothing could actually drive.
            blockage = max(0.0, min(1.0, depth / ROAD_BLOCKAGE_DEPTH_SCALE_M))
            if blockage >= IMPASSABLE_BLOCKAGE:
                return None
            max_depth = max(max_depth, depth)
        return max_depth
