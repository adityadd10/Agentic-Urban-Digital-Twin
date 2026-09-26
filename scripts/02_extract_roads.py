#!/usr/bin/env python3
"""Road network (dev doc §2.2 stage 2).

Fetches OSM road data directly from the official OSM REST API
(`api.openstreetmap.org`) rather than via `osmnx`/Overpass — see
`_pipeline_common.py`'s "OSM via the official REST API" section for why
(every Overpass mirror tried was unreachable or non-functional on
2026-08-31, confirmed from two independent networks; the official API
stayed up throughout). Builds the drivable road graph locally, clips it to
the ward polygon, then applies the same largest-strongly-connected-
component + segment-cap logic as before.

Usage:
  uv run python scripts/02_extract_roads.py --config configs/data.yaml
"""

from __future__ import annotations

import argparse
import json

import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, shape

from _pipeline_common import (
    build_road_graph,
    configure_logging,
    fetch_osm_bbox,
    load_config,
    resolve_path,
    write_manifest,
)


def _ward_polygon_and_bounds(
    boundary_geojson_path: str,
) -> tuple[object, tuple[float, float, float, float]]:
    with open(boundary_geojson_path) as f:
        d = json.load(f)
    geom = shape(d["features"][0]["geometry"])
    return geom, geom.bounds  # (minx, miny, maxx, maxy) = (west, south, east, north)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()

    log = configure_logging()
    cfg = load_config(args.config)
    roads_cfg = cfg["roads"]

    boundary_path = resolve_path(cfg["paths"]["processed_dir"]) / "ward_boundary.geojson"
    if not boundary_path.exists():
        raise SystemExit(f"{boundary_path} not found — run 01_get_boundary.py first")
    polygon, (west, south, east, north) = _ward_polygon_and_bounds(str(boundary_path))

    log.info("fetching_osm_bbox", west=west, south=south, east=east, north=north)
    osm = fetch_osm_bbox(west, south, east, north)
    log.info("fetched_osm_bbox", n_nodes=len(osm.nodes), n_ways=len(osm.ways))

    full_graph = build_road_graph(osm, polygon)
    log.info(
        "built_road_graph",
        n_nodes=full_graph.number_of_nodes(),
        n_edges=full_graph.number_of_edges(),
    )
    if full_graph.number_of_edges() == 0:
        raise SystemExit(
            "No drivable road edges found in the ward boundary — check the OSM API response "
            "and DRIVABLE_HIGHWAY_TYPES in _pipeline_common.py"
        )

    # Largest strongly-connected component (dev doc §2.2) — a testbed with
    # unreachable islands would make access-edge routing (§3.3) ill-defined.
    largest_scc_nodes = max(nx.strongly_connected_components(full_graph), key=len)
    scc_graph = full_graph.subgraph(largest_scc_nodes).copy()
    log.info(
        "largest_scc", n_nodes=scc_graph.number_of_nodes(), n_edges=scc_graph.number_of_edges()
    )

    processed_dir = resolve_path(cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    full_path = processed_dir / "roads_full.graphml"
    nx.write_graphml(full_graph, full_path)
    log.info("wrote_full_graph", path=str(full_path))

    # Cap to `segment_cap_max` edges for the RL testbed (dev doc §2.2): take
    # the highest-capacity/most-connected edges first (by node degree sum)
    # so the capped graph favors arterial roads, not stray residential
    # fragments, and stays above `segment_cap_min` where the ward supports it.
    edges = list(scc_graph.edges(keys=True, data=True))
    cap = roads_cfg["segment_cap_max"]
    if len(edges) > cap:
        degree = dict(scc_graph.degree())
        edges.sort(key=lambda e: degree[e[0]] + degree[e[1]], reverse=True)
        edges = edges[:cap]
        capped: nx.MultiDiGraph = nx.MultiDiGraph()
        for u, v, k, data in edges:
            capped.add_node(u, **scc_graph.nodes[u])
            capped.add_node(v, **scc_graph.nodes[v])
            capped.add_edge(u, v, key=k, **data)
        testbed_graph = capped
    else:
        testbed_graph = scc_graph
    log.info(
        "capped_testbed_graph",
        n_nodes=testbed_graph.number_of_nodes(),
        n_edges=testbed_graph.number_of_edges(),
        cap=cap,
        min_target=roads_cfg["segment_cap_min"],
    )

    testbed_path = processed_dir / "roads.graphml"
    nx.write_graphml(testbed_graph, testbed_path)

    rows = []
    for i, (u, v, _k, data) in enumerate(testbed_graph.edges(keys=True, data=True)):
        nu, nv = testbed_graph.nodes[u], testbed_graph.nodes[v]
        rows.append(
            {
                "asset_id": f"R{i}",
                "u": u,
                "v": v,
                "osmid": data.get("osmid"),
                "highway": data.get("highway"),
                "length_m": data.get("length"),
                "geometry": LineString([(nu["x"], nu["y"]), (nv["x"], nv["y"])]),
            }
        )
    gdf_edges = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    geojson_path = processed_dir / "roads.geojson"
    gdf_edges.to_file(geojson_path, driver="GeoJSON")
    log.info("wrote_testbed_graph", graphml=str(testbed_path), geojson=str(geojson_path))

    write_manifest(
        testbed_path,
        inputs={
            "source": "OSM official REST API (api.openstreetmap.org/api/0.6/map)",
            "boundary": boundary_path,
            "bbox": [west, south, east, north],
        },
        row_count=testbed_graph.number_of_edges(),
        extra={
            "osm_nodes_fetched": len(osm.nodes),
            "osm_ways_fetched": len(osm.ways),
            "full_graph_nodes": full_graph.number_of_nodes(),
            "full_graph_edges": full_graph.number_of_edges(),
            "scc_nodes": scc_graph.number_of_nodes(),
            "scc_edges": scc_graph.number_of_edges(),
            "testbed_nodes": testbed_graph.number_of_nodes(),
            "testbed_edges": testbed_graph.number_of_edges(),
        },
    )


if __name__ == "__main__":
    main()
