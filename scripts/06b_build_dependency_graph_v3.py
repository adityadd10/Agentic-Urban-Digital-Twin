#!/usr/bin/env python3
"""Twin-v3 dependency graph (dev doc §3.9, layout approved 2026-09-29).

Starts from the v2 graph (`dependency_graph.json`, left unchanged) so the two
real hospitals, S0 and the two synthetic pumps keep their exact attributes,
then:

- adds H3 (`H_synth_0`, synthetic), its pump `W_synth_2` (synthetic), S1
  (Chhedanagar Sub Station, Reliance Energy: real OSM node 2242242001, just
  outside the ward) and S2 (`S_synth_0`, synthetic);
- rewires supply so each hospital has its own substation, plus one deliberate
  cross-link (S0 also powers H1's pump `W_synth_0`);
- gives every new facility road-access edges by the same rule as
  `06_build_dependency_graph.py` (real roads within `facility_road_buffer_m`).

New assets are built by `06_build_dependency_graph.py`'s own builders (same
attribute ranges). They draw from a separate seeded stream (`seed + 3`), so
v2's draws are untouched. Writes `dependency_graph_v3.json` + manifest.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, Point

from _pipeline_common import configure_logging, load_config, resolve_path, write_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from udt.common.models import DependencyEdge, DependencyGraph  # noqa: E402
from udt.common.seeding import make_rng  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "build_v2", Path(__file__).with_name("06_build_dependency_graph.py")
)
assert _spec is not None and _spec.loader is not None
v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v2)

UTM_CRS = v2.UTM_CRS
PUMP_OFFSET = (0.0005, 0.00049)  # same offset W_synth_0/1 have from their hospitals

H1, H2 = "H_peripheral_0", "H_polyclinic_0"
H3, W0, W1, W2 = "H_synth_0", "W_synth_0", "W_synth_1", "W_synth_2"
S0, S1, S2 = "S0", "S1", "S_synth_0"
NEW_POINTS = {
    H3: (72.89224, 19.06807),
    S1: (72.90612, 19.06863),
    S2: (72.87265, 19.06100),
}
PROVENANCE = {
    H3: "synthetic_twin_v3_H3 (dev doc 3.9): location chosen for median flood exposure",
    W2: "synthetic_twin_v3_pump_for_H3 (dev doc 3.9)",
    S1: "real_osm_node_2242242001 Chhedanagar Sub Station (Reliance Energy); "
    "attributes are placeholders like S0's",
    S2: "synthetic_twin_v3_S2 (dev doc 3.9): no distribution substation mapped near H1",
}
# (supplier, consumer): one entry per power/water edge in twin-v3
POWER_EDGES = [(S2, H1), (S0, H2), (S1, H3), (S0, W1), (S1, W2), (S0, W0)]
WATER_EDGES = [(W0, H1), (W1, H2), (W2, H3)]


def _point(lon: float, lat: float) -> dict[str, Any]:
    return {"type": "Point", "coordinates": [lon, lat]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    log = configure_logging()
    cfg = load_config(args.config)
    processed = resolve_path(cfg["paths"]["processed_dir"])
    out_path = processed / "dependency_graph_v3.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; twin-v3 inputs are frozen once written")

    base = DependencyGraph.model_validate_json((processed / "dependency_graph.json").read_text())
    rng = make_rng(cfg["seed"] + 3)

    # Keep every v2 asset except the roads (re-derived below for all facilities).
    assets = {a.asset_id: a for a in base.assets if a.asset_type.value != "road"}
    assets[H3] = v2._build_hospital_asset(rng, H3, _point(*NEW_POINTS[H3]))
    assets[S1] = v2._build_substation_asset(rng, S1, _point(*NEW_POINTS[S1]))
    assets[S2] = v2._build_substation_asset(rng, S2, _point(*NEW_POINTS[S2]))
    lon, lat = NEW_POINTS[H3]
    assets[W2] = v2._build_water_asset(
        rng, W2, _point(lon + PUMP_OFFSET[0], lat + PUMP_OFFSET[1]), provenance=PROVENANCE[W2]
    )
    for asset_id in (H3, S1, S2):
        assets[asset_id].attributes["provenance"] = PROVENANCE[asset_id]

    edges: list[DependencyEdge] = []
    for supplier, consumer in POWER_EDGES:
        c = assets[consumer]
        if c.asset_type.value == "hospital":
            demand, crit, buf, floor = v2.HOSPITAL_POWER_SHARE_DEFAULT, 0.9, 8.0, 0.3
        else:  # pump: its power as a share of the supplying substation's capacity
            demand = c.attributes["pump_power_mw"] / assets[supplier].attributes["capacity_mw"]
            crit, buf, floor = 0.9, 0.0, 0.0
        edges.append(
            DependencyEdge(
                edge_id=f"V3_{supplier}_{consumer}_power",
                supplier=supplier,
                consumer=consumer,
                kind="power",
                demand=demand,
                criticality=crit,
                buffer_hours=buf,
                floor=floor,
            )
        )
    for supplier, consumer in WATER_EDGES:
        edges.append(
            DependencyEdge(
                edge_id=f"V3_{supplier}_{consumer}_water",
                supplier=supplier,
                consumer=consumer,
                kind="water",
                demand=v2.HOSPITAL_WATER_SHARE_DEFAULT,
                criticality=0.7,
                buffer_hours=v2.HOSPITAL_WATER_RESERVE_HOURS,
                floor=0.5,
            )
        )

    # Access edges: same rule and parameters as 06_build_dependency_graph.py.
    g = nx.read_graphml(processed / "roads_full.graphml")
    roads = gpd.GeoDataFrame(
        [
            {
                "asset_id": f"RF_{u}_{v}",
                "geometry": LineString(
                    [
                        (float(g.nodes[u]["x"]), float(g.nodes[u]["y"])),
                        (float(g.nodes[v]["x"]), float(g.nodes[v]["y"])),
                    ]
                ),
            }
            for u, v in g.edges()
        ],
        crs="EPSG:4326",
    ).to_crs(UTM_CRS)
    buffer_m = cfg["dependency_rules"]["facility_road_buffer_m"]
    road_assets: dict[str, Any] = {}
    facility_ids = [
        a for a in assets if assets[a].asset_type.value in ("hospital", "substation", "water")
    ]
    for fid in facility_ids:
        lon, lat = assets[fid].geometry["coordinates"][:2]
        p = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(UTM_CRS).iloc[0]
        nearby = roads[roads.geometry.distance(p) <= buffer_m]
        if nearby.empty:
            nearby = roads.loc[[roads.geometry.distance(p).idxmin()]]
            log.warning("facility_beyond_road_buffer_using_nearest", facility=fid)
        floor = 0.7 if assets[fid].asset_type.value == "hospital" else 0.5
        for _, r in nearby.iterrows():
            rid = str(r["asset_id"])
            if rid not in road_assets:
                geom = json.loads(gpd.GeoSeries([r.geometry], crs=UTM_CRS).to_crs(4326).to_json())[
                    "features"
                ][0]["geometry"]
                length_m = float(r.geometry.length)
                road_assets[rid] = v2._build_road_asset(
                    rid, geom, length_m, length_m / 1000 / 30 * 60
                )
            edges.append(
                DependencyEdge(
                    edge_id=f"V3_{rid}_{fid}_access",
                    supplier=rid,
                    consumer=fid,
                    kind="access",
                    demand=1.0,
                    criticality=0.6,
                    buffer_hours=0.0,
                    floor=floor,
                )
            )

    graph = DependencyGraph(
        version="v3", assets=[*assets.values(), *road_assets.values()], edges=edges
    )
    out_path.write_text(graph.model_dump_json(indent=2))
    counts = {
        t: sum(a.asset_type.value == t for a in graph.assets)
        for t in ("hospital", "substation", "water", "road")
    }
    log.info("wrote_dependency_graph_v3", path=str(out_path), n_edges=len(edges), **counts)
    write_manifest(
        out_path,
        inputs={
            "dependency_graph_v2": processed / "dependency_graph.json",
            "roads": processed / "roads_full.graphml",
        },
        row_count=len(graph.assets),
        extra={"n_assets": len(graph.assets), "n_edges": len(edges), **counts},
    )


if __name__ == "__main__":
    main()
