#!/usr/bin/env python3
"""Dependency graph (dev doc §2.2 stage 6, schema §3.4).

Builds `dependency_graph.json` from `facilities.geojson` (stage 3) and
`roads_full.graphml` (stage 2, the complete uncapped network) using the
dev doc's edge-assignment rules:
hospital/water <- nearest substation (power); hospital <- nearest water
(water); facility <- road segments within `facility_road_buffer_m`
(access). Water facilities also get a power edge from their nearest
substation (the substation -> water -> hospital cascade chain, §2.2).

**Access roads sourced from the complete real network, not the degree cap
(revised after the advisor's "use open data properly, no ArcGIS" call):**
`02_extract_roads.py`'s degree-based segment cap (top ~100 of ~20,000+
roads by connectivity) has no idea where facilities are — running it
against real data left both Kurla hospitals with zero roads in the capped
set within `facility_road_buffer_m`, silently dropping the flood-blocks-
access mechanic for exactly the assets it matters most for. Rather than
patch this with a fallback for just the facilities the cap happened to
miss, this module now searches `roads_full.graphml` (the complete,
uncapped, real OSM network `02_extract_roads.py` also writes) directly for
**every** facility's access roads — the degree cap and `roads.geojson` are
no longer consulted here at all. Road assets are added lazily, only when
a road is actually within buffer distance of a real facility, so the
graph stays small without needing an arbitrary connectivity-based cap.

Prototype-1 default attribute/edge values (not otherwise pinned by the dev
doc beyond ranges/examples — disclosed modeling assumptions, same spirit
as the dev doc's own "simplified by design" flood raster):
  - hospital: beds_total ~ Uniform(60,300); icu_total = 10% of beds;
    backup_gen_hours=8.0; water_reserve_hours=6.0 (dev doc §3.2 exactly)
  - substation: capacity_mw ~ Uniform(20,50); shed_tier=0 (dev doc §3.2)
  - water: output_lps ~ Uniform(50,200); reservoir_hours=4.0 (dev doc
    §3.2); pump_power_mw ~ Uniform(0.5,2.0)
  - power->hospital edge: criticality=0.9, buffer_hours=8.0 (=
    backup_gen_hours), floor=0.3 (dev doc §3.3 exactly)
  - water->hospital edge: criticality=0.7, buffer_hours=6.0 (=
    water_reserve_hours), floor=0.5 (dev doc §3.3 exactly)
  - power->water edge: criticality=0.9, buffer_hours=0.0 ("pumps just
    stop" per dev doc §3.3), floor=0.0 (dev doc §3.3 exactly)
  - access->facility edge: criticality=0.6, buffer_hours=0.0 (route
    quality is recomputed live via Dijkstra, not buffered, per §3.3),
    floor=0.7 for hospitals (dev doc §3.3 exactly), 0.5 for other types
    (not specified in the dev doc — same conservative pattern)
`demand` resolution: dev doc §3.3's formula needs `supply(d)/demand(d)` to be
dimensionless (`supply(d) = functional_level(supplier)` is already a 0..1
fraction), so `demand` here is each edge's **fractional share of the
supplier's capacity**, not the raw physical quantity §3.4's schema comment
illustrates ("demand: 2.5" — that example isn't dimensionally consistent
with §3.3's own formula, so this is the resolution used here, disclosed as
such). Power->water is computed exactly (`pump_power_mw / capacity_mw`);
power/water->hospital use a flat 0.1 default share (no per-hospital load
estimate exists yet to compute it properly); access edges use demand=1.0
since route-quality supply is already 0..1 (dev doc §3.3's own access
description), so `sat = clip(supply, floor, 1.0)` directly.

Usage:
  uv run python scripts/06_build_dependency_graph.py --config configs/data.yaml
  (requires facilities.geojson and roads_full.graphml, both from 02/03)
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import geopandas as gpd

from _pipeline_common import configure_logging, load_config, resolve_path, write_manifest

sys_path_added = False


def _load_udt_models() -> Any:
    """Import udt.common.models with `src/` on sys.path (scripts aren't
    inside the installed package, but the repo's src layout is importable
    once `src/` is added — matches how `uv run` resolves the editable
    install; this is a fallback for running the script standalone)."""
    import sys
    from pathlib import Path

    global sys_path_added
    if not sys_path_added:
        src_dir = Path(__file__).resolve().parents[1] / "src"
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        sys_path_added = True
    from udt.common import models

    return models


UTM_CRS = "EPSG:32643"  # UTM 43N — metric CRS covering Mumbai, for nearest-neighbor distances

HOSPITAL_BEDS_RANGE = (60, 300)
HOSPITAL_BACKUP_GEN_HOURS = 8.0
HOSPITAL_WATER_RESERVE_HOURS = 6.0
HOSPITAL_POWER_SHARE_DEFAULT = 0.1  # fraction of substation capacity, see module docstring
HOSPITAL_WATER_SHARE_DEFAULT = 0.1
SUBSTATION_CAPACITY_MW_RANGE = (20.0, 50.0)
WATER_OUTPUT_LPS_RANGE = (50.0, 200.0)
WATER_RESERVOIR_HOURS = 4.0
WATER_PUMP_POWER_MW_RANGE = (0.5, 2.0)


def _build_hospital_asset(rng: Any, asset_id: str, geometry: dict[str, Any]) -> Any:
    models = _load_udt_models()
    beds_total = int(rng.integers(*HOSPITAL_BEDS_RANGE))
    icu_total = max(1, int(round(0.10 * beds_total)))
    return models.Asset(
        asset_id=asset_id,
        asset_type=models.AssetType.HOSPITAL,
        geometry=geometry,
        attributes={
            "beds_total": beds_total,
            "beds_occupied": int(rng.integers(int(0.6 * beds_total), int(0.9 * beds_total) + 1)),
            "icu_total": icu_total,
            "icu_occupied": int(rng.integers(0, icu_total + 1)),
            "backup_gen_hours": HOSPITAL_BACKUP_GEN_HOURS,
            "water_reserve_hours": HOSPITAL_WATER_RESERVE_HOURS,
            "patient_queue": 0,
        },
    )


def _build_substation_asset(rng: Any, asset_id: str, geometry: dict[str, Any]) -> Any:
    models = _load_udt_models()
    capacity_mw = float(rng.uniform(*SUBSTATION_CAPACITY_MW_RANGE))
    return models.Asset(
        asset_id=asset_id,
        asset_type=models.AssetType.SUBSTATION,
        geometry=geometry,
        attributes={
            "capacity_mw": capacity_mw,
            "load_mw": float(capacity_mw * rng.uniform(0.4, 0.7)),
            "shed_tier": 0,
        },
    )


def _build_water_asset(
    rng: Any, asset_id: str, geometry: dict[str, Any], provenance: str | None = None
) -> Any:
    models = _load_udt_models()
    pump_power_mw = float(rng.uniform(*WATER_PUMP_POWER_MW_RANGE))
    attributes: dict[str, Any] = {
        "output_lps": float(rng.uniform(*WATER_OUTPUT_LPS_RANGE)),
        "reservoir_hours": WATER_RESERVOIR_HOURS,
        "pump_power_mw": pump_power_mw,
    }
    if provenance:
        # Threaded through from data/manual_facilities.yaml's own
        # `provenance` field (see that file's own header comment) so a
        # fabricated asset stays traceable as such inside the twin state
        # itself, not just in the pipeline's source data — a decision-log
        # reader or the frontend can surface this, not just someone
        # reading the YAML. No other asset type carries this key yet
        # (none needed it before this one, real-but-placeholder-valued
        # facility).
        attributes["provenance"] = provenance
    return models.Asset(
        asset_id=asset_id,
        asset_type=models.AssetType.WATER,
        geometry=geometry,
        attributes=attributes,
    )


def _build_road_asset(
    asset_id: str, geometry: dict[str, Any], length_m: float, base_travel_min: float
) -> Any:
    models = _load_udt_models()
    return models.Asset(
        asset_id=asset_id,
        asset_type=models.AssetType.ROAD,
        geometry=geometry,
        attributes={
            "length_m": length_m,
            "base_travel_min": base_travel_min,
            "blockage": 0.0,
            "flood_depth_m": 0.0,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()

    log = configure_logging()
    cfg = load_config(args.config)
    processed_dir = resolve_path(cfg["paths"]["processed_dir"])

    facilities_path = processed_dir / "facilities.geojson"
    roads_full_path = processed_dir / "roads_full.graphml"
    for p in (facilities_path, roads_full_path):
        if not p.exists():
            raise SystemExit(
                f"{p} not found — run 03_extract_facilities.py / 02_extract_roads.py first"
            )

    from udt.common.seeding import make_rng

    models = _load_udt_models()
    rng = make_rng(cfg["seed"])

    facilities = gpd.read_file(facilities_path).to_crs(UTM_CRS)

    hospitals = facilities[facilities["asset_type"] == "hospital"].reset_index(drop=True)
    substations = facilities[facilities["asset_type"] == "substation"].reset_index(drop=True)
    water_facilities = facilities[facilities["asset_type"] == "water"].reset_index(drop=True)

    if substations.empty:
        raise SystemExit(
            "No substations in facilities.geojson — the power->hospital/power->water cascade "
            "chain has nothing to attach to. Add entries to data/manual_facilities.yaml or "
            "re-run 03_extract_facilities.py with OSM access."
        )

    assets: list[Any] = []
    edges: list[Any] = []

    for _, row in hospitals.iterrows():
        assets.append(
            _build_hospital_asset(
                rng,
                row["asset_id"],
                json.loads(gpd.GeoSeries([row.geometry], crs=UTM_CRS).to_crs(4326).to_json())[
                    "features"
                ][0]["geometry"],
            )
        )
    for _, row in substations.iterrows():
        assets.append(
            _build_substation_asset(
                rng,
                row["asset_id"],
                json.loads(gpd.GeoSeries([row.geometry], crs=UTM_CRS).to_crs(4326).to_json())[
                    "features"
                ][0]["geometry"],
            )
        )
    for _, row in water_facilities.iterrows():
        assets.append(
            _build_water_asset(
                rng,
                row["asset_id"],
                json.loads(gpd.GeoSeries([row.geometry], crs=UTM_CRS).to_crs(4326).to_json())[
                    "features"
                ][0]["geometry"],
                provenance=row.get("provenance"),
            )
        )

    def nearest(row_geom: Any, candidates: gpd.GeoDataFrame) -> Any:
        dists = candidates.geometry.distance(row_geom)
        return candidates.loc[dists.idxmin()]

    edge_counter = 0

    def new_edge_id(kind: str) -> str:
        nonlocal edge_counter
        edge_counter += 1
        return f"E{edge_counter}_{kind}"

    # hospital <- nearest substation (power), hospital <- nearest water (water)
    for _, h in hospitals.iterrows():
        s = nearest(h.geometry, substations)
        edges.append(
            models.DependencyEdge(
                edge_id=new_edge_id("power"),
                supplier=s["asset_id"],
                consumer=h["asset_id"],
                kind="power",
                demand=HOSPITAL_POWER_SHARE_DEFAULT,
                criticality=0.9,
                buffer_hours=HOSPITAL_BACKUP_GEN_HOURS,
                floor=0.3,
            )
        )
        if not water_facilities.empty:
            w = nearest(h.geometry, water_facilities)
            edges.append(
                models.DependencyEdge(
                    edge_id=new_edge_id("water"),
                    supplier=w["asset_id"],
                    consumer=h["asset_id"],
                    kind="water",
                    demand=HOSPITAL_WATER_SHARE_DEFAULT,
                    criticality=0.7,
                    buffer_hours=HOSPITAL_WATER_RESERVE_HOURS,
                    floor=0.5,
                )
            )

    # water <- nearest substation (power) — the substation -> water -> hospital chain
    for _, w in water_facilities.iterrows():
        s = nearest(w.geometry, substations)
        pump_power_mw = float(
            next(a.attributes["pump_power_mw"] for a in assets if a.asset_id == w["asset_id"])
        )
        capacity_mw = float(
            next(a.attributes["capacity_mw"] for a in assets if a.asset_id == s["asset_id"])
        )
        edges.append(
            models.DependencyEdge(
                edge_id=new_edge_id("power"),
                supplier=s["asset_id"],
                consumer=w["asset_id"],
                kind="power",
                demand=pump_power_mw / capacity_mw,
                criticality=0.9,
                buffer_hours=0.0,
                floor=0.0,
            )
        )

    # facility <- road segments within buffer (access). Sourced directly
    # from the complete, real, uncapped OSM network (roads_full.graphml) —
    # see module docstring for why this replaced the degree-capped
    # roads.geojson + single-nearest-road-fallback approach.
    buffer_m = cfg["dependency_rules"]["facility_road_buffer_m"]
    all_facility_rows = (
        list(hospitals.iterrows())
        + list(substations.iterrows())
        + list(water_facilities.iterrows())
    )

    def _load_full_roads() -> gpd.GeoDataFrame:
        import networkx as nx
        from shapely.geometry import LineString

        g = nx.read_graphml(roads_full_path)
        rows = []
        for u, v in g.edges():
            nu, nv = g.nodes[u], g.nodes[v]
            rows.append(
                {
                    "asset_id": f"RF_{u}_{v}",
                    "geometry": LineString(
                        [(float(nu["x"]), float(nu["y"])), (float(nv["x"]), float(nv["y"]))]
                    ),
                }
            )
        return gpd.GeoDataFrame(rows, crs="EPSG:4326").to_crs(UTM_CRS)

    full_roads_gdf = _load_full_roads()
    road_ids_added: set[str] = set()

    for _, f in all_facility_rows:
        nearby = full_roads_gdf[full_roads_gdf.geometry.distance(f.geometry) <= buffer_m]
        floor = 0.7 if f["asset_type"] == "hospital" else 0.5

        if nearby.empty:
            # Even the complete real network has nothing within the
            # buffer — a genuinely isolated facility, not a sampling
            # artifact. Widen to the single nearest road so the facility
            # still has an access edge (the cascade math needs at least
            # one to mean anything), and log it since it's worth knowing.
            dists = full_roads_gdf.geometry.distance(f.geometry)
            nearby = full_roads_gdf.loc[[dists.idxmin()]]
            log.warning(
                "facility_beyond_road_buffer_using_nearest",
                facility=f["asset_id"],
                distance_m=round(float(dists.min()), 1),
                buffer_m=buffer_m,
            )

        for _, r in nearby.iterrows():
            rid = str(r["asset_id"])
            if rid not in road_ids_added:
                geom4326 = json.loads(
                    gpd.GeoSeries([r.geometry], crs=UTM_CRS).to_crs(4326).to_json()
                )["features"][0]["geometry"]
                length_m = float(r.geometry.length)
                assets.append(_build_road_asset(rid, geom4326, length_m, length_m / 1000 / 30 * 60))
                road_ids_added.add(rid)
            edges.append(
                models.DependencyEdge(
                    edge_id=new_edge_id("access"),
                    supplier=rid,
                    consumer=f["asset_id"],
                    kind="access",
                    demand=1.0,
                    criticality=0.6,
                    buffer_hours=0.0,
                    floor=floor,
                )
            )

    graph = models.DependencyGraph(assets=assets, edges=edges)

    out_path = processed_dir / "dependency_graph.json"
    out_path.write_text(graph.model_dump_json(indent=2))
    log.info(
        "wrote_dependency_graph",
        path=str(out_path),
        n_assets=len(assets),
        n_edges=len(edges),
        n_hospitals=len(hospitals),
        n_substations=len(substations),
        n_water=len(water_facilities),
        n_roads=len(road_ids_added),
    )

    write_manifest(
        out_path,
        inputs={"facilities": facilities_path, "roads": roads_full_path},
        row_count=len(assets),
        extra={"n_assets": len(assets), "n_edges": len(edges)},
    )


if __name__ == "__main__":
    main()
