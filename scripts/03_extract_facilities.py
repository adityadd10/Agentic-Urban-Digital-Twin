#!/usr/bin/env python3
"""Facilities (dev doc §2.2 stage 3).

Hospitals: BMC's `Health_Facilities` ArcGIS REST service (Main, Special,
Peripheral, Polyclinic layers — real, typed, named), filtered to the
chosen ward. Substations and water facilities: OSM tags, fetched from the
official OSM REST API (`api.openstreetmap.org`) rather than
`osmnx`/Overpass — see `_pipeline_common.py`'s "OSM via the official REST
API" section for why. BMC has no public power/water-utility layer
(confirmed; see MTP_Development_Document.md §2), so OSM stays the source
for those, same as the dev doc originally planned.
`data/manual_facilities.yaml` overrides/adds entries for either source
(flagged `source: manual`), same mechanism the dev doc specifies.

Usage:
  uv run python scripts/03_extract_facilities.py --config configs/data.yaml --ward L
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import yaml
from shapely.geometry import shape

from _pipeline_common import (
    BmcArcGisClient,
    configure_logging,
    extract_tagged_points,
    fetch_osm_bbox,
    load_config,
    resolve_path,
    write_manifest,
)


def _ward_polygon_and_bounds(
    boundary_geojson_path: str,
) -> tuple[Any, tuple[float, float, float, float]]:
    with open(boundary_geojson_path) as f:
        d = json.load(f)
    geom = shape(d["features"][0]["geometry"])
    return geom, geom.bounds


def _fetch_hospitals(
    bmc: BmcArcGisClient, cfg: dict[str, Any], ward_code: str, log: Any
) -> list[dict[str, Any]]:
    bmc_cfg = cfg["bmc_arcgis"]
    facilities: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for layer in bmc_cfg["hospital_layers"]:
        geojson = bmc.query_geojson(
            bmc_cfg["health_service"],
            layer["id"],
            where=f"{bmc_cfg['ward_field']}='{ward_code}'",
        )
        for i, feat in enumerate(geojson.get("features", [])):
            props = feat.get("properties", {})
            name = props.get(bmc_cfg["name_field"]) or f"{layer['hospital_type']}_{i}"
            asset_id = f"H_{layer['hospital_type']}_{i}"
            if asset_id in seen_ids:
                continue
            seen_ids.add(asset_id)
            facilities.append(
                {
                    "asset_id": asset_id,
                    "asset_type": "hospital",
                    "name": name,
                    "hospital_type": layer["hospital_type"],
                    "geometry": feat["geometry"],
                    "source": "bmc_arcgis",
                }
            )
    log.info("fetched_hospitals", count=len(facilities))
    return facilities


def _points_to_facilities(
    points: list[dict[str, Any]], asset_type: str, prefix: str
) -> list[dict[str, Any]]:
    facilities = []
    for i, p in enumerate(points):
        name = p["tags"].get("name", f"{asset_type}_{i}")
        facilities.append(
            {
                "asset_id": f"{prefix}{i}",
                "asset_type": asset_type,
                "name": name,
                "geometry": {"type": "Point", "coordinates": [p["lon"], p["lat"]]},
                "source": "osm",
                "osm_id": p["osm_id"],
                "osm_type": p["osm_type"],
            }
        )
    return facilities


def _load_manual_overrides(path: str) -> list[dict[str, Any]]:
    p = resolve_path(path)
    if not p.exists():
        return []
    with p.open() as f:
        data = yaml.safe_load(f) or {}
    entries = data.get("facilities", [])
    for e in entries:
        e["source"] = "manual"
    return entries  # type: ignore[no-any-return]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--ward", required=True)
    args = parser.parse_args()

    log = configure_logging()
    cfg = load_config(args.config)

    boundary_path = resolve_path(cfg["paths"]["processed_dir"]) / "ward_boundary.geojson"
    if not boundary_path.exists():
        raise SystemExit(f"{boundary_path} not found — run 01_get_boundary.py first")
    polygon, (west, south, east, north) = _ward_polygon_and_bounds(str(boundary_path))

    with BmcArcGisClient(cfg["bmc_arcgis"]["base_url"]) as bmc:
        hospitals = _fetch_hospitals(bmc, cfg, args.ward, log)

    log.info("fetching_osm_bbox_for_facilities", west=west, south=south, east=east, north=north)
    osm = fetch_osm_bbox(west, south, east, north)
    log.info("fetched_osm_bbox", n_nodes=len(osm.nodes), n_ways=len(osm.ways))

    substation_points = extract_tagged_points(osm, polygon, "power", {"substation"})
    water_points = extract_tagged_points(
        osm, polygon, "man_made", {"water_works", "water_tower", "pumping_station"}
    )
    substations = _points_to_facilities(substation_points, "substation", "S")
    water = _points_to_facilities(water_points, "water", "W")
    log.info("fetched_osm_substations", count=len(substations))
    log.info("fetched_osm_water", count=len(water))

    manual = _load_manual_overrides(cfg["paths"]["manual_facilities"])
    if manual:
        log.info("loaded_manual_overrides", count=len(manual))

    all_facilities = hospitals + substations + water + manual
    feature_collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": f["geometry"],
                "properties": {k: v for k, v in f.items() if k != "geometry"},
            }
            for f in all_facilities
        ],
    }

    out_path = resolve_path(cfg["paths"]["processed_dir"]) / "facilities.geojson"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(feature_collection, indent=2))
    log.info(
        "wrote_facilities",
        path=str(out_path),
        n_hospitals=len(hospitals),
        n_substations=len(substations),
        n_water=len(water),
        n_manual=len(manual),
    )

    write_manifest(
        out_path,
        inputs={
            "hospitals_source": "BMC ArcGIS REST (Health_Facilities)",
            "substations_source": "OSM official REST API, power=substation",
            "water_source": "OSM official API, man_made=water_works|water_tower|pumping_station",
            "manual_overrides": cfg["paths"]["manual_facilities"],
        },
        row_count=len(all_facilities),
        extra={
            "n_hospitals": len(hospitals),
            "n_substations": len(substations),
            "n_water": len(water),
            "n_manual": len(manual),
        },
    )


if __name__ == "__main__":
    main()
