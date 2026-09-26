#!/usr/bin/env python3
"""Land use + building footprints (feeds `05_flood_susceptibility.py`'s
impervious-surface term — unlike the responder/transit layers, this one
does feed back into the twin's flood model, not just the map/dashboard).

OSM `landuse=*` polygons and `building=*` footprints, fetched from the same
official OSM REST API bbox already used for roads/facilities/drainage — no
new data dependency. Confirmed real and substantial for Kurla during the
2026-09-19 project-review session: 346 residential, 112 commercial, 50
industrial landuse polygons; 8,883 building footprints.

Usage:
  uv run python scripts/03e_extract_landuse.py --config configs/data.yaml
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from shapely.geometry import mapping, shape

from _pipeline_common import (
    configure_logging,
    extract_tagged_polygons,
    fetch_osm_bbox,
    load_config,
    resolve_path,
    write_manifest,
)

# landuse=* values treated as impervious (paved/built-up) for the flood
# formula's land-cover term. Everything else (grass/forest/farmland/
# cemetery/recreation_ground/meadow/greenfield/quarry/...) is pervious.
# Disclosed classification, not an authoritative imperviousness survey.
IMPERVIOUS_LANDUSE_TYPES = {
    "residential",
    "commercial",
    "industrial",
    "retail",
    "garages",
    "construction",
    "civic",
    "military",
    "railway",
}


def _ward_polygon(boundary_geojson_path: str) -> Any:
    with open(boundary_geojson_path) as f:
        d = json.load(f)
    return shape(d["features"][0]["geometry"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()

    log = configure_logging()
    cfg = load_config(args.config)

    boundary_path = resolve_path(cfg["paths"]["processed_dir"]) / "ward_boundary.geojson"
    if not boundary_path.exists():
        raise SystemExit(f"{boundary_path} not found — run 01_get_boundary.py first")
    polygon = _ward_polygon(str(boundary_path))
    west, south, east, north = polygon.bounds

    log.info("fetching_osm_bbox_for_landuse", west=west, south=south, east=east, north=north)
    osm = fetch_osm_bbox(west, south, east, north)
    log.info("fetched_osm_bbox", n_nodes=len(osm.nodes), n_ways=len(osm.ways))

    landuse_polys = extract_tagged_polygons(osm, polygon, "landuse")
    building_polys = extract_tagged_polygons(osm, polygon, "building")
    log.info("extracted_landuse", n_landuse=len(landuse_polys), n_buildings=len(building_polys))

    by_type: dict[str, int] = {}
    features = []
    for p in landuse_polys:
        lu = p["tags"].get("landuse", "unknown")
        by_type[lu] = by_type.get(lu, 0) + 1
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(p["geometry"]),
                "properties": {
                    "osm_id": p["osm_id"],
                    "layer": "landuse",
                    "landuse": lu,
                    "impervious": lu in IMPERVIOUS_LANDUSE_TYPES,
                },
            }
        )
    for p in building_polys:
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(p["geometry"]),
                "properties": {
                    "osm_id": p["osm_id"],
                    "layer": "building",
                    "landuse": None,
                    "impervious": True,
                },
            }
        )
    log.info("landuse_by_type", **by_type)

    feature_collection = {"type": "FeatureCollection", "features": features}

    out_path = resolve_path(cfg["paths"]["processed_dir"]) / "landuse.geojson"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(feature_collection, indent=2))
    log.info("wrote_landuse", path=str(out_path), count=len(features))

    write_manifest(
        out_path,
        inputs={
            "source": "OSM official REST API (api.openstreetmap.org/api/0.6/map)",
            "boundary": boundary_path,
            "impervious_landuse_types": sorted(IMPERVIOUS_LANDUSE_TYPES),
            "bbox": [west, south, east, north],
        },
        row_count=len(features),
        extra={
            "n_landuse_polygons": len(landuse_polys),
            "n_buildings": len(building_polys),
            "by_landuse_type": by_type,
        },
    )


if __name__ == "__main__":
    main()
