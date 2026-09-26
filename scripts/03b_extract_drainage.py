#!/usr/bin/env python3
"""Drainage network (feeds `05_flood_susceptibility.py`, dev doc §2.2 stage 5).

Real drains/streams/rivers/canals from OSM `waterway=*` tags, fetched from
the same official OSM REST API bbox fetch already used for roads/facilities
(see `_pipeline_common.py`'s "OSM via the official REST API" section).

This replaces `05_flood_susceptibility.py`'s earlier DEM-elevation-
percentile stand-in for "distance to drainage" — that stand-in (bottom
decile of elevation cells treated as if they were drainage) was written
when Overpass mirrors were thought unreachable. Confirmed 2026-09-18 that
the official REST API returns real waterway data for free in the same bbox
already fetched for roads: 53 waterway ways intersect the L-ward/Kurla
polygon (drain/stream/river/canal), including the real Mithi River.

Usage:
  uv run python scripts/03b_extract_drainage.py --config configs/data.yaml
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from shapely.geometry import mapping, shape

from _pipeline_common import (
    configure_logging,
    extract_tagged_lines,
    fetch_osm_bbox,
    load_config,
    resolve_path,
    write_manifest,
)

# Linear waterway types that carry water (drainage in the flood-relevant
# sense). Excludes `waterway=dam` — a dam is a barrier structure, not a
# flow path, so it doesn't belong in a "distance to drainage" measure.
DRAINAGE_WATERWAY_TYPES = {"river", "stream", "canal", "drain", "ditch"}


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

    log.info("fetching_osm_bbox_for_drainage", west=west, south=south, east=east, north=north)
    osm = fetch_osm_bbox(west, south, east, north)
    log.info("fetched_osm_bbox", n_nodes=len(osm.nodes), n_ways=len(osm.ways))

    lines = extract_tagged_lines(osm, polygon, "waterway", DRAINAGE_WATERWAY_TYPES)
    log.info("extracted_drainage_lines", count=len(lines))

    by_type: dict[str, int] = {}
    features = []
    for line in lines:
        waterway_type = line["tags"].get("waterway", "unknown")
        by_type[waterway_type] = by_type.get(waterway_type, 0) + 1
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(line["geometry"]),
                "properties": {
                    "osm_id": line["osm_id"],
                    "waterway": waterway_type,
                    "name": line["tags"].get("name", ""),
                },
            }
        )
    log.info("drainage_by_type", **by_type)

    feature_collection = {"type": "FeatureCollection", "features": features}

    out_path = resolve_path(cfg["paths"]["processed_dir"]) / "drainage.geojson"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(feature_collection, indent=2))
    log.info("wrote_drainage", path=str(out_path), count=len(features))

    write_manifest(
        out_path,
        inputs={
            "source": "OSM official REST API (api.openstreetmap.org/api/0.6/map)",
            "boundary": boundary_path,
            "waterway_types": sorted(DRAINAGE_WATERWAY_TYPES),
            "bbox": [west, south, east, north],
        },
        row_count=len(features),
        extra={"by_waterway_type": by_type},
    )


if __name__ == "__main__":
    main()
