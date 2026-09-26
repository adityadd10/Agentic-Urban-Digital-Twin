#!/usr/bin/env python3
"""Rail/metro transit infrastructure — reference/context layer.

**Data-only by design, not part of the cascade dependency graph** — same
2026-09-19 decision as `03c_extract_responders.py` (see its docstring).

BMC ArcGIS layers (`Existing_Suburban_Stations`/`_Line`, `Metro_Stations`/
`Metro_Lines`) have no ward attribute, unlike `Fire_Station`/
`Police_Stations`, so filtering here is spatial (clip to the ward polygon),
the same approach `02_extract_roads.py` uses for OSM. Confirmed reachable
and populated for Kurla during the 2026-09-19 project-review session: 6
suburban stations, 8 suburban line segments, 6 metro stations, 1 metro line
intersect the ward polygon — Kurla is a real rail junction.

Usage:
  uv run python scripts/03d_extract_transit.py --config configs/data.yaml
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from shapely.geometry import shape

from _pipeline_common import (
    BmcArcGisClient,
    configure_logging,
    load_config,
    resolve_path,
    write_manifest,
)


def _ward_polygon(boundary_geojson_path: str) -> Any:
    with open(boundary_geojson_path) as f:
        d = json.load(f)
    return shape(d["features"][0]["geometry"])


def _clip_features(
    geojson: dict[str, Any], polygon: Any, asset_type: str, prefix: str, name_field: str
) -> list[dict[str, Any]]:
    features = []
    i = 0
    for feat in geojson.get("features", []):
        geom = shape(feat["geometry"])
        if not geom.intersects(polygon):
            continue
        props = feat.get("properties", {})
        name = props.get(name_field) or f"{asset_type}_{i}"
        features.append(
            {
                "type": "Feature",
                "geometry": feat["geometry"],
                "properties": {
                    "asset_id": f"{prefix}{i}",
                    "asset_type": asset_type,
                    "name": name,
                    "source": "bmc_arcgis",
                },
            }
        )
        i += 1
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()

    log = configure_logging()
    cfg = load_config(args.config)
    bmc_cfg = cfg["bmc_arcgis"]

    boundary_path = resolve_path(cfg["paths"]["processed_dir"]) / "ward_boundary.geojson"
    if not boundary_path.exists():
        raise SystemExit(f"{boundary_path} not found — run 01_get_boundary.py first")
    polygon = _ward_polygon(str(boundary_path))

    with BmcArcGisClient(bmc_cfg["base_url"]) as bmc:
        suburban_stations = bmc.query_geojson("Existing_Suburban_Stations", 0)
        suburban_lines = bmc.query_geojson("Existing_Suburban_Line", 0)
        metro_stations = bmc.query_geojson("Metro_Stations", 0)
        metro_lines = bmc.query_geojson("Metro_Lines", 0)

    features = []
    features += _clip_features(suburban_stations, polygon, "suburban_station", "RS", "NAME")
    features += _clip_features(suburban_lines, polygon, "suburban_line", "RL", "NAME")
    features += _clip_features(metro_stations, polygon, "metro_station", "MS", "Name")
    features += _clip_features(metro_lines, polygon, "metro_line", "ML", "Name")
    log.info("clipped_transit_features", count=len(features))

    feature_collection = {"type": "FeatureCollection", "features": features}

    out_path = resolve_path(cfg["paths"]["processed_dir"]) / "transit.geojson"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(feature_collection, indent=2))
    log.info("wrote_transit", path=str(out_path), count=len(features))

    by_type = {
        t: sum(1 for feat in features if feat["properties"]["asset_type"] == t)
        for t in ("suburban_station", "suburban_line", "metro_station", "metro_line")
    }
    write_manifest(
        out_path,
        inputs={
            "source": "BMC ArcGIS REST (Existing_Suburban_Stations/_Line, Metro_Stations/_Lines)",
            "boundary": boundary_path,
            "filter": "spatial intersect with ward polygon (no ward attribute on these layers)",
        },
        row_count=len(features),
        extra={"by_type": by_type},
    )


if __name__ == "__main__":
    main()
