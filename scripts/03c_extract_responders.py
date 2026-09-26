#!/usr/bin/env python3
"""Emergency responder facilities — reference/context layer.

**Data-only by design, not part of the cascade dependency graph.** The dev
doc's asset schema (§3.4) stays hospital/substation/water/road/ambulance/ecc
only — adding fire/police as cascade-participating assets would need a new
`AssetType`, dependency edges, and (per the water-facilities precedent)
would shift the RL action/observation space shapes, requiring env/test
updates. That's explicitly deferred; see MTP_Module_Planner.md's 2026-09-19
entry for the decision. This script just gets the data on record for the
map/dashboard and a future integration.

Real, named, ward-filterable BMC ArcGIS layers (`Fire_Station`,
`Police_Stations` — both carry a `WARD` field, same as `Health_Facilities`).
Confirmed reachable and populated for L-ward/Kurla during the 2026-09-19
project-review session: 1 fire station, 5 police stations.

Usage:
  uv run python scripts/03c_extract_responders.py --config configs/data.yaml --ward L
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from _pipeline_common import (
    BmcArcGisClient,
    configure_logging,
    load_config,
    resolve_path,
    write_manifest,
)


def _stations_to_features(
    geojson: dict[str, Any], asset_type: str, prefix: str
) -> list[dict[str, Any]]:
    features = []
    for i, feat in enumerate(geojson.get("features", [])):
        props = feat.get("properties", {})
        name = props.get("NAME") or f"{asset_type}_{i}"
        features.append(
            {
                "type": "Feature",
                "geometry": feat["geometry"],
                "properties": {
                    "asset_id": f"{prefix}{i}",
                    "asset_type": asset_type,
                    "name": name,
                    "location": props.get("LOCATION"),
                    "ward": props.get("WARD"),
                    "source": "bmc_arcgis",
                },
            }
        )
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--ward", required=True)
    args = parser.parse_args()

    log = configure_logging()
    cfg = load_config(args.config)
    bmc_cfg = cfg["bmc_arcgis"]

    with BmcArcGisClient(bmc_cfg["base_url"]) as bmc:
        fire = bmc.query_geojson("Fire_Station", 0, where=f"WARD='{args.ward}'")
        police = bmc.query_geojson("Police_Stations", 0, where=f"WARD='{args.ward}'")

    fire_features = _stations_to_features(fire, "fire_station", "FS")
    police_features = _stations_to_features(police, "police_station", "PS")
    log.info("fetched_responders", n_fire=len(fire_features), n_police=len(police_features))

    all_features = fire_features + police_features
    feature_collection = {"type": "FeatureCollection", "features": all_features}

    out_path = resolve_path(cfg["paths"]["processed_dir"]) / "responders.geojson"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(feature_collection, indent=2))
    log.info("wrote_responders", path=str(out_path), count=len(all_features))

    write_manifest(
        out_path,
        inputs={
            "fire_source": "BMC ArcGIS REST (Fire_Station), WARD filter",
            "police_source": "BMC ArcGIS REST (Police_Stations), WARD filter",
            "ward": args.ward,
        },
        row_count=len(all_features),
        extra={"n_fire": len(fire_features), "n_police": len(police_features)},
    )


if __name__ == "__main__":
    main()
