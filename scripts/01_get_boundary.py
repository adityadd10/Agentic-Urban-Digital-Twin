#!/usr/bin/env python3
"""Ward boundary (dev doc §2.2 stage 1).

Pulls the chosen ward's polygon from BMC's public ArcGIS REST `BMC_Ward`
feature service (confirmed reachable, no auth) instead of a Datameet/OSM
relation — see MTP_Development_Document.md §2.2 for the source deviation
and Prototype-1's plan for why (real official boundary > OSM/bounding-box
guess).

Usage:
  uv run python scripts/01_get_boundary.py --config configs/data.yaml --ward L
"""

from __future__ import annotations

import argparse
import json

from _pipeline_common import (
    BmcArcGisClient,
    configure_logging,
    load_config,
    resolve_path,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--ward", required=True, help="BMC ward code, e.g. L or H/E")
    args = parser.parse_args()

    log = configure_logging()
    cfg = load_config(args.config)
    bmc_cfg = cfg["bmc_arcgis"]

    with BmcArcGisClient(bmc_cfg["base_url"]) as bmc:
        geojson = bmc.query_geojson(
            bmc_cfg["ward_service"],
            bmc_cfg["ward_layer_id"],
            where=f"{bmc_cfg['ward_name_field']}='{args.ward}'",
        )

    features = geojson.get("features", [])
    if not features:
        raise SystemExit(
            f"No ward boundary found for code {args.ward!r} in {bmc_cfg['ward_service']}"
        )
    if len(features) > 1:
        log.warning("multiple_ward_features", ward=args.ward, count=len(features))

    out_path = resolve_path(cfg["paths"]["processed_dir"]) / "ward_boundary.geojson"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(geojson, indent=2))
    log.info("wrote_ward_boundary", path=str(out_path), ward=args.ward)

    write_manifest(
        out_path,
        inputs={
            "source": "BMC ArcGIS REST",
            "service": bmc_cfg["ward_service"],
            "layer_id": bmc_cfg["ward_layer_id"],
            "where": f"{bmc_cfg['ward_name_field']}='{args.ward}'",
        },
        row_count=len(features),
        extra={"ward_code": args.ward},
    )


if __name__ == "__main__":
    main()
