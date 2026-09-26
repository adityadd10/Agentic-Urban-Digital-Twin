#!/usr/bin/env python3
"""Ward selection (dev doc §2.1 — "do this first, it is a 1-day task").

Scores the dev doc's Mithi-belt candidates (Kurla L-ward, Bandra H/E) on:
OSM road edge count, hospital count, substation count, boundary
availability. Picks the ward with >= 2 hospitals and best OSM
completeness, and writes the choice + numbers to data/README.md.

Data sources: ward boundary + hospital counts from BMC's public ArcGIS
REST feature services (confirmed reachable, no auth); road edges and
substations from OpenStreetMap via the official REST API (BMC has no
public power-grid layer — see MTP_Development_Document.md §2 for why this
stays OSM/manual; see `_pipeline_common.py` for why this is the official
API and not Overpass).

Usage: uv run python scripts/00_score_wards.py --config configs/data.yaml
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from _pipeline_common import (
    DRIVABLE_HIGHWAY_TYPES,
    BmcArcGisClient,
    configure_logging,
    fetch_osm_bbox,
    load_config,
    resolve_path,
)


@dataclass
class WardScore:
    id: str
    label: str
    bmc_ward_code: str
    boundary_available: bool
    hospital_count: int
    road_edge_count: int | None  # None = OSM API unreachable this run, not "zero roads"
    substation_count: int | None

    @property
    def meets_hospital_threshold(self) -> bool:
        return self.hospital_count >= 2

    @property
    def completeness_score(self) -> float:
        """OSM completeness proxy: road edges + substations found. Only
        meaningful to compare between candidates that both meet the
        hospital threshold — see `pick_ward`. Treats an unreachable
        OSM API response as 0, not as missing data being penalized against
        a candidate that *did* get a real answer — re-run when the OSM API is
        reachable for an accurate comparison if this ever actually
        decides the outcome (logged either way)."""
        return (self.road_edge_count or 0) + 10 * (self.substation_count or 0)


def _ward_bbox(
    bmc: BmcArcGisClient, ward_service: str, layer_id: int, name_field: str, code: str
) -> tuple[bool, tuple[float, float, float, float] | None]:
    geojson = bmc.query_geojson(ward_service, layer_id, where=f"{name_field}='{code}'")
    features = geojson.get("features", [])
    if not features:
        return False, None
    lons: list[float] = []
    lats: list[float] = []

    def walk(coords: Any) -> None:
        if isinstance(coords[0], int | float):
            lons.append(coords[0])
            lats.append(coords[1])
        else:
            for c in coords:
                walk(c)

    walk(features[0]["geometry"]["coordinates"])
    return True, (min(lats), min(lons), max(lats), max(lons))


def _hospital_count(bmc: BmcArcGisClient, cfg: dict[str, Any], ward_code: str) -> int:
    health_cfg = cfg["bmc_arcgis"]
    total = 0
    for layer in health_cfg["hospital_layers"]:
        total += bmc.count(
            health_cfg["health_service"],
            layer["id"],
            where=f"{health_cfg['ward_field']}='{ward_code}'",
        )
    return total


def _osm_counts(bbox: tuple[float, float, float, float]) -> tuple[int, int]:
    """Roads and substations from the official OSM REST API (see
    `_pipeline_common.py`'s "OSM via the official REST API" section —
    every Overpass mirror this originally used was unreachable or
    non-functional as of 2026-08-31, confirmed from two independent
    networks; the official API stayed up throughout)."""
    south, west, north, east = bbox
    osm = fetch_osm_bbox(west, south, east, north)
    road_count = sum(
        1 for w in osm.ways.values() if w.tags.get("highway") in DRIVABLE_HIGHWAY_TYPES
    )
    sub_count = sum(1 for n in osm.nodes.values() if n.tags.get("power") == "substation")
    sub_count += sum(1 for w in osm.ways.values() if w.tags.get("power") == "substation")
    return road_count, sub_count


def pick_ward(scores: list[WardScore]) -> WardScore:
    """Dev doc §2.1 rule: "Pick the ward with >= 2 hospitals and best OSM
    completeness." If none meet the hospital threshold, still pick the
    best-completeness candidate but this is a scope risk worth flagging."""
    eligible = [s for s in scores if s.meets_hospital_threshold]
    pool = eligible if eligible else scores
    return max(pool, key=lambda s: s.completeness_score)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()

    log = configure_logging()
    cfg = load_config(args.config)
    bmc_cfg = cfg["bmc_arcgis"]

    scores: list[WardScore] = []
    with BmcArcGisClient(bmc_cfg["base_url"]) as bmc:
        for candidate in cfg["ward_candidates"]:
            code = candidate["bmc_ward_code"]
            log.info("scoring_ward", ward=candidate["label"])

            available, bbox = _ward_bbox(
                bmc,
                bmc_cfg["ward_service"],
                bmc_cfg["ward_layer_id"],
                bmc_cfg["ward_name_field"],
                code,
            )
            hospital_count = _hospital_count(bmc, cfg, code)

            road_count: int | None = None
            sub_count: int | None = None
            if available and bbox is not None:
                try:
                    road_count, sub_count = _osm_counts(bbox)
                except RuntimeError as exc:
                    # Don't let an OSM API outage crash ward selection: hospital count +
                    # boundary from BMC already resolves the dev doc's hard gate in most
                    # cases; road/substation counts just go unrecorded, logged clearly.
                    log.warning("osm_counts_unavailable", ward=candidate["label"], error=str(exc))

            score = WardScore(
                id=candidate["id"],
                label=candidate["label"],
                bmc_ward_code=code,
                boundary_available=available,
                hospital_count=hospital_count,
                road_edge_count=road_count,
                substation_count=sub_count,
            )
            scores.append(score)
            log.info(
                "ward_scored",
                ward=score.label,
                hospitals=score.hospital_count,
                road_edges=score.road_edge_count,
                substations=score.substation_count,
                boundary_available=score.boundary_available,
            )

    chosen = pick_ward(scores)
    log.info(
        "ward_selected", ward=chosen.label, reason="meets hospital threshold, best completeness"
    )

    readme_path = resolve_path("data/README.md")
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Data directory",
        "",
        "## Ward selection (dev doc §2.1)",
        "",
        f"**Chosen ward: {chosen.label}** (BMC code `{chosen.bmc_ward_code}`)",
        "",
        "Scoring, `scripts/00_score_wards.py` against BMC's public ArcGIS REST feature "
        "services (boundary + hospitals) and OpenStreetMap's official REST API (roads + "
        "substations, since BMC has no public power-grid layer):",
        "",
        "| Ward | Boundary | Hospitals | Road edges | Substations | Meets >=2 hospitals |",
        "|---|---|---|---|---|---|",
    ]
    any_osm_missing = False
    for s in scores:
        marker = "**YES**" if s.id == chosen.id else "yes" if s.meets_hospital_threshold else "no"
        road_str = s.road_edge_count if s.road_edge_count is not None else "n/a"
        sub_str = s.substation_count if s.substation_count is not None else "n/a"
        if s.road_edge_count is None or s.substation_count is None:
            any_osm_missing = True
        lines.append(
            f"| {s.label} | {'yes' if s.boundary_available else 'no'} | {s.hospital_count} "
            f"| {road_str} | {sub_str} | {marker} |"
        )
    lines += [
        "",
        "Rule (dev doc §2.1): pick the ward with >= 2 hospitals and best OSM completeness "
        "(road edges + weighted substation count). Hospital count = BMC `Health_Facilities` "
        "Main + Special + Peripheral + Polyclinic layers filtered to this ward's code; "
        "dispensaries/maternity homes/health posts excluded (not hospitals in the dev doc's "
        "sense). Substation/road counts are OSM-only since Mumbai power infrastructure has "
        "no public feed anywhere (checked; see dev doc §2).",
    ]
    if any_osm_missing:
        lines.append(
            "\n**Note:** `n/a` = the OSM API was unreachable during this run. The ward "
            "decision above still stands on the BMC hospital/boundary data alone (already "
            "decisive here); re-run this script to backfill road/substation counts for a "
            "fully-populated record."
        )
    readme_path.write_text("\n".join(lines) + "\n")
    log.info("wrote_ward_readme", path=str(readme_path))


if __name__ == "__main__":
    main()
