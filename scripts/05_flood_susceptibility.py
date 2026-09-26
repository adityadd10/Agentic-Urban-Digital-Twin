#!/usr/bin/env python3
"""Flood susceptibility raster (dev doc §2.2 stage 5).

Dev doc formula: "simplified by design (inverse normalized elevation ×
drainage proximity)". Drainage proximity is real: distance to the actual
mapped drains/streams/rivers/canals in `data/processed/drainage.geojson`
(from `03b_extract_drainage.py`), not a stand-in. An earlier version of
this script used a DEM-only proxy (bottom decile of elevation cells
treated as if they were drainage) because OSM `waterway=*` tags were
thought unreachable from the build sandbox; that limitation turned out not
to hold (see `03b_extract_drainage.py`'s docstring — confirmed 2026-09-18),
so the proxy was replaced with the real thing.

2026-09-19: added a third, real land-cover term — paved/built-up ground
sheds water faster than open ground, so impervious cells (from
`data/processed/landuse.geojson`, `03e_extract_landuse.py`) keep full
susceptibility while pervious cells (grass/forest/farmland/...) are damped
by a disclosed constant (`PERVIOUS_DAMPENING`), not zeroed — open ground in
Mumbai still floods, just somewhat less than paved ground at the same
elevation/drainage-distance. Still "simplified by design," same disclosure
tier as the rest of this formula.

Usage:
  uv run python scripts/05_flood_susceptibility.py --config configs/data.yaml
  (requires data/processed/dem.tif from 04_get_dem.py,
  data/processed/drainage.geojson from 03b_extract_drainage.py, and
  data/processed/landuse.geojson from 03e_extract_landuse.py)
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import rasterio
import rasterio.features
from scipy.ndimage import distance_transform_edt
from shapely.geometry import shape

from _pipeline_common import configure_logging, load_config, resolve_path, write_manifest

# Disclosed dampening for pervious ground (grass/forest/farmland/...) in the
# land-cover term — open ground still floods, just less readily than paved/
# built-up ground at the same elevation/drainage-distance. Not fitted to
# real infiltration data.
PERVIOUS_DAMPENING = 0.7


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()

    log = configure_logging()
    cfg = load_config(args.config)

    dem_path = resolve_path(cfg["paths"]["processed_dir"]) / "dem.tif"
    if not dem_path.exists():
        raise SystemExit(f"{dem_path} not found — run 04_get_dem.py first")
    drainage_path = resolve_path(cfg["paths"]["processed_dir"]) / "drainage.geojson"
    if not drainage_path.exists():
        raise SystemExit(f"{drainage_path} not found — run 03b_extract_drainage.py first")
    landuse_path = resolve_path(cfg["paths"]["processed_dir"]) / "landuse.geojson"
    if not landuse_path.exists():
        raise SystemExit(f"{landuse_path} not found — run 03e_extract_landuse.py first")

    with rasterio.open(dem_path) as src:
        elevation = src.read(1).astype(np.float64)
        profile = src.profile.copy()
        transform = src.transform
        # Pixel size in metres — DEM is in WGS84 degrees, so approximate via
        # the transform's resolution at this latitude (~111km/deg lat).
        pixel_deg = abs(src.transform.a)
        pixel_m = pixel_deg * 111_320.0

    # 1. Inverse normalized elevation: lower ground -> higher susceptibility.
    elev_min, elev_max = np.nanmin(elevation), np.nanmax(elevation)
    if elev_max == elev_min:
        raise SystemExit("DEM has no elevation variation — cannot normalize")
    norm_elev = (elevation - elev_min) / (elev_max - elev_min)
    inv_norm_elev = 1.0 - norm_elev

    # 2. Distance-to-real-drainage: rasterize the real drain/stream/river/
    # canal lines onto the DEM grid, then take the Euclidean distance
    # (in metres) from every cell to the nearest drainage cell.
    with drainage_path.open() as f:
        drainage_fc = json.load(f)
    drainage_geoms = [shape(feat["geometry"]) for feat in drainage_fc["features"]]
    if not drainage_geoms:
        raise SystemExit(f"{drainage_path} has no drainage features — re-run 03b for this ward")
    drainage_mask = rasterio.features.geometry_mask(
        drainage_geoms,
        out_shape=elevation.shape,
        transform=transform,
        invert=True,
        all_touched=True,
    )
    log.info(
        "real_drainage_mask",
        n_drainage_features=len(drainage_geoms),
        n_drainage_cells=int(drainage_mask.sum()),
        total_cells=int(drainage_mask.size),
    )
    # distance_transform_edt gives distance to the nearest *False* cell by
    # default when fed a mask of "drainage" as True — invert so it measures
    # distance FROM drainage cells (0 at drainage, growing outward).
    distance_m = distance_transform_edt(~drainage_mask, sampling=(pixel_m, pixel_m))
    dist_min, dist_max = float(np.min(distance_m)), float(np.max(distance_m))
    norm_dist = (
        (distance_m - dist_min) / (dist_max - dist_min) if dist_max > dist_min else distance_m * 0
    )
    inv_norm_dist = 1.0 - norm_dist  # closer to drainage -> higher susceptibility

    # 3. Land-cover term: impervious (paved/built-up) cells keep full
    # susceptibility; pervious cells are damped by a disclosed constant.
    with landuse_path.open() as f:
        landuse_fc = json.load(f)
    impervious_geoms = [
        shape(feat["geometry"])
        for feat in landuse_fc["features"]
        if feat["properties"].get("impervious")
    ]
    if not impervious_geoms:
        raise SystemExit(f"{landuse_path} has no impervious features — re-run 03e for this ward")
    impervious_mask = rasterio.features.geometry_mask(
        impervious_geoms,
        out_shape=elevation.shape,
        transform=transform,
        invert=True,
        all_touched=True,
    )
    log.info(
        "land_cover_mask",
        n_impervious_features=len(impervious_geoms),
        n_impervious_cells=int(impervious_mask.sum()),
        total_cells=int(impervious_mask.size),
    )
    land_cover_factor = np.where(impervious_mask, 1.0, PERVIOUS_DAMPENING)

    susceptibility = inv_norm_elev * inv_norm_dist * land_cover_factor
    susceptibility = np.clip(susceptibility, 0.0, 1.0)

    out_path = resolve_path(cfg["paths"]["processed_dir"]) / "flood_susceptibility.tif"
    profile.update(dtype="float32", count=1)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(susceptibility.astype(np.float32), 1)

    log.info(
        "wrote_flood_susceptibility",
        path=str(out_path),
        min=float(susceptibility.min()),
        max=float(susceptibility.max()),
        mean=float(susceptibility.mean()),
    )

    write_manifest(
        out_path,
        inputs={
            "dem": dem_path,
            "drainage": drainage_path,
            "landuse": landuse_path,
            "formula": (
                "inv_norm_elevation * inv_norm_distance_to_real_drainage * land_cover_factor"
            ),
            "pervious_dampening": PERVIOUS_DAMPENING,
        },
        row_count=int(susceptibility.size),
        extra={
            "n_drainage_features": len(drainage_geoms),
            "n_drainage_cells": int(drainage_mask.sum()),
            "n_impervious_features": len(impervious_geoms),
            "n_impervious_cells": int(impervious_mask.sum()),
            "min": float(susceptibility.min()),
            "max": float(susceptibility.max()),
            "mean": float(susceptibility.mean()),
        },
    )


if __name__ == "__main__":
    main()
