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

2026-09-26 (dev doc §2.2 stage 5, revised): the combination changed from
`inv_norm_elev × inv_norm_dist × land_cover` to a weighted linear
combination of **5-class quantile ratings**, the standard GIS/AHP flood-
susceptibility form (Shrestha et al. 2025, Water 17:937, Eq. 5):

    S = W_ELEV·R_elev + W_DRAIN·R_drain + W_LANDCOVER·R_lc

Why: min-max normalisation over the whole DEM (−8.3 to 208 m) put 90% of
ward cells at ≥ 0.57 (median 0.81), because a few hills stretched the
range. Every flood then covered the whole ward (Stage 2 review, finding
2.6). Quantile classes computed over cells *inside the ward* spread the
ratings across the ward by construction. Weights are Mann & Gupta 2023's
Greater-Mumbai AHP weights (Environ. Monit. Assess. 195:1534),
renormalised over the factors this pipeline has: terrain ← their slope
(20.96), drainage ← natural drainage + sewers/storm drains (8.97 + 13.99),
LULC ← 17.52. Mapping their slope weight onto our elevation term is a
disclosed substitution. The formula is still unvalidated against
observed flood locations.

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

# Mann & Gupta 2023 AHP weights, renormalised (see module docstring).
_MG_TERRAIN, _MG_DRAINAGE, _MG_LULC = 20.96, 8.97 + 13.99, 17.52
_MG_TOTAL = _MG_TERRAIN + _MG_DRAINAGE + _MG_LULC
W_ELEV = _MG_TERRAIN / _MG_TOTAL  # ~0.341
W_DRAIN = _MG_DRAINAGE / _MG_TOTAL  # ~0.374
W_LANDCOVER = _MG_LULC / _MG_TOTAL  # ~0.285
N_CLASSES = 5


def quantile_class_rating(
    values: np.ndarray, in_ward: np.ndarray, *, higher_is_riskier: bool
) -> np.ndarray:
    """Reclassify `values` into `N_CLASSES` equal-count classes, with class
    breaks taken from cells inside the ward only, and return ratings
    1/N..1 (e.g. 0.2..1.0). `higher_is_riskier=False` means low values
    (low ground, near drainage) get the highest rating."""
    breaks = np.quantile(values[in_ward], np.linspace(0, 1, N_CLASSES + 1)[1:-1])
    classes = np.digitize(values, breaks)  # 0..N_CLASSES-1
    if not higher_is_riskier:
        classes = (N_CLASSES - 1) - classes
    return (classes + 1) / N_CLASSES


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
    boundary_path = resolve_path(cfg["paths"]["processed_dir"]) / "ward_boundary.geojson"
    if not boundary_path.exists():
        raise SystemExit(f"{boundary_path} not found — run 01_get_boundary.py first")
    with boundary_path.open() as f:
        ward_geom = shape(json.load(f)["features"][0]["geometry"])

    with rasterio.open(dem_path) as src:
        elevation = src.read(1).astype(np.float64)
        profile = src.profile.copy()
        transform = src.transform
        # Pixel size in metres — DEM is in WGS84 degrees, so approximate via
        # the transform's resolution at this latitude (~111km/deg lat).
        pixel_deg = abs(src.transform.a)
        pixel_m = pixel_deg * 111_320.0

    # Quantile class breaks come from cells inside the ward only.
    in_ward = rasterio.features.geometry_mask(
        [ward_geom], out_shape=elevation.shape, transform=transform, invert=True
    )

    # 1. Elevation rating: lower ground -> higher rating.
    if np.nanmax(elevation[in_ward]) == np.nanmin(elevation[in_ward]):
        raise SystemExit("DEM has no elevation variation inside the ward — cannot classify")
    r_elev = quantile_class_rating(elevation, in_ward, higher_is_riskier=False)

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
    # Closer to drainage -> higher rating.
    r_drain = quantile_class_rating(distance_m, in_ward, higher_is_riskier=False)

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

    susceptibility = W_ELEV * r_elev + W_DRAIN * r_drain + W_LANDCOVER * land_cover_factor
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
            "boundary": boundary_path,
            "formula": (
                "W_ELEV*quantile5(elevation) + W_DRAIN*quantile5(distance_to_real_drainage)"
                " + W_LANDCOVER*land_cover_factor (weights: Mann & Gupta 2023, renormalised)"
            ),
            "weights": {"elevation": W_ELEV, "drainage": W_DRAIN, "land_cover": W_LANDCOVER},
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
