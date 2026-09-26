#!/usr/bin/env python3
"""Elevation (dev doc §2.2 stage 4).

Deviation from the dev doc's literal "SRTM 30m via elevation/rasterio":
reads the Copernicus GLO-30 DEM as a Cloud-Optimized GeoTIFF directly from
AWS Open Data over HTTPS (`rasterio` `/vsicurl/`), keyless, no GDAL-CLI
`eio` dependency. Same ~30m resolution class as SRTM. See
MTP_Development_Document.md §2.2 for this documented deviation.

Usage:
  uv run python scripts/04_get_dem.py --config configs/data.yaml
  (requires data/processed/ward_boundary.geojson from 01_get_boundary.py)
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import rasterio
from rasterio.windows import from_bounds

from _pipeline_common import configure_logging, load_config, resolve_path, write_manifest


def _boundary_bounds(boundary_geojson_path: str) -> tuple[float, float, float, float]:
    with open(boundary_geojson_path) as f:
        d = json.load(f)
    coords = d["features"][0]["geometry"]["coordinates"]

    def flat(c: object) -> list[list[float]]:
        if isinstance(c[0], int | float):  # type: ignore[index]
            return [c]  # type: ignore[list-item]
        out: list[list[float]] = []
        for sub in c:  # type: ignore[union-attr]
            out += flat(sub)
        return out

    pts = flat(coords)
    lons = [p[0] for p in pts]
    lats = [p[1] for p in pts]
    return min(lons), min(lats), max(lons), max(lats)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--buffer-deg", type=float, default=0.01, help="bbox padding, ~1km")
    args = parser.parse_args()

    log = configure_logging()
    cfg = load_config(args.config)
    dem_cfg = cfg["dem"]

    boundary_path = resolve_path(cfg["paths"]["processed_dir"]) / "ward_boundary.geojson"
    if not boundary_path.exists():
        raise SystemExit(f"{boundary_path} not found — run 01_get_boundary.py first")

    west, south, east, north = _boundary_bounds(str(boundary_path))
    b = args.buffer_deg
    west, south, east, north = west - b, south - b, east + b, north + b
    log.info("dem_clip_bounds", west=west, south=south, east=east, north=north)

    tile_url = f"{dem_cfg['base_url']}/{dem_cfg['tile']}/{dem_cfg['tile']}.tif"
    vsi_url = f"/vsicurl/{tile_url}"

    out_path = resolve_path(cfg["paths"]["processed_dir"]) / "dem.tif"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_USE_HEAD="NO"),
        rasterio.open(vsi_url) as src,
    ):
        window = from_bounds(west, south, east, north, transform=src.transform)
        data = src.read(1, window=window)
        if data.size == 0:
            raise SystemExit(
                f"Empty DEM clip for bounds ({west},{south},{east},{north}) — "
                f"ward likely falls outside tile {dem_cfg['tile']}"
            )
        transform = src.window_transform(window)
        profile = src.profile.copy()
        profile.update(
            height=data.shape[0],
            width=data.shape[1],
            transform=transform,
            driver="GTiff",
        )
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data, 1)

    log.info(
        "wrote_dem",
        path=str(out_path),
        shape=data.shape,
        min_elev=float(np.nanmin(data)),
        max_elev=float(np.nanmax(data)),
    )

    write_manifest(
        out_path,
        inputs={"source": "Copernicus GLO-30 DEM (AWS Open Data)", "tile_url": tile_url},
        row_count=int(data.size),
        extra={
            "bounds": [west, south, east, north],
            "shape": list(data.shape),
            "min_elev_m": float(np.nanmin(data)),
            "max_elev_m": float(np.nanmax(data)),
        },
    )


if __name__ == "__main__":
    main()
