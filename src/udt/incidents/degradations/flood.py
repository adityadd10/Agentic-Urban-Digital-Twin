"""Flood degradation (dev doc §4.2, S1 row, module M3 — flood-only scope).

    depth field = severity x susceptibility_raster, grows over 2h, holds
    8h, recedes over 6h
    Roads:                blockage = clip(depth / 0.6m)
    Ground-floor facilities in footprint: intrinsic -= f(depth)

Two things the dev doc leaves for the implementation to pin down,
disclosed here:

1. **"Footprint."** Rather than a separate discrete flood-extent polygon,
   this uses the susceptibility raster itself as the spatial extent — a
   near-zero susceptibility cell already produces a near-zero depth, so a
   separate footprint concept would be redundant for a single ward-scale
   incident. `location` on the `Incident` is still populated (the ward
   boundary) for the schema's sake, just not consulted per-asset here.
2. **f(depth) for facilities.** No formula is given beyond the name.
   Uses the same linear clip as the road formula, at a different scale
   (`FACILITY_DEPTH_SCALE_M`, both configurable at the call site):
   `f(depth) = clip(depth / FACILITY_DEPTH_SCALE_M, 0, 1)`.
3. **What "intrinsic_level -= degradation_fn(...)" means for a receding
   hazard.** Read as a literal permanent per-tick subtraction, a facility
   could never recover as floodwaters recede (intrinsic_level only ever
   decreases). Depth-based damage should instead track *current* depth —
   the same depth does the same damage whether it's tick 10 or tick 50,
   and recession should let intrinsic_level recover. So `flood_degradation`
   returns `current_intrinsic_level - target_intrinsic_level` (target =
   `1 - f(depth)`), which is negative during recession — still literally
   an `intrinsic_level -= degradation_fn(...)` per §3.5 step 1, just one
   that can subtract a negative number and give some level back.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import rasterio

from udt.common.models import Asset, AssetType, Incident
from udt.incidents.registry import register_incident

ROAD_BLOCKAGE_DEPTH_SCALE_M = 0.6  # dev doc §4.2, exact
FACILITY_DEPTH_SCALE_M = 1.2  # disclosed default, see module docstring point 2
MAX_DEPTH_AT_SEVERITY_1_M = 2.0  # depth when severity=1, susceptibility=1, at temporal peak

GROWTH_HOURS = 2.0
HOLD_HOURS = 8.0
RECEDE_HOURS = 6.0


class SusceptibilityRaster:
    """Thin wrapper over `flood_susceptibility.tif` (written by
    `scripts/05_flood_susceptibility.py`) — samples the 0..1 susceptibility
    value at a lon/lat point."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._dataset = rasterio.open(self._path)

    def value_at(self, lon: float, lat: float) -> float:
        row, col = self._dataset.index(lon, lat)
        height, width = self._dataset.shape
        if not (0 <= row < height and 0 <= col < width):
            return 0.0  # outside the raster's coverage -> treat as unaffected
        value = self._dataset.read(1, window=((row, row + 1), (col, col + 1)))[0, 0]
        return float(max(0.0, min(1.0, value)))

    def close(self) -> None:
        self._dataset.close()

    def __enter__(self) -> SusceptibilityRaster:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _asset_lon_lat(asset: Asset) -> tuple[float, float]:
    """Point coordinate for susceptibility sampling: the geometry itself
    if it's a point, else the first coordinate of a linestring/polygon
    (roads) — precise enough at this raster's ~30m resolution."""
    geom = asset.geometry
    coords = geom["coordinates"]
    gtype = geom["type"]
    if gtype == "Point":
        return float(coords[0]), float(coords[1])
    if gtype == "LineString":
        return float(coords[0][0]), float(coords[0][1])
    if gtype == "Polygon":
        return float(coords[0][0][0]), float(coords[0][0][1])
    raise ValueError(f"Unsupported geometry type for susceptibility sampling: {gtype!r}")


def temporal_multiplier(hours_since_onset: float) -> float:
    """Growth (0->1 over 2h) / hold (1 for 8h) / recede (1->0 over 6h)
    envelope, dev doc §4.2 exactly. 0 before onset and after the incident
    has fully receded."""
    if hours_since_onset < 0:
        return 0.0
    if hours_since_onset < GROWTH_HOURS:
        return hours_since_onset / GROWTH_HOURS
    if hours_since_onset < GROWTH_HOURS + HOLD_HOURS:
        return 1.0
    recede_elapsed = hours_since_onset - GROWTH_HOURS - HOLD_HOURS
    if recede_elapsed < RECEDE_HOURS:
        return 1.0 - recede_elapsed / RECEDE_HOURS
    return 0.0


def flood_depth_m(
    incident: Incident,
    asset: Asset,
    tick: int,
    susceptibility_raster: SusceptibilityRaster,
    dt_minutes: float = 5.0,
) -> float:
    """`depth field = severity x susceptibility_raster x temporal_envelope
    x MAX_DEPTH_AT_SEVERITY_1_M` (dev doc §4.2, with the temporal envelope
    and max-depth scale spelled out — see module docstring)."""
    hours_since_onset = (tick - incident.onset_tick) * dt_minutes / 60.0
    envelope = temporal_multiplier(hours_since_onset)
    if envelope <= 0.0:
        return 0.0
    lon, lat = _asset_lon_lat(asset)
    susceptibility = susceptibility_raster.value_at(lon, lat)
    return incident.severity * susceptibility * envelope * MAX_DEPTH_AT_SEVERITY_1_M


@register_incident("flood")
def flood_degradation(
    incident: Incident,
    asset: Asset,
    tick: int,
    susceptibility_raster: SusceptibilityRaster,
    dt_minutes: float = 5.0,
) -> float:
    """Return the intrinsic-level reduction for `asset` at `tick` (dev doc
    §4.2's exact signature — see module docstring point 3 for why this is
    "current level minus depth-derived target", not a permanent
    increment). For roads this returns 0 — roads are affected via
    `blockage` (an attribute, mutated directly by
    `make_flood_degradation_fn` below), not via intrinsic_level, per
    `cascade.py`'s road handling.

    **Bug found and fixed in M4's ambulance-dispatch slice:** ambulances
    (added as regular graph nodes once `twin/ambulances.py` existed) were
    silently taking real depth-based damage here — nothing about their
    dispatch/movement (`twin/ambulances.py`'s `dispatch_ambulance`/
    `advance_ambulances`) ever reads `intrinsic_level`/`functional_level`,
    so the damage had no behavioral effect, it only inflated
    `cascading_failure_count` with meaningless "failures". Excluded here,
    same as roads — vehicles aren't a depth-damaged facility in this
    model. `ECC` is excluded too, for the same reason dev doc §3.2 gives
    it no physical attributes ("coordination bookkeeping only").
    """
    if asset.asset_type in (AssetType.ROAD, AssetType.AMBULANCE, AssetType.ECC):
        return 0.0
    depth = flood_depth_m(incident, asset, tick, susceptibility_raster, dt_minutes)
    damage_fraction = max(0.0, min(1.0, depth / FACILITY_DEPTH_SCALE_M))
    target_intrinsic_level = 1.0 - damage_fraction
    return asset.intrinsic_level - target_intrinsic_level


def make_flood_degradation_fn(
    incident: Incident,
    susceptibility_raster: SusceptibilityRaster,
    *,
    dt_minutes: float = 5.0,
) -> Any:
    """Adapts `flood_degradation` (per-asset, dev doc §4.2 signature) into
    the `(tick, graph) -> {asset_id: reduction}` shape `Simulator.step`
    expects — also sets each road's `blockage` attribute directly (roads
    don't lose intrinsic_level to flooding, they get blocked; see
    `cascade.py`'s `_road_functional_level`)."""

    def degradation_fn(tick: int, graph: nx.DiGraph[str]) -> dict[str, float]:
        reductions: dict[str, float] = {}
        for asset_id in graph.nodes:
            asset: Asset = graph.nodes[asset_id]["asset"]
            if asset.asset_type == AssetType.ROAD:
                depth = flood_depth_m(incident, asset, tick, susceptibility_raster, dt_minutes)
                asset.attributes["flood_depth_m"] = depth
                asset.attributes["blockage"] = max(
                    0.0, min(1.0, depth / ROAD_BLOCKAGE_DEPTH_SCALE_M)
                )
            else:
                reduction = flood_degradation(
                    incident, asset, tick, susceptibility_raster, dt_minutes
                )
                if (
                    reduction != 0
                ):  # can be negative during recession — see module docstring point 3
                    reductions[asset_id] = reduction
        return reductions

    return degradation_fn
