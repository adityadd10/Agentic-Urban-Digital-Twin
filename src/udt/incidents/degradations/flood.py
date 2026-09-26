"""Flood degradation (dev doc §4.2, S1 row, and §3.8 — module M3, flood-only scope).

    depth = severity x susceptibility(x) x footprint(x) x envelope(t) x MAX_DEPTH
    Roads:      blockage = clip(depth / 0.6 m)       (reversible: drained road = passable)
    Facilities: fragility-sampled critical depth d_c; depth >= d_c -> failed,
                and the failure persists until repaired (dev doc §3.8 item 4)

**2026-09-26 revision (dev doc §3.8), replacing the original design.** The
Stage 2 review measured that every flood took down the whole ward at every
severity, that facilities healed themselves as water receded, and that damage
was a deterministic straight line of depth. Changes:

1. **Spatial footprint.** `footprint_weight` is a Gaussian around a rainfall
   centroid (`incident.profile["footprint"]`: `lon`, `lat`, `sigma_m`), so
   different scenarios flood different parts of the ward. No footprint in the
   profile -> weight 1 everywhere (the old whole-ward behaviour, kept for
   hand-built incidents in tests).
2. **Envelope timing from the scenario.** `incident_envelope` reads
   `growth_hours`/`hold_hours`/`recede_hours` from `incident.profile`
   (defaults 2/8/6 h, the original constants).
3. **Fragility, not a straight line.** Each facility has a critical depth
   `d_c` drawn from an empirical fragility curve. Substation: Nukavarapu &
   Durbha 2020 (ISPRS IJGI 9(6):387, Table 1), used directly as a piecewise-
   linear CDF that is 0 below 0.1 m and jumps to 0.333 there, as the source
   says flooding starts at 0.1 m (twin-v2 fix; twin-v1 wrongly interpolated
   from 0 m). Hospital and water pump: the same shape rescaled so its median
   is 0.6 m, the level at which that paper's Hospital A floods (it treats the
   pumping station as flooding at the hospital's level). The rescaling is a
   disclosed assumption. A lognormal fit to the table was tried and rejected:
   it misses the steep top end (0.87 vs 0.968 at 0.5 m).
4. **Persistent damage.** When `depth >= d_c` the facility drops to a per-type
   residual (`FAILED_RESIDUAL`) and stays there; receding water no longer
   restores it. Only the agent's repair action does, and a repair doesn't
   hold while the facility is still under `depth >= d_c`. This replaces the
   earlier "intrinsic tracks current depth" interpretation, which let the
   flood repair things for free and made the repair action nearly pointless.
5. **Reproducible randomness.** `d_c` comes from
   `np.random.default_rng([fragility_seed, crc32(asset_id)])`, so it doesn't
   depend on graph iteration order. `FloodDegradation.with_fragility_seed`
   returns a copy that redraws `d_c` for not-yet-failed facilities,
   conditional on `d_c` exceeding the deepest water each has already survived
   (`attributes["max_flood_depth_m"]`). The counterfactual engine uses this so
   each rollout samples the fragility it can't know (dev doc §3.8 item 5).

Unchanged: roads, ambulances and the ECC take no intrinsic damage. Roads get
`blockage`/`flood_depth_m` attributes instead, read by `cascade.py`.
"""

from __future__ import annotations

import math
import zlib
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import rasterio

from udt.common.models import Asset, AssetType, Incident
from udt.incidents.registry import register_incident

ROAD_BLOCKAGE_DEPTH_SCALE_M = 0.6  # dev doc §4.2, exact
MAX_DEPTH_AT_SEVERITY_1_M = 2.0  # depth when severity=1, susceptibility=1, footprint=1, at peak

GROWTH_HOURS = 2.0  # defaults when incident.profile doesn't say
HOLD_HOURS = 8.0
RECEDE_HOURS = 6.0

# Nukavarapu & Durbha 2020, Table 1 (Electrical Substation A), used as given:
# P(fail) = 0 below 0.1 m ("the flooding of the substation would start at
# 0.1 m"), 0.333 at 0.1 m, then the table. twin-v2 fix (2026-09-27): twin-v1
# prepended a (0 m, 0) point and interpolated, which let 42% of sampled
# substations fail below 0.1 m and 22% below 5 cm, which the source doesn't
# support (and the Indian Electricity Rules it cites put a substation's
# formation level >= 600 mm above its surroundings).
_SUBSTATION_DEPTHS_M = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
_FRAGILITY_PROBS = np.array([0.333, 0.475, 0.67, 0.84, 0.968, 1.0])
HOSPITAL_WATER_MEDIAN_DEPTH_M = 0.6  # same paper: Hospital A floods at ~0.6 m
_SUBSTATION_MEDIAN_M = float(np.interp(0.5, _FRAGILITY_PROBS, _SUBSTATION_DEPTHS_M))
_HOSPITAL_WATER_DEPTHS_M = _SUBSTATION_DEPTHS_M * (
    HOSPITAL_WATER_MEDIAN_DEPTH_M / _SUBSTATION_MEDIAN_M
)
FRAGILITY_DEPTHS_M: dict[AssetType, np.ndarray[Any, np.dtype[np.float64]]] = {
    AssetType.SUBSTATION: _SUBSTATION_DEPTHS_M,
    AssetType.HOSPITAL: _HOSPITAL_WATER_DEPTHS_M,
    AssetType.WATER: _HOSPITAL_WATER_DEPTHS_M,
}
# Intrinsic level a facility drops to once flooded (disclosed constants): a
# flooded substation is switched off (JRC 2019), a flooded pump stops, a
# flooded hospital loses its ground floor/basement but upper floors still work.
FAILED_RESIDUAL: dict[AssetType, float] = {
    AssetType.SUBSTATION: 0.0,
    AssetType.WATER: 0.0,
    AssetType.HOSPITAL: 0.3,
}
_NOT_FLOOD_DAMAGED = (AssetType.ROAD, AssetType.AMBULANCE, AssetType.ECC)
_M_PER_DEG_LAT = 110_540.0
_M_PER_DEG_LON_EQUATOR = 111_320.0


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


def temporal_multiplier(
    hours_since_onset: float,
    growth_hours: float = GROWTH_HOURS,
    hold_hours: float = HOLD_HOURS,
    recede_hours: float = RECEDE_HOURS,
) -> float:
    """Growth (0->1) / hold (1) / recede (1->0) envelope. 0 before onset
    and after the incident has fully receded."""
    if hours_since_onset < 0:
        return 0.0
    if hours_since_onset < growth_hours:
        return hours_since_onset / growth_hours
    if hours_since_onset < growth_hours + hold_hours:
        return 1.0
    recede_elapsed = hours_since_onset - growth_hours - hold_hours
    if recede_elapsed < recede_hours:
        return 1.0 - recede_elapsed / recede_hours
    return 0.0


def incident_envelope(incident: Incident, hours_since_onset: float) -> float:
    """`temporal_multiplier` with this incident's own durations (dev doc §4.3)."""
    p = incident.profile
    return temporal_multiplier(
        hours_since_onset,
        float(p.get("growth_hours", GROWTH_HOURS)),
        float(p.get("hold_hours", HOLD_HOURS)),
        float(p.get("recede_hours", RECEDE_HOURS)),
    )


def footprint_weight(incident: Incident, lon: float, lat: float) -> float:
    """Gaussian rainfall footprint, `exp(-d^2 / 2 sigma^2)` (dev doc §4.2).
    1.0 everywhere if the incident has no footprint."""
    fp = incident.profile.get("footprint")
    if not fp:
        return 1.0
    c_lon, c_lat, sigma = float(fp["lon"]), float(fp["lat"]), float(fp["sigma_m"])
    dx = (lon - c_lon) * _M_PER_DEG_LON_EQUATOR * math.cos(math.radians(c_lat))
    dy = (lat - c_lat) * _M_PER_DEG_LAT
    return math.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))


def flood_depth_m(
    incident: Incident,
    asset: Asset,
    tick: int,
    susceptibility_raster: SusceptibilityRaster,
    dt_minutes: float = 5.0,
) -> float:
    """`depth = severity x susceptibility x footprint x envelope x MAX_DEPTH`."""
    hours_since_onset = (tick - incident.onset_tick) * dt_minutes / 60.0
    envelope = incident_envelope(incident, hours_since_onset)
    if envelope <= 0.0:
        return 0.0
    lon, lat = _asset_lon_lat(asset)
    susceptibility = susceptibility_raster.value_at(lon, lat)
    return (
        incident.severity
        * susceptibility
        * footprint_weight(incident, lon, lat)
        * envelope
        * MAX_DEPTH_AT_SEVERITY_1_M
    )


def fragility_probability(asset_type: AssetType, depth_m: float) -> float:
    """P(facility fails | depth), from the empirical curve."""
    return float(
        np.interp(depth_m, FRAGILITY_DEPTHS_M[asset_type], _FRAGILITY_PROBS, left=0.0, right=1.0)
    )


def sample_critical_depth(
    asset_type: AssetType, rng: np.random.Generator, survived_depth_m: float = 0.0
) -> float:
    """Inverse-CDF sample of `d_c`, conditional on `d_c > survived_depth_m`.
    The CDF jumps from 0 to 0.333 at the curve's first depth, so u < 0.333
    maps to that depth (np.interp clamps below the first point)."""
    depths = FRAGILITY_DEPTHS_M[asset_type]
    u_min = fragility_probability(asset_type, survived_depth_m)
    if u_min >= 1.0:
        return survived_depth_m + 1e-6  # survived past the curve's end: fails at any more water
    u = float(rng.uniform(u_min, 1.0))
    return max(float(np.interp(u, _FRAGILITY_PROBS, depths)), survived_depth_m + 1e-6)


def median_critical_depth(asset_type: AssetType) -> float:
    return float(np.interp(0.5, _FRAGILITY_PROBS, FRAGILITY_DEPTHS_M[asset_type]))


@register_incident("flood")
def flood_degradation(
    incident: Incident,
    asset: Asset,
    tick: int,
    susceptibility_raster: SusceptibilityRaster,
    dt_minutes: float = 5.0,
    critical_depth_m: float | None = None,
) -> float:
    """Intrinsic-level reduction for `asset` at `tick` (dev doc §4.2's
    signature, plus `critical_depth_m`). Never negative: damage persists.
    Without a sampled `critical_depth_m`, the curve's median is used.
    Roads, ambulances and the ECC always return 0 (see module docstring)."""
    if asset.asset_type in _NOT_FLOOD_DAMAGED:
        return 0.0
    depth = flood_depth_m(incident, asset, tick, susceptibility_raster, dt_minutes)
    d_c = (
        critical_depth_m
        if critical_depth_m is not None
        else median_critical_depth(asset.asset_type)
    )
    if depth < d_c:
        return 0.0
    return max(0.0, asset.intrinsic_level - FAILED_RESIDUAL[asset.asset_type])


class FloodDegradation:
    """The `(tick, graph) -> {asset_id: reduction}` callable `Simulator.step`
    expects. Also writes road `blockage`/`flood_depth_m` and each facility's
    `max_flood_depth_m` attribute, and holds the per-facility critical depths."""

    def __init__(
        self,
        incident: Incident,
        susceptibility_raster: SusceptibilityRaster,
        *,
        dt_minutes: float = 5.0,
        fragility_seed: int | None = None,
        resample_after_survival: bool = False,
    ) -> None:
        self.incident = incident
        self.raster = susceptibility_raster
        self.dt_minutes = dt_minutes
        self.fragility_seed = (
            int(incident.profile.get("fragility_seed", 0))
            if fragility_seed is None
            else fragility_seed
        )
        self._resample_after_survival = resample_after_survival
        self._critical_depth: dict[str, float] = {}
        # susceptibility x footprint per asset: static for one incident, and
        # sampling the raster every tick was ~60% of a rollout's runtime.
        # Shared (not copied) with `with_fragility_seed` copies.
        self._spatial_weight: dict[str, float] = {}

    def with_fragility_seed(self, seed: int) -> FloodDegradation:
        """A copy for one counterfactual rollout: critical depths redrawn
        with `seed`, conditional on what each facility already survived."""
        copy = FloodDegradation(
            self.incident,
            self.raster,
            dt_minutes=self.dt_minutes,
            fragility_seed=seed,
            resample_after_survival=True,
        )
        copy._spatial_weight = self._spatial_weight
        return copy

    def depth_m(self, asset: Asset, tick: int) -> float:
        """Same value as `flood_depth_m`, using the cached spatial weight."""
        hours_since_onset = (tick - self.incident.onset_tick) * self.dt_minutes / 60.0
        envelope = incident_envelope(self.incident, hours_since_onset)
        if envelope <= 0.0:
            return 0.0
        weight = self._spatial_weight.get(asset.asset_id)
        if weight is None:
            lon, lat = _asset_lon_lat(asset)
            weight = self.raster.value_at(lon, lat) * footprint_weight(self.incident, lon, lat)
            self._spatial_weight[asset.asset_id] = weight
        return self.incident.severity * weight * envelope * MAX_DEPTH_AT_SEVERITY_1_M

    def critical_depth(self, asset: Asset) -> float:
        d_c = self._critical_depth.get(asset.asset_id)
        if d_c is None:
            rng = np.random.default_rng([self.fragility_seed, zlib.crc32(asset.asset_id.encode())])
            survived = (
                float(asset.attributes.get("max_flood_depth_m", 0.0))
                if self._resample_after_survival
                else 0.0
            )
            d_c = sample_critical_depth(asset.asset_type, rng, survived)
            self._critical_depth[asset.asset_id] = d_c
        return d_c

    def __call__(self, tick: int, graph: nx.DiGraph[str]) -> dict[str, float]:
        reductions: dict[str, float] = {}
        for asset_id in graph.nodes:
            asset: Asset = graph.nodes[asset_id]["asset"]
            if asset.asset_type == AssetType.ROAD:
                depth = self.depth_m(asset, tick)
                asset.attributes["flood_depth_m"] = depth
                asset.attributes["blockage"] = max(
                    0.0, min(1.0, depth / ROAD_BLOCKAGE_DEPTH_SCALE_M)
                )
            elif asset.asset_type in FAILED_RESIDUAL:
                d_c = self.critical_depth(asset)  # drawn before this tick's depth is recorded
                depth = self.depth_m(asset, tick)
                # Same rule as `flood_degradation`, inlined to reuse the cached depth.
                reduction = (
                    max(0.0, asset.intrinsic_level - FAILED_RESIDUAL[asset.asset_type])
                    if depth >= d_c
                    else 0.0
                )
                asset.attributes["max_flood_depth_m"] = max(
                    float(asset.attributes.get("max_flood_depth_m", 0.0)), depth
                )
                if reduction > 0:
                    reductions[asset_id] = reduction
        return reductions


def make_flood_degradation_fn(
    incident: Incident,
    susceptibility_raster: SusceptibilityRaster,
    *,
    dt_minutes: float = 5.0,
) -> FloodDegradation:
    """Builds the flood's degradation callable. Critical depths use
    `incident.profile["fragility_seed"]` (default 0)."""
    return FloodDegradation(incident, susceptibility_raster, dt_minutes=dt_minutes)
