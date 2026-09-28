"""Physical scenario features for the twin-v3 suite (protocol
2026-09-29_twin_v3_success_criteria §3).

Everything here is computed from the flood field, the fragility curves and the
network alone, before any policy runs, so stratifying or oversampling on these
features never selects scenarios by how a policy performs.

- `sector`: the hospital whose Voronoi sector contains the footprint centre
  (the nearest hospital to it).
- `severity_band`: mild / moderate / severe = equal thirds of the generator's
  severity range.
- T1 (alternative-access closure): some hospital that is *not* expected to
  fail (peak failure probability <= 0.5) has every access road blocked (road
  depth >= `ROAD_BLOCKAGE_DEPTH_SCALE_M`) within 6 h of onset.
- T2 (multi-facility threat): at least 2 facilities have peak failure
  probability > 0.5.
- T1' (route-based alternative-destination trade-off, protocol addendum 3,
  replacing T1, which this twin cannot produce): some hospital has peak
  failure probability > 0.5 and, at some 30-min check within 6 h of onset,
  the route from it to the alternative hospital that is nearer without
  flooding is unreachable or slower than the route to the farther one, which
  is reachable. Needs a `RoadNetwork` (`route_tradeoff`).
- T3 (high call volume): expected emergency calls above the suite median; the
  expected count is `rate x severity x 24 h`, and the median is taken over the
  split in the generator.
"""

from __future__ import annotations

import math
from typing import Any

from udt.common.models import Asset, AssetType, DependencyGraph, Scenario
from udt.incidents.degradations.flood import (
    ROAD_BLOCKAGE_DEPTH_SCALE_M,
    SusceptibilityRaster,
    flood_depth_m,
    fragility_probability,
    incident_envelope,
)
from udt.scenarios.generator import DEFAULT_SEVERITY_RANGE
from udt.twin.ambulances import REQUEST_RATE_PER_HOUR_AT_SEVERITY_1_DEFAULT
from udt.twin.road_network import RoadNetwork

FACILITY_TYPES = (AssetType.HOSPITAL, AssetType.SUBSTATION, AssetType.WATER)
SEVERITY_BANDS = ("mild", "moderate", "severe")
ACCESS_WINDOW_HOURS = 6.0
EPISODE_HOURS = 24.0


def _lon_lat(asset: Asset) -> tuple[float, float]:
    coords = asset.geometry["coordinates"]
    if asset.geometry["type"] == "Point":
        return float(coords[0]), float(coords[1])
    xs, ys = [c[0] for c in coords], [c[1] for c in coords]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def sector(scenario: Scenario, graph: DependencyGraph) -> str:
    fp = scenario.incident.profile["footprint"]
    lon0, lat0 = float(fp["lon"]), float(fp["lat"])
    k = math.cos(math.radians(lat0))
    hospitals = [a for a in graph.assets if a.asset_type == AssetType.HOSPITAL]
    return min(
        hospitals,
        key=lambda h: ((_lon_lat(h)[0] - lon0) * k) ** 2 + (_lon_lat(h)[1] - lat0) ** 2,
    ).asset_id


def severity_band(severity: float) -> str:
    lo, hi = DEFAULT_SEVERITY_RANGE
    third = (hi - lo) / 3.0
    index = min(2, max(0, int((severity - lo) // third)))
    return SEVERITY_BANDS[index]


def _peak_tick(scenario: Scenario, dt_minutes: float = 5.0) -> int:
    """A tick inside the hold phase (envelope = 1)."""
    growth = float(scenario.incident.profile["growth_hours"])
    return scenario.incident.onset_tick + math.ceil(growth * 60.0 / dt_minutes) + 1


def tradeoff_features(
    scenario: Scenario, graph: DependencyGraph, raster: SusceptibilityRaster
) -> dict[str, Any]:
    incident = scenario.incident
    peak = _peak_tick(scenario)
    assets = {a.asset_id: a for a in graph.assets}

    def depth(asset: Asset) -> float:
        return flood_depth_m(incident, asset, peak, raster)

    p_fail = {
        a.asset_id: fragility_probability(a.asset_type, depth(a))
        for a in graph.assets
        if a.asset_type in FACILITY_TYPES
    }
    window_ticks = int(ACCESS_WINDOW_HOURS * 12)
    envelope_max = max(incident_envelope(incident, t / 12.0) for t in range(0, window_ticks + 1))
    access_closed: dict[str, bool] = {}
    for h in (a for a in graph.assets if a.asset_type == AssetType.HOSPITAL):
        roads = [e.supplier for e in graph.edges if e.consumer == h.asset_id and e.kind == "access"]
        access_closed[h.asset_id] = bool(roads) and all(
            depth(assets[r]) * envelope_max >= ROAD_BLOCKAGE_DEPTH_SCALE_M for r in roads
        )
    t1 = any(access_closed[h] and p_fail[h] <= 0.5 for h in access_closed)
    t2 = sum(p > 0.5 for p in p_fail.values()) >= 2
    expected_calls = REQUEST_RATE_PER_HOUR_AT_SEVERITY_1_DEFAULT * incident.severity * EPISODE_HOURS
    return {
        "sector": sector(scenario, graph),
        "severity_band": severity_band(incident.severity),
        "T1_alt_access_closure": t1,
        "T2_multi_facility_threat": t2,
        "expected_calls": expected_calls,
        "p_fail": {k: round(v, 3) for k, v in p_fail.items()},
        "access_closed": access_closed,
    }


def route_tradeoff(
    scenario: Scenario,
    graph: DependencyGraph,
    road_network: RoadNetwork,
    p_fail: dict[str, float],
) -> bool:
    """T1' (see module docstring). Mutates `road_network`'s per-tick state."""
    hospitals = [a for a in graph.assets if a.asset_type == AssetType.HOSPITAL]
    source = max(hospitals, key=lambda h: p_fail[h.asset_id])
    if p_fail[source.asset_id] <= 0.5 or len(hospitals) < 3:
        return False
    node = {h.asset_id: road_network.nearest_node(*_lon_lat(h)) for h in hospitals}
    src = node[source.asset_id]
    incident = scenario.incident
    road_network.update_for_tick(incident, incident.onset_tick, 5.0)
    others = [h.asset_id for h in hospitals if h.asset_id != source.asset_id]
    free: dict[str, float] = {}
    for o in others:
        t = road_network.travel_time_minutes(src, node[o])
        free[o] = t if t is not None else float("inf")
    near, far = sorted(others, key=lambda o: free[o])[:2]
    for tick in range(6, int(ACCESS_WINDOW_HOURS * 12) + 1, 6):
        road_network.update_for_tick(incident, incident.onset_tick + tick, 5.0)
        t_near = road_network.travel_time_minutes(src, node[near])
        t_far = road_network.travel_time_minutes(src, node[far])
        if t_far is not None and (t_near is None or t_far < t_near):
            return True
    return False
