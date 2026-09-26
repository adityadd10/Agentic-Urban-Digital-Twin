"""Substation load + shedding (dev doc §3.2 `shed_tier`, §5.3's shed-tier
action, §5.6's load-shedding rule, module M4 — the 4th and last of §5.6's
rules).

**Why this needed a real design decision, not just code (flagged before
building it):** `cascade.py` never read a substation's `load_mw`/
`capacity_mw` at all — nothing in the twin failed from being overloaded,
so "shed load before it fails" had nothing to protect against. Two
things had to be decided and are disclosed here, same status as every
other placeholder constant in this codebase:

1. **What makes load change at all.** `06_build_dependency_graph.py`
   sets `load_mw` once, statically, at 40-70% of capacity — it never
   moves, so shedding would stay permanently irrelevant. Resolved: load
   surges with the *same* incident severity/temporal envelope that
   drives flood depth (`degradations/flood.py`'s `temporal_multiplier`)
   — during a flood, more households draw more power (pumps, backup
   equipment), a real if simplified coupling, and one that ties overload
   risk to the actual incident being simulated rather than an unrelated
   always-on day/night cycle. `update_substation_load` snapshots the
   original static value into `attributes["load_mw_base"]` on first call
   and makes `attributes["load_mw"]` the *current* effective value from
   then on — same convention as `degradations/flood.py` making a road's
   `blockage`/`flood_depth_m` live, tick-updated attributes.
2. **What "overloaded" actually does.** Resolved: symmetric to
   `Simulator.apply_repair` (which this module's constant is
   deliberately scaled to match) — a substation whose post-shed load
   still exceeds `OVERLOAD_THRESHOLD` (dev doc §5.6's exact "95%
   capacity") takes `OVERLOAD_DAMAGE_RATE_PER_TICK` intrinsic-level
   damage each tick it stays overloaded, applied the same way exogenous
   degradation is (`Simulator.step`'s step 1 slot).

`SHED_FRACTION_BY_TIER`: dev doc §3.2 pins only tier 3 ("shed 60% of
non-critical load"); tiers 1-2 are linearly interpolated, a disclosed
assumption, not dev-doc-specified.
"""

from __future__ import annotations

import networkx as nx

from udt.common.models import AssetType, Incident
from udt.incidents.degradations.flood import temporal_multiplier

LOAD_SURGE_FACTOR = 0.5  # load can rise up to 50% above baseline at peak severity x envelope
OVERLOAD_THRESHOLD = 0.95  # dev doc §5.6 exact
DESHED_THRESHOLD = 0.80  # disclosed hysteresis band, prevents tier flapping tick to tick
MAX_SHED_TIER = 3
OVERLOAD_DAMAGE_RATE_PER_TICK = 0.05  # matches DEFAULT_REPAIR_RATE_PER_TICK's order of magnitude
SHED_FRACTION_BY_TIER = {0: 0.0, 1: 0.2, 2: 0.4, 3: 0.6}


def post_shed_ratio(load_mw: float, capacity_mw: float, shed_tier: int) -> float:
    """(load after this tier's shedding) / capacity — the single number
    both the deciding agent and the enactment side need, factored out so
    they can't drift apart."""
    if capacity_mw <= 0:
        return 0.0
    return (load_mw * (1.0 - SHED_FRACTION_BY_TIER[shed_tier])) / capacity_mw


def update_substation_load(
    graph: nx.DiGraph[str], tick: int, incident: Incident, dt_minutes: float
) -> None:
    """Call once per tick, before the deciding agent's own query — same
    ordering requirement as `RoadNetwork.update_for_tick`, and for the
    same reason (the agent needs this tick's current state, not last
    tick's, to decide whether to escalate/de-escalate shedding)."""
    hours_since_onset = (tick - incident.onset_tick) * dt_minutes / 60.0
    envelope = temporal_multiplier(hours_since_onset)
    surge = 1.0 + LOAD_SURGE_FACTOR * incident.severity * envelope

    for asset_id in graph.nodes:
        asset = graph.nodes[asset_id]["asset"]
        if asset.asset_type != AssetType.SUBSTATION:
            continue
        attrs = asset.attributes
        if "load_mw_base" not in attrs:
            attrs["load_mw_base"] = float(attrs.get("load_mw", 0.0))
        attrs["load_mw"] = attrs["load_mw_base"] * surge


def apply_overload_damage(
    graph: nx.DiGraph[str], damage_rate: float = OVERLOAD_DAMAGE_RATE_PER_TICK
) -> dict[str, float]:
    """Returns `{substation_id: intrinsic_level reduction}` for every
    substation still over `OVERLOAD_THRESHOLD` *after* this tick's
    shed_tier is applied — same `{asset_id: reduction}` shape as
    `degradations/flood.py`'s degradation functions, so `Simulator.step`
    can apply it through the exact same `apply_degradation` call."""
    reductions: dict[str, float] = {}
    for asset_id in graph.nodes:
        asset = graph.nodes[asset_id]["asset"]
        if asset.asset_type != AssetType.SUBSTATION:
            continue
        attrs = asset.attributes
        ratio = post_shed_ratio(
            float(attrs.get("load_mw", 0.0)),
            float(attrs.get("capacity_mw", 0.0)),
            int(attrs.get("shed_tier", 0)),
        )
        if ratio > OVERLOAD_THRESHOLD:
            reductions[asset_id] = damage_rate
    return reductions
