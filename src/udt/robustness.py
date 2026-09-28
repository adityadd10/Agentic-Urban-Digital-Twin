"""Evaluation-time twin perturbations for the pre-registered robustness study.

Protocol: `results/protocol/2026-09-28_robustness_study.md` (R-ids from
`docs/parameter_provenance.md`). Each condition overrides runtime parameters
and/or transforms the dependency graph, **for evaluation only**. `apply()` must
be called once, at the start of a fresh process, before any env, simulator or
road network is built, so nothing cached survives from another condition.
The "nominal" condition changes nothing (verified bit-identical against the
reported results before the study runs).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from udt.common.models import Asset, AssetType, DependencyEdge, DependencyGraph

GraphTransform = Callable[[DependencyGraph], DependencyGraph]

# name -> (family, description). Rule-based-only conditions change the agents'
# observation size, so trained policies cannot run on them (protocol §2).
CONDITIONS: dict[str, tuple[str, str]] = {
    "nominal": ("-", "no perturbation"),
    "R1_low": ("R1", "dependency floors x0.5"),
    "R1_high": ("R1", "dependency floors x1.5 (cap 0.95)"),
    "R2_low": ("R2", "buffers x0.5"),
    "R2_high": ("R2", "buffers x1.5"),
    "R3_low": ("R3", "demand shares x0.5"),
    "R3_high": ("R3", "demand shares x2"),
    "R4_low": ("R4", "patient arrival rate x0.5"),
    "R4_high": ("R4", "patient arrival rate x1.5"),
    "R5_low": ("R5", "length of stay 24 h"),
    "R5_high": ("R5", "length of stay 72 h"),
    "R6_low": ("R6", "death-proxy deadline 2 h"),
    "R6_high": ("R6", "death-proxy deadline 6 h"),
    "R8_low": ("R8", "fragility depths x0.75"),
    "R8_high": ("R8", "fragility depths x1.25"),
    "R9_low": ("R9", "road blockage scale 0.27 m"),
    "R9_high": ("R9", "road blockage scale 0.9 m"),
    "R10_low": ("R10", "repair rate x0.5"),
    "R10_high": ("R10", "repair rate x2"),
    "R11_low": ("R11", "load surge 0.25, base load 50%"),
    "R11_high": ("R11", "load surge 1.0, base load 80%"),
    "R13": ("R13", "aggregation: min instead of product"),
    "R14": ("R14", "travel-time noise ±15%"),
    "R7_low": ("R7", "1 ambulance per hospital (rule-based only)"),
    "R7_high": ("R7", "3 ambulances per hospital (rule-based only)"),
    "R12": ("R12", "second synthetic substation (rule-based only)"),
}
RULE_ONLY = {"R7_low", "R7_high", "R12"}


def _map_edges(
    dep: DependencyGraph, fn: Callable[[DependencyEdge, dict[str, Asset]], dict[str, Any]]
) -> DependencyGraph:
    g = dep.model_copy(deep=True)
    assets = {a.asset_id: a for a in g.assets}
    g.edges = [e.model_copy(update=fn(e, assets)) for e in g.edges]
    return g


def _identity(dep: DependencyGraph) -> DependencyGraph:
    return dep


def _floors(factor: float) -> GraphTransform:
    def fn(e: DependencyEdge, assets: dict[str, Asset]) -> dict[str, Any]:
        if assets[e.consumer].asset_type == AssetType.HOSPITAL:
            return {"floor": min(0.95, e.floor * factor)}
        return {}

    return lambda dep: _map_edges(dep, fn)


def _buffers(factor: float) -> GraphTransform:
    return lambda dep: _map_edges(
        dep, lambda e, _a: {"buffer_hours": e.buffer_hours * factor} if e.buffer_hours > 0 else {}
    )


def _demand_shares(factor: float) -> GraphTransform:
    return lambda dep: _map_edges(
        dep, lambda e, _a: {"demand": e.demand * factor} if e.kind != "access" else {}
    )


def _base_load(share: float) -> GraphTransform:
    def t(dep: DependencyGraph) -> DependencyGraph:
        g = dep.model_copy(deep=True)
        for a in g.assets:
            if a.asset_type == AssetType.SUBSTATION:
                a.attributes["load_mw"] = float(a.attributes["capacity_mw"]) * share
        return g

    return t


def _second_substation(dep: DependencyGraph) -> DependencyGraph:
    """R12: a synthetic second substation co-located with the hospital farthest
    from the existing one, supplying that hospital and its water facility
    (disclosed design choice: a hospital-dedicated supply)."""
    g = dep.model_copy(deep=True)
    assets = {a.asset_id: a for a in g.assets}
    s0 = next(a for a in g.assets if a.asset_type == AssetType.SUBSTATION)
    hospitals = [a for a in g.assets if a.asset_type == AssetType.HOSPITAL]

    def dist(a: Asset, b: Asset) -> float:
        (x1, y1), (x2, y2) = a.geometry["coordinates"][:2], b.geometry["coordinates"][:2]
        return math.hypot(x1 - x2, y1 - y2)

    far = max(hospitals, key=lambda h: dist(h, s0))
    new = s0.model_copy(deep=True, update={"asset_id": "S_synth_1", "geometry": far.geometry})
    new.attributes["provenance"] = "synthetic_robustness_R12"
    g.assets.append(new)
    water_ids = {e.supplier for e in g.edges if e.consumer == far.asset_id and e.kind == "water"}
    for e in g.edges:
        if (
            e.kind == "power"
            and e.supplier == s0.asset_id
            and (e.consumer == far.asset_id or e.consumer in water_ids)
        ):
            e.supplier = "S_synth_1"
            e.edge_id = e.edge_id + "_R12"
    assert assets  # graph built from the same assets
    return g


def apply(condition: str) -> GraphTransform:
    """Apply `condition`'s runtime overrides (module attributes) and return
    its dependency-graph transform. Call once, at process start."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    from udt.incidents.degradations import flood
    from udt.twin import cascade, demand, power, road_network, simulator

    family = CONDITIONS[condition][0]
    level = condition.rsplit("_", 1)[-1]

    if family == "R1":
        return _floors(0.5 if level == "low" else 1.5)
    if family == "R2":
        return _buffers(0.5 if level == "low" else 1.5)
    if family == "R3":
        return _demand_shares(0.5 if level == "low" else 2.0)
    if family == "R4":
        demand.BASELINE_ARRIVAL_RATE_PER_BED_PER_HOUR *= 0.5 if level == "low" else 1.5
    elif family == "R5":
        demand.MEAN_LENGTH_OF_STAY_HOURS = 24.0 if level == "low" else 72.0
    elif family == "R6":
        demand.PATIENT_WAIT_DEADLINE_HOURS = 2.0 if level == "low" else 6.0
    elif family == "R8":
        f = 0.75 if level == "low" else 1.25
        flood.FRAGILITY_DEPTHS_M = {
            k: np.asarray(v) * f for k, v in flood.FRAGILITY_DEPTHS_M.items()
        }
    elif family == "R9":
        scale = 0.27 if level == "low" else 0.9
        flood.ROAD_BLOCKAGE_DEPTH_SCALE_M = scale
        road_network.ROAD_BLOCKAGE_DEPTH_SCALE_M = scale
    elif family == "R10":
        rate = simulator.DEFAULT_REPAIR_RATE_PER_TICK * (0.5 if level == "low" else 2.0)
        # The rate is a function default, bound when the module loaded: patch it there.
        simulator.Simulator.step.__kwdefaults__["repair_rate"] = rate
        simulator.Simulator.apply_repair.__defaults__ = (rate,)
    elif family == "R11":
        power.LOAD_SURGE_FACTOR = 0.25 if level == "low" else 1.0
        return _base_load(0.5 if level == "low" else 0.8)
    elif family == "R13":
        cascade.AGGREGATION = "min"
    elif family == "R14":
        road_network.TRAVEL_TIME_NOISE = 0.15
        road_network.TRAVEL_TIME_NOISE_SEED = 0
    elif family == "R7":
        import runner  # experiments/runner.py (on sys.path in the eval script)

        runner.N_AMBULANCES_PER_HOSPITAL = 1 if level == "low" else 3
    elif family == "R12":
        return _second_substation
    return _identity
