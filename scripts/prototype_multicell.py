#!/usr/bin/env python3
"""Feasibility test 3 (prototype only): two-cell rainfall + acuity
(results/twin_v3_gates/prototype_multicell_plan.md). Not part of twin-v3.

  run --part i --n-parts k   RB-S, acuity-aware rule, oracle (45 strategies)
  report                     -> results/twin_v3_gates/prototype_multicell_results.*

A second storm cell (deterministic from the scenario seed, generator's own ranges,
onset 2-4 h after the first) adds its depth everywhere: facilities and dependency-graph
roads (`DualFlood`) and the routing network (`DualRoadNetwork`). Acuity mechanics and
policies are reused from `prototype_acuity.py`.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import Point, shape

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import prototype_acuity as acuity  # noqa: E402
import udt.envs.multi_env_v3 as env_mod  # noqa: E402
from udt.agents.rbs_v3 import RuleBasedStrongV3  # noqa: E402
from udt.common.models import Incident  # noqa: E402
from udt.incidents.degradations.flood import (  # noqa: E402
    MAX_DEPTH_AT_SEVERITY_1_M,
    ROAD_BLOCKAGE_DEPTH_SCALE_M,
    FloodDegradation,
    footprint_weight,
    incident_envelope,
)
from udt.logging.metrics import compute_episode_metrics  # noqa: E402
from udt.scenarios.generator import (  # noqa: E402
    DEFAULT_SEVERITY_RANGE,
    FOOTPRINT_SIGMA_RANGE_M,
    GROWTH_HOURS_RANGE,
    HOLD_HOURS_RANGE,
    RECEDE_HOURS_RANGE,
)
from udt.twin.road_network import IMPASSABLE_BLOCKAGE, RoadNetwork  # noqa: E402

RUN_DIR = REPO_ROOT / "runs" / "prototype_multicell"
OUT = REPO_ROOT / "results" / "twin_v3_gates"
ONSET_OFFSET_HOURS = (2.0, 4.0)


def second_cell(incident: Incident, seed: int, ward: Any) -> Incident:
    rng = np.random.default_rng([seed, 7])
    minx, miny, maxx, maxy = ward.bounds
    lon, lat = (minx + maxx) / 2, (miny + maxy) / 2
    for _ in range(200):
        x, y = float(rng.uniform(minx, maxx)), float(rng.uniform(miny, maxy))
        if ward.contains(Point(x, y)):
            lon, lat = x, y
            break
    onset = incident.onset_tick + int(round(float(rng.uniform(*ONSET_OFFSET_HOURS)) * 12))
    return Incident(
        incident_id=f"{incident.incident_id}_cell2",
        type="flood",
        location={"type": "Point", "coordinates": [lon, lat]},
        onset_tick=onset,
        severity=float(rng.uniform(*DEFAULT_SEVERITY_RANGE)),
        directly_affected_assets=[],
        profile={
            "growth_hours": float(rng.uniform(*GROWTH_HOURS_RANGE)),
            "hold_hours": float(rng.uniform(*HOLD_HOURS_RANGE)),
            "recede_hours": float(rng.uniform(*RECEDE_HOURS_RANGE)),
            "footprint": {
                "lon": lon,
                "lat": lat,
                "sigma_m": float(rng.uniform(*FOOTPRINT_SIGMA_RANGE_M)),
            },
        },
    )


class DualFlood(FloodDegradation):
    """Cell 1's fragility and timing, depth = cell 1 + cell 2 everywhere."""

    cell2_fd: FloodDegradation | None = None

    @property  # type: ignore[override]
    def raster(self) -> Any:
        return self._raster_ref

    @raster.setter
    def raster(self, value: Any) -> None:
        self._raster_ref = value
        if self.cell2_fd is not None:
            self.cell2_fd.raster = value

    def depth_m(self, asset: Any, tick: int) -> float:
        d = super().depth_m(asset, tick)
        return d + (self.cell2_fd.depth_m(asset, tick) if self.cell2_fd is not None else 0.0)


class DualRoadNetwork(RoadNetwork):
    """Routing depth = susceptibility x (s1 x footprint1 + s2 x footprint2)."""

    cell2: Incident | None = None
    _fp2_id: str | None = None
    _s2: float = 0.0

    def _set_fp2(self) -> None:
        for _u, _v, data in self.graph.edges(data=True):
            has_xy = "lon" in data and "lat" in data
            data["footprint2"] = (
                footprint_weight(self.cell2, data["lon"], data["lat"])
                if has_xy and self.cell2
                else 0.0
            )
        self._fp2_id = self.cell2.incident_id if self.cell2 else None
        self._edge_arrays = None

    def update_for_tick(self, incident: Incident, tick: int, dt_minutes: float) -> None:
        super().update_for_tick(incident, tick, dt_minutes)
        if self.cell2 is not None and self.cell2.incident_id != self._fp2_id:
            self._set_fp2()
        if self.cell2 is not None:
            hours = (tick - self.cell2.onset_tick) * dt_minutes / 60.0
            self._s2 = (
                self.cell2.severity
                * incident_envelope(self.cell2, hours)
                * MAX_DEPTH_AT_SEVERITY_1_M
            )
        else:
            self._s2 = 0.0

    def _ensure_edge_arrays(self) -> dict[str, Any]:
        arrays = super()._ensure_edge_arrays()
        if "fp2" not in arrays:
            arrays["fp2"] = np.array(
                [float(d.get("footprint2", 0.0)) for _u, _v, d in self.graph.edges(data=True)]
            )
            self._weights_scale = None
        return arrays

    def _edge_weights(self) -> list[float]:
        arrays = self._ensure_edge_arrays()
        key = (self._current_depth_scale, self._s2)
        if self._weights_scale != key:
            depth = arrays["sus"] * (
                self._current_depth_scale * arrays["fp"] + self._s2 * arrays["fp2"]
            )
            blockage = np.maximum(0.0, np.minimum(1.0, depth / ROAD_BLOCKAGE_DEPTH_SCALE_M))
            weights = arrays["base"] / np.maximum(0.05, 1.0 - blockage)
            self._weights = np.where(blockage >= IMPASSABLE_BLOCKAGE, np.inf, weights).tolist()
            self._weights_scale = key  # type: ignore[assignment]
            self._route_cache.clear()
        return self._weights

    def shortest_path_max_depth(self, source: str, target: str) -> float | None:
        route = self._route(source, target)
        if route is None:
            return None
        worst = 0.0
        for u, v in zip(route[1][:-1], route[1][1:], strict=True):
            d = self.graph[u][v]
            depth = float(d.get("susceptibility", 0.0)) * (
                self._current_depth_scale * float(d.get("footprint", 1.0))
                + self._s2 * float(d.get("footprint2", 0.0))
            )
            if min(1.0, depth / ROAD_BLOCKAGE_DEPTH_SCALE_M) >= IMPASSABLE_BLOCKAGE:
                return None
            worst = max(worst, depth)
        return worst

    def mutable_state(self) -> dict[str, Any]:
        return {**super().mutable_state(), "cell2": self.cell2, "s2": self._s2}

    def load_mutable_state(self, state: dict[str, Any], incident: Incident | None) -> None:
        super().load_mutable_state(
            {k: v for k, v in state.items() if k not in ("cell2", "s2")}, incident
        )
        self.cell2, self._s2 = state["cell2"], float(state["s2"])
        self._set_fp2()


_orig_generate = env_mod.generate_requests


class MultiCellEnv(acuity.AcuityEnv):
    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self._road_network.__class__ = DualRoadNetwork
        self._ward = shape(self._ward_boundary["features"][0]["geometry"])

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> Any:
        index = int((options or {})["scenario_index"])
        sc = self._scenarios[index]
        self.cell2 = second_cell(sc.incident, sc.seed, self._ward)
        rn = self._road_network
        rn.cell2 = self.cell2  # type: ignore[attr-defined]
        out = super().reset(seed=seed, options=options)
        fd = DualFlood(self.incident, self._raster)
        fd.cell2_fd = FloodDegradation(self.cell2, self._raster)
        self._degradation_fn = fd
        assert self.sim is not None
        self.sim.graph.graph["_cell2"] = self.cell2
        rn.update_for_tick(self.incident, self.sim.tick, self.sim.dt_minutes)
        self._observations()
        return out


def generate_both_cells(
    graph: Any, tick: int, dt_hours: float, incident: Incident, *a: Any, **kw: Any
) -> None:
    acuity.generate_with_acuity(graph, tick, dt_hours, incident, *a, **kw)
    cell2 = graph.graph.get("_cell2")
    if cell2 is not None and tick >= cell2.onset_tick:
        acuity.generate_with_acuity(graph, tick, dt_hours, cell2, *a, **kw)


def install() -> None:
    acuity.install()
    env_mod.generate_requests = generate_both_cells  # type: ignore[assignment]


def _env() -> MultiCellEnv:
    return MultiCellEnv(
        processed_dir=REPO_ROOT / "data/processed",
        reward_config=REPO_ROOT / "configs/reward.yaml",
        suite_dir=acuity.SUITE,
        scenario_split="train",
    )


def run(part: int, n_parts: int) -> None:
    install()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    env, scratch = _env(), _env()
    params = RuleBasedStrongV3.frozen().params
    rbs = RuleBasedStrongV3(params)
    rule = acuity.Composite(acuity.ICUEvacHealth, RuleBasedStrongV3, acuity.AcuityTransport, params)
    cands = [
        acuity.Composite(h, p, t, params)
        for h, p, t in itertools.product(acuity.HEALTH, acuity.POWER, acuity.TRANSPORT)
    ]
    for j, index in enumerate(acuity.test_scenarios(env)):
        if j % n_parts != part:
            continue
        sc = env._scenarios[index]
        out = RUN_DIR / f"{sc.scenario_id}.json"
        if out.exists():
            continue
        res: dict[str, Any] = {
            "seed": sc.seed,
            "rbs": acuity._episode(env, index, sc.seed, rbs),
            "rule": acuity._episode(env, index, sc.seed, rule),
        }
        env.reset(seed=sc.seed, options={"scenario_index": index, "goal": [1, 1, 1, 1]})
        chosen, picks, d = rule, [], 0
        while env.agents:
            if d % acuity.CHOICE_EVERY == 0:
                blob = acuity._snapshot(env)
                scores = [acuity._score(scratch, blob, c, rule) for c in cands]
                chosen = cands[int(np.argmin(scores))]
                picks.append(chosen.name)
            env.step(chosen.act(env))
            d += 1
        res["oracle"] = compute_episode_metrics(
            sc.scenario_id, "oracle", env.episode_trace
        ).model_dump()
        res["picks"] = picks
        res["cell2"] = env.cell2.model_dump(mode="json")
        out.write_text(json.dumps(res))
        print(f"{sc.scenario_id} done", flush=True)


def report() -> None:
    rows = []
    for f in sorted(RUN_DIR.glob("flood_v3_train_*.json")):
        d = json.loads(f.read_text())
        rows.append(
            {
                "scenario_id": d["rbs"]["scenario_id"],
                **{k: d[k]["patient_deaths"] for k in ("rbs", "rule", "oracle")},
                **{f"{k}_unmet": d[k]["unmet_patient_hours"] for k in ("rbs", "rule", "oracle")},
                "picks": d["picks"],
            }
        )
    m = {
        k: float(np.mean([r[k] for r in rows]))
        for k in ("rbs", "rule", "oracle", "rbs_unmet", "rule_unmet", "oracle_unmet")
    }
    best = min(m["rbs"], m["rule"])
    head = (best - m["oracle"]) / best if best else 0.0
    verdict = "HEADROOM (>= 15%)" if head >= 0.15 else "no meaningful headroom (< 15%)"
    picks = Counter(p for r in rows for p in r["picks"])
    (OUT / "prototype_multicell_results.json").write_text(
        json.dumps({"rows": rows, "means": m, "headroom_share": head, "verdict": verdict}, indent=2)
    )
    lines = [
        "# Feasibility test 3: two-cell rainfall + acuity (prototype only)",
        "",
        "Plan: prototype_multicell_plan.md. 10 moderate/severe train scenarios, seed k = 0.",
        "",
        "| Condition | Mean deaths | Mean unmet p-h |",
        "|---|---|---|",
        f"| 1. RB-S (acuity-blind) | {m['rbs']:.2f} | {m['rbs_unmet']:.1f} |",
        f"| 2. Simple acuity-aware rule | {m['rule']:.2f} | {m['rule_unmet']:.1f} |",
        f"| 3. Oracle (45 strategies) | {m['oracle']:.2f} | {m['oracle_unmet']:.1f} |",
        "",
        f"Headroom (best of 1-2 -> 3): {100 * head:+.1f}%",
        "",
        f"**Reading (fixed in the plan): {verdict}**",
        "",
        "| Scenario | RB-S | Rule | Oracle |",
        "|---|---|---|---|",
    ]
    lines += [f"| {r['scenario_id']} | {r['rbs']} | {r['rule']} | {r['oracle']} |" for r in rows]
    lines += [
        "",
        "Most common oracle choices: " + "; ".join(f"{k} ({v})" for k, v in picks.most_common(5)),
    ]
    (OUT / "prototype_multicell_results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--part", type=int, required=True)
    r.add_argument("--n-parts", type=int, required=True)
    sub.add_parser("report")
    a = ap.parse_args()
    run(a.part, a.n_parts) if a.cmd == "run" else report()
