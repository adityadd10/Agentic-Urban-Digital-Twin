#!/usr/bin/env python3
"""Feasibility test (prototype only): reactive capacity-restoring levers
(results/twin_v3_gates/prototype_reactive_plan.md). Not part of twin-v3.

  run --part i --n-parts k   conditions 2 (RB-S + rule) and 3 (oracle) on the test scenarios
  report                     compare with condition 1 (RB-S, existing gate runs)

Resources: 1 mobile generator (keeps a hospital's power buffer full while on site) and
1 dewatering pump (suppresses flood damage at a substation/pump while on site). Both start
at S0's road node and travel on the current flooded roads, like the repair crew.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from udt.agents.rbs_v3 import REPAIRABLE, RuleBasedStrongV3  # noqa: E402
from udt.common.models import AssetType  # noqa: E402
from udt.envs.multi_env_v3 import UDTMultiAgentEnvV3  # noqa: E402
from udt.logging.metrics import compute_episode_metrics  # noqa: E402
from udt.twin.graph import dependency_edges_of  # noqa: E402
from udt.twin.simulator import Simulator  # noqa: E402

SUITE = REPO_ROOT / "data" / "scenarios" / "flood_v3"
RUN_DIR = REPO_ROOT / "runs" / "prototype_reactive_levers"
OUT = REPO_ROOT / "results" / "twin_v3_gates"
UNITS_KEY = "proto_units"
DEPOT = "S0"
CHOICE_EVERY, HORIZON = 4, 24  # decisions: re-choose hourly, score over 6 h
N_SCENARIOS = 10


# ------------------------------------------------------------------ twin hooks (picklable)
class DewaterDegradation:
    """Flood damage is suppressed at a facility while the pump is on site."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def __call__(self, tick: int, graph: nx.DiGraph[str]) -> dict[str, float]:
        red = self.inner(tick, graph)
        return {
            k: v for k, v in red.items() if not graph.nodes[k]["asset"].attributes.get("dewatered")
        }

    def depth_m(self, asset: Any, tick: int) -> float:
        return float(self.inner.depth_m(asset, tick))

    @property
    def raster(self) -> Any:
        return self.inner.raster

    @raster.setter
    def raster(self, value: Any) -> None:
        self.inner.raster = value


class SimHook:
    """Wraps `Simulator.step`: moves the units and applies on-site effects each tick."""

    def __init__(self, sim: Simulator) -> None:
        self.sim = sim

    def __call__(self, **kw: Any) -> Any:
        g, sim = self.sim.graph, self.sim
        units = g.graph[UNITS_KEY]
        for unit in units.values():
            if unit["status"] == "travelling":
                unit["remaining_min"] -= sim.dt_minutes
                if unit["remaining_min"] <= 0:
                    unit["status"], unit["node"] = "onsite", unit["target_node"]
        for f in g.nodes:
            a = g.nodes[f]["asset"]
            if a.asset_type in REPAIRABLE:
                a.attributes["dewatered"] = (
                    units["pump"]["status"] == "onsite" and units["pump"]["target"] == f
                )
        gen = units["gen"]
        if gen["status"] == "onsite":
            for e in dependency_edges_of(g, gen["target"]):
                if e.kind == "power":
                    st = sim.edge_states[e.edge_id]
                    st.remaining_hours = st.capacity_hours
        return Simulator.step(sim, **kw)


class ProtoEnv(UDTMultiAgentEnvV3):
    """Twin-v3 env plus the two reactive resources. `actions["resources"] = [gen, pump]`
    with gen in {0 keep, 1..3 hospital} and pump in {0 keep, 1..6 substation/pump}."""

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> Any:
        out = super().reset(seed=seed, options=options)
        assert self.sim is not None
        rn, g = self._road_network, self.sim.graph
        depot = rn.nearest_node(*g.nodes[DEPOT]["asset"].geometry["coordinates"][:2])
        g.graph[UNITS_KEY] = {
            u: {
                "node": depot,
                "status": "idle",
                "target": None,
                "target_node": None,
                "remaining_min": 0.0,
            }
            for u in ("gen", "pump")
        }
        self.sim.step = SimHook(self.sim)  # type: ignore[method-assign]
        self._degradation_fn = DewaterDegradation(self._degradation_fn)
        self.infra_ids = [
            f
            for f in self._facility_ids
            if g.nodes[f]["asset"].asset_type in (AssetType.SUBSTATION, AssetType.WATER)
        ]
        return out

    def assign(self, unit: str, target: str) -> bool:
        assert self.sim is not None
        g, rn = self.sim.graph, self._road_network
        u = g.graph[UNITS_KEY][unit]
        if u["target"] == target:
            return True
        node = rn.nearest_node(*g.nodes[target]["asset"].geometry["coordinates"][:2])
        t = rn.travel_time_minutes(u["node"], node)
        if t is None:
            return False
        u.update(
            {
                "status": "travelling",
                "target": target,
                "target_node": node,
                "remaining_min": float(t),
            }
        )
        return True

    def step(self, actions: dict[str, Any]) -> Any:
        actions = dict(actions)
        res = actions.pop("resources", None)
        if res is not None:
            gen, pump = int(res[0]), int(res[1])
            if gen > 0:
                self.assign("gen", self._hospital_ids[gen - 1])
            if pump > 0:
                self.assign("pump", self.infra_ids[pump - 1])
        return super().step(actions)


# ------------------------------------------------------------------ policies
def _supplier_failing(g: nx.DiGraph[str], h: str) -> bool:
    return any(
        g.nodes[e.supplier]["asset"].functional_level < 0.5
        for e in dependency_edges_of(g, h)
        if e.kind == "power"
    )


def _flooded(env: ProtoEnv, f: str) -> bool:
    assert env.sim is not None
    asset = env.sim.graph.nodes[f]["asset"]
    return env._degradation_fn.depth_m(asset, env.sim.tick) >= 0.1


def resource_rule(env: ProtoEnv) -> list[int]:
    """Condition 2: the simple reactive rule (plan)."""
    assert env.sim is not None
    g = env.sim.graph
    units = g.graph[UNITS_KEY]
    gen = 0
    cur = units["gen"]["target"]
    if not (cur and _supplier_failing(g, cur) and g.nodes[cur]["asset"].intrinsic_level >= 0.5):
        need = [
            h
            for h in env._hospital_ids
            if _supplier_failing(g, h) and g.nodes[h]["asset"].intrinsic_level >= 0.5
        ]

        def buffer_left(h: str) -> float:
            return min(
                (
                    env.sim.edge_states[e.edge_id].remaining_hours
                    for e in dependency_edges_of(g, h)
                    if e.kind == "power"
                ),
                default=99.0,
            )

        if need:
            gen = env._hospital_ids.index(min(need, key=buffer_left)) + 1
    pump = 0
    cur = units["pump"]["target"]
    if not (cur and _flooded(env, cur) and g.nodes[cur]["asset"].intrinsic_level < 1.0):
        cands = [
            f
            for f in env.infra_ids
            if _flooded(env, f) and g.nodes[f]["asset"].intrinsic_level < 1.0
        ]

        def dependants(f: str) -> int:
            return sum(
                1 for d in nx.descendants(g, f) if g.nodes[d]["asset"].asset_type in REPAIRABLE
            )

        if cands:
            pump = env.infra_ids.index(max(cands, key=dependants)) + 1
    return [gen, pump]


def _act(env: ProtoEnv, rbs: RuleBasedStrongV3, res: list[int]) -> dict[str, Any]:
    return {**rbs.act(env), "resources": np.array(res)}


def _snapshot(env: ProtoEnv) -> bytes:
    trace = env.episode_trace
    env.episode_trace = []
    try:
        return env.resume_state()
    finally:
        env.episode_trace = trace


def _score(scratch: ProtoEnv, blob: bytes, choice: list[int], rbs: RuleBasedStrongV3) -> float:
    scratch.load_resume_state(blob)
    assert scratch.sim is not None
    start = scratch.sim._patient_deaths_total
    for i in range(HORIZON):
        if not scratch.agents:
            break
        res = choice if i == 0 else (resource_rule(scratch) if i >= CHOICE_EVERY else [0, 0])
        scratch.step(_act(scratch, rbs, res))
    trace = scratch.episode_trace
    if not trace:
        return 0.0
    return (
        float(trace[-1].patient_deaths_cumulative - start)
        + compute_episode_metrics("x", "x", trace).unmet_patient_hours
    )


def _env() -> ProtoEnv:
    return ProtoEnv(
        processed_dir=REPO_ROOT / "data/processed",
        reward_config=REPO_ROOT / "configs/reward.yaml",
        suite_dir=SUITE,
        scenario_split="train",
    )


def test_scenarios(env: UDTMultiAgentEnvV3) -> list[int]:
    feats = {
        f["scenario_id"]: f["severity_band"]
        for f in json.loads((SUITE / "manifest.json").read_text())["scenario_features"]["train"]
    }
    idx = [i for i, sc in enumerate(env._scenarios) if feats[sc.scenario_id] != "mild"]
    return idx[:N_SCENARIOS]


def run(part: int, n_parts: int) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    env, scratch = _env(), _env()
    rbs = RuleBasedStrongV3.frozen()
    choices = [[g, p] for g, p in itertools.product(range(4), range(7))]
    for j, index in enumerate(test_scenarios(env)):
        if j % n_parts != part:
            continue
        sc = env._scenarios[index]
        out = RUN_DIR / f"{sc.scenario_id}.json"
        if out.exists():
            continue
        result: dict[str, Any] = {"seed": sc.seed}
        # sanity: no resources used -> identical to RB-S
        env.reset(seed=sc.seed, options={"scenario_index": index, "goal": [1, 1, 1, 1]})
        while env.agents:
            env.step(_act(env, rbs, [0, 0]))
        result["rbs_in_proto"] = compute_episode_metrics(
            sc.scenario_id, "rbs", env.episode_trace
        ).model_dump()
        # condition 2: RB-S + simple reactive rule
        env.reset(seed=sc.seed, options={"scenario_index": index, "goal": [1, 1, 1, 1]})
        while env.agents:
            env.step(_act(env, rbs, resource_rule(env)))
        result["rule"] = compute_episode_metrics(
            sc.scenario_id, "rule", env.episode_trace
        ).model_dump()
        # condition 3: oracle over resource placements
        env.reset(seed=sc.seed, options={"scenario_index": index, "goal": [1, 1, 1, 1]})
        d, picks = 0, []
        while env.agents:
            res = [0, 0]
            if d % CHOICE_EVERY == 0:
                blob = _snapshot(env)
                scores = [_score(scratch, blob, c, rbs) for c in choices]
                res = choices[int(np.argmin(scores))]
                picks.append(res)
            env.step(_act(env, rbs, res))
            d += 1
        result["oracle"] = compute_episode_metrics(
            sc.scenario_id, "oracle", env.episode_trace
        ).model_dump()
        result["oracle_picks"] = picks
        out.write_text(json.dumps(result))
        print(f"{sc.scenario_id} done", flush=True)


def report() -> None:
    rbs_all = {
        (r["scenario_id"], r["eval_seed"]): r
        for r in json.loads((REPO_ROOT / "runs/gates_v3/rbs_g1.json").read_text())
    }
    rows = []
    for f in sorted(RUN_DIR.glob("flood_v3_train_*.json")):
        d = json.loads(f.read_text())
        sid = d["rule"]["scenario_id"]
        base = rbs_all[(sid, d["seed"])]
        same = all(
            base[k] == d["rbs_in_proto"][k] for k in base if k not in ("agent_name", "eval_seed")
        )
        rows.append(
            {
                "scenario_id": sid,
                "rbs": base["patient_deaths"],
                "rule": d["rule"]["patient_deaths"],
                "oracle": d["oracle"]["patient_deaths"],
                "rbs_unmet": base["unmet_patient_hours"],
                "rule_unmet": d["rule"]["unmet_patient_hours"],
                "oracle_unmet": d["oracle"]["unmet_patient_hours"],
                "proto_reproduces_rbs": same,
                "picks": d["oracle_picks"],
            }
        )
    m = {
        k: float(np.mean([r[k] for r in rows]))
        for k in ("rbs", "rule", "oracle", "rbs_unmet", "rule_unmet", "oracle_unmet")
    }
    lever = (m["rbs"] - m["rule"]) / m["rbs"] if m["rbs"] else 0.0
    head = (m["rule"] - m["oracle"]) / m["rule"] if m["rule"] else 0.0
    verdict = "HEADROOM (>= 15%)" if head >= 0.15 else "no meaningful headroom (< 15%)"
    picks = Counter(tuple(p) for r in rows for p in r["picks"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "prototype_reactive_results.json").write_text(
        json.dumps(
            {
                "rows": rows,
                "means": m,
                "lever_value_share": lever,
                "headroom_share": head,
                "verdict": verdict,
            },
            indent=2,
        )
    )
    lines = [
        "# Feasibility test: reactive levers (prototype only)",
        "",
        "Plan: prototype_reactive_plan.md. 10 moderate/severe train scenarios, seed k = 0.",
        "",
        "Prototype with no resources reproduces RB-S exactly: "
        f"{all(r['proto_reproduces_rbs'] for r in rows)}",
        "",
        "| Condition | Mean deaths | Mean unmet p-h |",
        "|---|---|---|",
        f"| 1. RB-S (no resources) | {m['rbs']:.2f} | {m['rbs_unmet']:.1f} |",
        f"| 2. RB-S + reactive rule | {m['rule']:.2f} | {m['rule_unmet']:.1f} |",
        f"| 3. Oracle placement | {m['oracle']:.2f} | {m['oracle_unmet']:.1f} |",
        "",
        f"Lever value (1 -> 2): {100 * lever:+.1f}% deaths. Headroom (2 -> 3): {100 * head:+.1f}%.",
        "",
        f"**Reading (fixed in the plan): {verdict}**",
        "",
        "| Scenario | RB-S | Rule | Oracle |",
        "|---|---|---|---|",
    ]
    lines += [f"| {r['scenario_id']} | {r['rbs']} | {r['rule']} | {r['oracle']} |" for r in rows]
    lines += [
        "",
        "Oracle hourly picks [generator, pump] (0 = keep): "
        + ", ".join(f"{k}: {v}" for k, v in picks.most_common(6)),
    ]
    (OUT / "prototype_reactive_results.md").write_text("\n".join(lines) + "\n")
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
