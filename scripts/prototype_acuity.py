#!/usr/bin/env python3
"""Feasibility test 2 (prototype only): patient acuity
(results/twin_v3_gates/prototype_acuity_plan.md). Not part of twin-v3.

  run --part i --n-parts k   conditions 1 (RB-S), 2 (acuity-aware rule), 3 (oracle)
  report                     -> results/twin_v3_gates/prototype_acuity_results.*

Acuity is encoded through the existing waiting clocks, deterministically (no extra
random draws): a critical patient or call gets a clock started CRITICAL_SHIFT earlier,
so the existing 4 h deadline leaves 1 h. Hospitals admit the longest-waiting first, which
puts critical patients first (triage). Casualties' clocks start at the call (time to care).
ICU patients in a hospital without power die at ICU_DEATH_RATE_PER_H (MIOT Chennai 2015).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import udt.envs.multi_env_v3 as env_mod  # noqa: E402
from udt.agents.gate_heuristics_v3 import (  # noqa: E402
    CallsFirst,
    Coordinated,
    NearestReachableRepair,
    TransfersFirst,
    TransportFixed,
    nearest_functioning_destination,
)
from udt.agents.rbs_v3 import RuleBasedStrongV3  # noqa: E402
from udt.envs.multi_env_v3 import (  # noqa: E402
    AGENT_HEALTH,
    AGENT_POWER,
    AGENT_TRANSPORT,
    N_JOB_SLOTS,
    TRANSFER_OPTIONS,
    UDTMultiAgentEnvV3,
    job_priority_order,
)
from udt.logging.metrics import compute_episode_metrics  # noqa: E402
from udt.twin import ambulances, simulator  # noqa: E402
from udt.twin.graph import dependency_edges_of  # noqa: E402

SUITE = REPO_ROOT / "data" / "scenarios" / "flood_v3"
RUN_DIR = REPO_ROOT / "runs" / "prototype_acuity"
OUT = REPO_ROOT / "results" / "twin_v3_gates"
CRITICAL_ARRIVAL_PCT = 20  # ESI 1-2 share (handbook 21-33%, NHAMCS ~11%)
CRITICAL_CALL_PCT = 30  # assumption
CRITICAL_SHIFT = 36  # ticks (3 h): 4 h deadline - 3 h = 1 h for critical patients
ICU_DEATH_RATE_PER_H = 0.005  # MIOT Chennai 2015: 18 of 75 ventilated over ~48 h
CHOICE_EVERY, HORIZON, N_SCENARIOS = 4, 24, 10


def _pct(*key: object) -> int:
    return zlib.crc32("|".join(map(str, key)).encode()) % 100


# --------------------------------------------------------------------- physics patches
def _powerless(graph: nx.DiGraph[str], h: str) -> bool:
    asset = graph.nodes[h]["asset"]
    if asset.intrinsic_level < 0.5:
        return True  # flooded hospital: generators lost too (MIOT)
    states = graph.graph.get("_proto_edge_states", {})
    return any(
        e.kind == "power" and e.edge_id in states and states[e.edge_id].remaining_hours <= 0
        for e in dependency_edges_of(graph, h)
    )


_orig_consume = simulator.consume_demand


def consume_with_acuity(
    graph: nx.DiGraph[str], tick: int, dt_hours: float, rng: Any, **kw: Any
) -> dict[str, int]:
    graph.graph["_proto_now"] = tick
    hospitals = [n for n in graph.nodes if graph.nodes[n]["asset"].asset_type.value == "hospital"]
    for h in hospitals:  # triage: longest-waiting (critical) first
        a = graph.nodes[h]["asset"].attributes
        q = list(a.get("queue_arrivals", []))
        c = list(a.get("queue_call_ticks", [None] * len(q)))
        if len(c) != len(q):
            c = [None] * len(q)
        order = sorted(range(len(q)), key=lambda i: q[i])
        a["queue_arrivals"] = [q[i] for i in order]
        a["queue_call_ticks"] = [c[i] for i in order]
    deaths = _orig_consume(graph, tick, dt_hours, rng, **kw)
    for h in hospitals:
        a = graph.nodes[h]["asset"].attributes
        q = list(a.get("queue_arrivals", []))
        j = 0
        for i, t in enumerate(q):  # this tick's new walk-ins: mark critical ones
            if t == tick:
                if _pct(h, tick, j) < CRITICAL_ARRIVAL_PCT:
                    q[i] = tick - CRITICAL_SHIFT
                j += 1
        a["queue_arrivals"] = q
        if _powerless(graph, h):  # ICU deaths without power
            acc = (
                float(a.get("_icu_acc", 0.0))
                + ICU_DEATH_RATE_PER_H * int(a.get("icu_occupied", 0)) * dt_hours
            )
            n = int(math.floor(acc))
            a["_icu_acc"] = acc - n
            if n > 0:
                a["icu_occupied"] = int(a.get("icu_occupied", 0)) - n
                a["beds_occupied"] = max(0, int(a.get("beds_occupied", 0)) - n)
                deaths[h] = deaths.get(h, 0) + n
    return deaths


_orig_generate = env_mod.generate_requests


def generate_with_acuity(graph: nx.DiGraph[str], tick: int, *args: Any, **kw: Any) -> None:
    before = {r["request_id"] for r in graph.graph.get("pending_requests", [])}
    _orig_generate(graph, tick, *args, **kw)
    crit = graph.graph.setdefault("critical_calls", [])
    for r in graph.graph.get("pending_requests", []):
        if r["request_id"] not in before and _pct(r["request_id"]) < CRITICAL_CALL_PCT:
            r["requested_at_tick"] -= CRITICAL_SHIFT  # 1 h golden hour under the 4 h rule
            crit.append(r["request_id"])


_orig_deliver = ambulances._deliver_casualty


def deliver_from_call(
    graph: nx.DiGraph[str],
    hospital_id: str,
    tick: int,
    wait_start_tick: int | None = None,
    call_tick: int | None = None,
) -> None:
    if wait_start_tick is None and call_tick is not None:
        wait_start_tick = call_tick  # time to care: the clock started at the call
    _orig_deliver(graph, hospital_id, tick, wait_start_tick, call_tick)


_orig_take = ambulances._take_patient_for_transfer


def take_icu_first(graph: nx.DiGraph[str], hospital_id: str) -> tuple[bool, int | None, int | None]:
    a = graph.nodes[hospital_id]["asset"].attributes
    if _powerless(graph, hospital_id) and int(a.get("icu_occupied", 0)) > 0:
        a["icu_occupied"] = int(a["icu_occupied"]) - 1
        a["beds_occupied"] = max(0, int(a.get("beds_occupied", 0)) - 1)
        now = int(graph.graph.get("_proto_now", 0))
        return True, now - CRITICAL_SHIFT, None  # moves as a critical patient
    return _orig_take(graph, hospital_id)


def install() -> None:
    simulator.consume_demand = consume_with_acuity  # type: ignore[assignment]
    env_mod.generate_requests = generate_with_acuity  # type: ignore[assignment]
    ambulances._deliver_casualty = deliver_from_call  # type: ignore[assignment]
    ambulances._take_patient_for_transfer = take_icu_first  # type: ignore[assignment]


class AcuityEnv(UDTMultiAgentEnvV3):
    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> Any:
        out = super().reset(seed=seed, options=options)
        assert self.sim is not None
        self.sim.graph.graph["_proto_edge_states"] = self.sim.edge_states
        return out


# --------------------------------------------------------------------- policies
class ICUEvacHealth(RuleBasedStrongV3):
    def health_action(self, env: UDTMultiAgentEnvV3) -> Any:
        assert env.sim is not None
        out = super().health_action(env)
        g = env.sim.graph
        for i, h in enumerate(env._hospital_ids):
            pending = any(j["from_hospital_id"] == h for j in g.graph.get("transfer_requests", []))
            if (
                _powerless(g, h)
                and int(g.nodes[h]["asset"].attributes.get("icu_occupied", 0)) > 0
                and not pending
            ):
                out[3 * i] = TRANSFER_OPTIONS.index((5, "urgent"))
        return out


class AcuityTransport(RuleBasedStrongV3):
    def transport_action(self, env: UDTMultiAgentEnvV3) -> Any:
        assert env.sim is not None
        g = env.sim.graph
        base = super().transport_action(env)
        crit = set(g.graph.get("critical_calls", []))
        jobs = {j["request_id"]: j for j in job_priority_order(g, env.sim.tick, env.sim.dt_hours)}
        rn = env._road_network
        nodes = {
            h: rn.nearest_node(*g.nodes[h]["asset"].geometry["coordinates"][:2])
            for h in env._hospital_ids
        }
        for slot, job_id in enumerate(env._slots[:N_JOB_SLOTS]):
            if job_id in crit and job_id in jobs:
                dest = nearest_functioning_destination(env, jobs[job_id], nodes)
                base[slot] = 0 if dest is None else env._hospital_ids.index(dest) + 1
        return base


class Composite:
    def __init__(self, h: type, p: type, t: type, params: Any) -> None:
        self.h, self.p, self.t = h(params), p(params), t(params)
        self.name = f"{h.__name__}|{p.__name__}|{t.__name__}"

    def act(self, env: UDTMultiAgentEnvV3) -> dict[str, Any]:
        return {
            AGENT_HEALTH: self.h.health_action(env),
            AGENT_POWER: self.p.power_action(env),
            AGENT_TRANSPORT: self.t.transport_action(env),
        }


HEALTH = (RuleBasedStrongV3, ICUEvacHealth, Coordinated)
POWER = (RuleBasedStrongV3, NearestReachableRepair, Coordinated)
TRANSPORT = (RuleBasedStrongV3, AcuityTransport, TransportFixed, CallsFirst, TransfersFirst)


# --------------------------------------------------------------------- runs
def _env() -> AcuityEnv:
    return AcuityEnv(
        processed_dir=REPO_ROOT / "data/processed",
        reward_config=REPO_ROOT / "configs/reward.yaml",
        suite_dir=SUITE,
        scenario_split="train",
    )


def _snapshot(env: UDTMultiAgentEnvV3) -> bytes:
    trace = env.episode_trace
    env.episode_trace = []
    try:
        return env.resume_state()
    finally:
        env.episode_trace = trace


def _score(scratch: UDTMultiAgentEnvV3, blob: bytes, cand: Composite, rule: Composite) -> float:
    scratch.load_resume_state(blob)
    assert scratch.sim is not None
    start = scratch.sim._patient_deaths_total
    for i in range(HORIZON):
        if not scratch.agents:
            break
        scratch.step((cand if i < CHOICE_EVERY else rule).act(scratch))
    tr = scratch.episode_trace
    return (
        0.0
        if not tr
        else float(tr[-1].patient_deaths_cumulative - start)
        + compute_episode_metrics("x", "x", tr).unmet_patient_hours
    )


def _episode(env: AcuityEnv, index: int, seed: int, policy: Any) -> dict[str, Any]:
    env.reset(seed=seed, options={"scenario_index": index, "goal": [1, 1, 1, 1]})
    while env.agents:
        env.step(policy.act(env))
    return compute_episode_metrics(
        env._scenarios[index].scenario_id, "x", env.episode_trace
    ).model_dump()


def test_scenarios(env: UDTMultiAgentEnvV3) -> list[int]:
    feats = {
        f["scenario_id"]: f["severity_band"]
        for f in json.loads((SUITE / "manifest.json").read_text())["scenario_features"]["train"]
    }
    return [i for i, sc in enumerate(env._scenarios) if feats[sc.scenario_id] != "mild"][
        :N_SCENARIOS
    ]


def run(part: int, n_parts: int) -> None:
    install()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    env, scratch = _env(), _env()
    params = RuleBasedStrongV3.frozen().params
    rbs = RuleBasedStrongV3(params)
    rule = Composite(ICUEvacHealth, RuleBasedStrongV3, AcuityTransport, params)
    cands = [Composite(h, p, t, params) for h, p, t in itertools.product(HEALTH, POWER, TRANSPORT)]
    for j, index in enumerate(test_scenarios(env)):
        if j % n_parts != part:
            continue
        sc = env._scenarios[index]
        out = RUN_DIR / f"{sc.scenario_id}.json"
        if out.exists():
            continue
        res: dict[str, Any] = {
            "seed": sc.seed,
            "rbs": _episode(env, index, sc.seed, rbs),
            "rule": _episode(env, index, sc.seed, rule),
        }
        env.reset(seed=sc.seed, options={"scenario_index": index, "goal": [1, 1, 1, 1]})
        chosen, picks, d = rule, [], 0
        while env.agents:
            if d % CHOICE_EVERY == 0:
                blob = _snapshot(env)
                scores = [_score(scratch, blob, c, rule) for c in cands]
                chosen = cands[int(np.argmin(scores))]
                picks.append(chosen.name)
            env.step(chosen.act(env))
            d += 1
        res["oracle"] = compute_episode_metrics(
            sc.scenario_id, "oracle", env.episode_trace
        ).model_dump()
        res["picks"] = picks
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
    best_simple = min(m["rbs"], m["rule"])
    head = (best_simple - m["oracle"]) / best_simple if best_simple else 0.0
    verdict = "HEADROOM (>= 15%)" if head >= 0.15 else "no meaningful headroom (< 15%)"
    picks = Counter(p for r in rows for p in r["picks"])
    (OUT / "prototype_acuity_results.json").write_text(
        json.dumps({"rows": rows, "means": m, "headroom_share": head, "verdict": verdict}, indent=2)
    )
    lines = [
        "# Feasibility test 2: patient acuity (prototype only)",
        "",
        "Plan: prototype_acuity_plan.md. 10 moderate/severe train scenarios, seed k = 0.",
        "",
        "| Condition | Mean deaths | Mean unmet p-h |",
        "|---|---|---|",
        f"| 1. RB-S (acuity-blind) | {m['rbs']:.2f} | {m['rbs_unmet']:.1f} |",
        f"| 2. Simple acuity-aware rule | {m['rule']:.2f} | {m['rule_unmet']:.1f} |",
        f"| 3. Oracle (45 strategies, perfect foresight) | {m['oracle']:.2f} "
        f"| {m['oracle_unmet']:.1f} |",
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
    (OUT / "prototype_acuity_results.md").write_text("\n".join(lines) + "\n")
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
