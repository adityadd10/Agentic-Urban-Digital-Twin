#!/usr/bin/env python3
"""Exploratory diagnostic after gates round 0 (results/twin_v3_gates/diagnostic_plan_oracle.md).

  run --part i --n-parts k   D1 oracle ceiling + D2 physics floor for train scenarios i, i+k, ...
  report                     compare with RB-S (seed k = 0)
                             -> results/twin_v3_gates/diagnostic_oracle.*

Cannot change any gate verdict. D1 re-chooses, every hour, among 30 combined sector strategies,
scoring each on an exact copy of the env (true future: calls, arrivals, fragility) over 6 h
(candidate 1 h, then RB-S 5 h). D2 is RB-S with no facility damage (roads still flood).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from udt.agents.gate_heuristics_v3 import (  # noqa: E402
    CallsFirst,
    Coordinated,
    NearestFunctioningDestination,
    NearestReachableRepair,
    TransfersFirst,
    TransportFixed,
)
from udt.agents.rbs_v3 import RuleBasedStrongV3  # noqa: E402
from udt.envs.multi_env_v3 import (  # noqa: E402
    AGENT_HEALTH,
    AGENT_POWER,
    AGENT_TRANSPORT,
    UDTMultiAgentEnvV3,
)
from udt.logging.metrics import compute_episode_metrics  # noqa: E402

SUITE = REPO_ROOT / "data" / "scenarios" / "flood_v3"
RUN_DIR = REPO_ROOT / "runs" / "diagnostic_oracle_v3"
OUT = REPO_ROOT / "results" / "twin_v3_gates"
CHOICE_EVERY = 4  # decisions (1 h)
HORIZON = 24  # decisions (6 h): candidate for CHOICE_EVERY, then RB-S

HEALTH = (RuleBasedStrongV3, Coordinated)
POWER = (RuleBasedStrongV3, NearestReachableRepair, Coordinated)
TRANSPORT = (
    RuleBasedStrongV3,
    TransportFixed,
    NearestFunctioningDestination,
    CallsFirst,
    TransfersFirst,
)


class Composite:
    """Health, power and transport rules taken from (possibly) different gate conditions."""

    def __init__(self, h: type, p: type, t: type, params: Any) -> None:
        self.h, self.p, self.t = h(params), p(params), t(params)
        self.name = f"{h.__name__}|{p.__name__}|{t.__name__}"

    def act(self, env: UDTMultiAgentEnvV3) -> dict[str, Any]:
        return {
            AGENT_HEALTH: self.h.health_action(env),
            AGENT_POWER: self.p.power_action(env),
            AGENT_TRANSPORT: self.t.transport_action(env),
        }


def _env() -> UDTMultiAgentEnvV3:
    return UDTMultiAgentEnvV3(
        processed_dir=REPO_ROOT / "data/processed",
        reward_config=REPO_ROOT / "configs/reward.yaml",
        suite_dir=SUITE,
        scenario_split="train",
    )


def _snapshot(env: UDTMultiAgentEnvV3) -> bytes:
    trace = env.episode_trace
    env.episode_trace = []  # keep the copy small; scores use the copy's own new trace
    try:
        return env.resume_state()
    finally:
        env.episode_trace = trace


def _score(
    scratch: UDTMultiAgentEnvV3, blob: bytes, cand: Composite, rbs: RuleBasedStrongV3
) -> float:
    scratch.load_resume_state(blob)
    assert scratch.sim is not None
    start_deaths = scratch.sim._patient_deaths_total
    for i in range(HORIZON):
        if not scratch.agents:
            break
        scratch.step((cand if i < CHOICE_EVERY else rbs).act(scratch))
    trace = scratch.episode_trace
    if not trace:
        return 0.0
    m = compute_episode_metrics("x", "x", trace)
    return float(trace[-1].patient_deaths_cumulative - start_deaths) + m.unmet_patient_hours


class _NoFacilityDamage:
    """D2: the flood still updates roads; facilities are never damaged."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def __call__(self, tick: int, graph: Any) -> dict[str, float]:
        self.inner(tick, graph)
        return {}

    def depth_m(self, asset: Any, tick: int) -> float:
        return float(self.inner.depth_m(asset, tick))

    @property
    def raster(self) -> Any:
        return self.inner.raster

    @raster.setter
    def raster(self, value: Any) -> None:
        self.inner.raster = value


def run(part: int, n_parts: int) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    env, scratch = _env(), _env()
    params = RuleBasedStrongV3.frozen().params
    rbs = RuleBasedStrongV3(params)
    candidates = [
        Composite(h, p, t, params) for h, p, t in itertools.product(HEALTH, POWER, TRANSPORT)
    ]
    for index, sc in enumerate(env._scenarios):
        if index % n_parts != part:
            continue
        out = RUN_DIR / f"{sc.scenario_id}.json"
        if out.exists():
            continue
        # D1 oracle ceiling
        env.reset(seed=sc.seed, options={"scenario_index": index, "goal": [1, 1, 1, 1]})
        chosen, choices, d = candidates[0], [], 0
        while env.agents:
            if d % CHOICE_EVERY == 0:
                blob = _snapshot(env)
                scores = [_score(scratch, blob, c, rbs) for c in candidates]
                chosen = candidates[int(np.argmin(scores))]
                choices.append(chosen.name)
            env.step(chosen.act(env))
            d += 1
        oracle = compute_episode_metrics(sc.scenario_id, "oracle", env.episode_trace).model_dump()
        # D2 physics floor
        env.reset(seed=sc.seed, options={"scenario_index": index, "goal": [1, 1, 1, 1]})
        env._degradation_fn = _NoFacilityDamage(env._degradation_fn)
        while env.agents:
            env.step(rbs.act(env))
        floor = compute_episode_metrics(sc.scenario_id, "floor", env.episode_trace).model_dump()
        out.write_text(
            json.dumps({"oracle": oracle, "floor": floor, "choices": choices, "seed": sc.seed})
        )
        print(f"{sc.scenario_id} done", flush=True)


def report() -> None:
    rbs_all = {
        (r["scenario_id"], r["eval_seed"]): r
        for r in json.loads((REPO_ROOT / "runs/gates_v3/rbs_g1.json").read_text())
    }
    rows = []
    for f in sorted(RUN_DIR.glob("flood_v3_train_*.json")):
        d = json.loads(f.read_text())
        sid = d["oracle"]["scenario_id"]
        rbs = rbs_all[(sid, d["seed"])]
        rows.append(
            {
                "scenario_id": sid,
                "rbs": rbs["patient_deaths"],
                "oracle": d["oracle"]["patient_deaths"],
                "floor": d["floor"]["patient_deaths"],
                "rbs_unmet": rbs["unmet_patient_hours"],
                "oracle_unmet": d["oracle"]["unmet_patient_hours"],
                "floor_unmet": d["floor"]["unmet_patient_hours"],
                "choices": d["choices"],
            }
        )
    n = len(rows)
    rbs_m = float(np.mean([r["rbs"] for r in rows]))
    ora_m = float(np.mean([r["oracle"] for r in rows]))
    flo_m = float(np.mean([r["floor"] for r in rows]))
    share = (rbs_m - ora_m) / rbs_m if rbs_m else 0.0
    verdict = "headroom exists (>= 15%)" if share >= 0.15 else "no meaningful headroom (< 15%)"
    from collections import Counter

    choice_counts = Counter(c for r in rows for c in r["choices"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "diagnostic_oracle.json").write_text(
        json.dumps(
            {
                "rows": rows,
                "rbs_mean": rbs_m,
                "oracle_mean": ora_m,
                "floor_mean": flo_m,
                "oracle_improvement_share": share,
                "verdict": verdict,
                "choice_counts": dict(choice_counts),
            },
            indent=2,
        )
    )
    lines = [
        "# Exploratory diagnostic: oracle ceiling and physics floor (train, seed k = 0)",
        "",
        "Plan: diagnostic_plan_oracle.md. Exploratory; cannot change gate verdicts.",
        "",
        f"Scenarios: {n}. Mean deaths: RB-S {rbs_m:.2f} | oracle ceiling {ora_m:.2f} "
        f"({100 * share:+.1f}% vs RB-S) | physics floor (no facility damage) {flo_m:.2f}",
        "",
        f"**Reading (fixed in the plan): {verdict}**",
        "",
        "| Scenario | RB-S | Oracle | Floor |",
        "|---|---|---|---|",
    ]
    lines += [f"| {r['scenario_id']} | {r['rbs']} | {r['oracle']} | {r['floor']} |" for r in rows]
    lines += ["", "Oracle's hourly choices (most common):"]
    lines += [f"- {k}: {v}" for k, v in choice_counts.most_common(8)]
    (OUT / "diagnostic_oracle.md").write_text("\n".join(lines) + "\n")
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
