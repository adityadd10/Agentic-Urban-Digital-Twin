#!/usr/bin/env python3
"""Feasibility test 4C (prototype only): evacuation under uncertainty
(results/twin_v3_gates/prototype_hardq_plan.md). Twin-v3 + acuity mechanics (test 2).

  run --part i --n-parts k   fixed rules E0-E4, fair planner (no foresight), clairvoyant oracle
  report                     -> results/twin_v3_gates/prototype_evacuation_results.*
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import prototype_acuity as acuity  # noqa: E402
from udt.agents.rbs_v3 import RuleBasedStrongV3  # noqa: E402
from udt.envs.multi_env_v3 import TRANSFER_OPTIONS, UDTMultiAgentEnvV3  # noqa: E402
from udt.logging.metrics import compute_episode_metrics  # noqa: E402
from udt.twin import ambulances  # noqa: E402
from udt.twin.graph import dependency_edges_of  # noqa: E402

RUN_DIR = REPO_ROOT / "runs" / "prototype_evacuation"
OUT = REPO_ROOT / "results" / "twin_v3_gates"
LATCH_KEY = "_evac_latched"
N_FAIR_ROLLOUTS = 4
TRIGGERS: dict[str, dict[str, float] | None] = {
    "E0_never": None,
    "E1_buffer_lt_2h": {"buffer_h": 2.0},
    "E2_depth_ge_0.2": {"depth_m": 0.2},
    "E3_depth_ge_0.4": {"depth_m": 0.4},
    "E4_aggressive": {"buffer_h": 6.0, "depth_m": 0.1},
}


def take_icu_first_when_evacuating(
    graph: Any, hospital_id: str
) -> tuple[bool, int | None, int | None]:
    a = graph.nodes[hospital_id]["asset"].attributes
    if (
        a.get("divert")
        and not acuity._powerless(graph, hospital_id)
        and int(a.get("icu_occupied", 0)) > 0
    ):
        a["icu_occupied"] = int(a["icu_occupied"]) - 1
        a["beds_occupied"] = max(0, int(a.get("beds_occupied", 0)) - 1)
        now = int(graph.graph.get("_proto_now", 0))
        return True, now - acuity.CRITICAL_SHIFT, None
    return acuity.take_icu_first(graph, hospital_id)


def install() -> None:
    acuity.install()
    ambulances._take_patient_for_transfer = take_icu_first_when_evacuating  # type: ignore[assignment]


class EvacHealth(RuleBasedStrongV3):
    trigger: dict[str, float] | None = None

    def health_action(self, env: UDTMultiAgentEnvV3) -> Any:
        assert env.sim is not None
        out = super().health_action(env)
        if self.trigger is None:
            return out
        g, tick = env.sim.graph, env.sim.tick
        latched: list[str] = g.graph.setdefault(LATCH_KEY, [])
        for i, h in enumerate(env._hospital_ids):
            asset = g.nodes[h]["asset"]
            if h not in latched:
                fire = False
                if "buffer_h" in self.trigger:
                    for e in dependency_edges_of(g, h):
                        st = env.sim.edge_states.get(e.edge_id)
                        if (
                            e.kind == "power"
                            and st is not None
                            and st.remaining_hours < st.capacity_hours
                            and st.remaining_hours < self.trigger["buffer_h"]
                        ):
                            fire = True
                if (
                    "depth_m" in self.trigger
                    and env._degradation_fn.depth_m(asset, tick) >= self.trigger["depth_m"]
                ):
                    fire = True
                if fire:
                    latched.append(h)
            a = asset.attributes
            if (
                h in latched
                and int(a.get("beds_occupied", 0)) + len(a.get("queue_arrivals", [])) > 0
            ):
                out[3 * i] = TRANSFER_OPTIONS.index((5, "urgent"))
                out[3 * i + 1] = 1  # divert
        return out


def make(name: str, params: Any) -> EvacHealth:
    p = EvacHealth(params)
    p.trigger = TRIGGERS[name]
    return p


def _rollout_score(scratch: Any, blob: bytes, policy: Any, seed: int | None) -> float:
    scratch.load_resume_state(blob)
    if seed is not None:  # fair planner: redraw unknown fragility and randomness
        scratch._degradation_fn = scratch._degradation_fn.with_fragility_seed(seed)
        scratch.sim.rng = np.random.default_rng(seed)
    start = scratch.sim._patient_deaths_total
    for _ in range(acuity.HORIZON):
        if not scratch.agents:
            break
        scratch.step(policy.act(scratch))
    tr = scratch.episode_trace
    return (
        0.0
        if not tr
        else float(tr[-1].patient_deaths_cumulative - start)
        + compute_episode_metrics("x", "x", tr).unmet_patient_hours
    )


def _planned_episode(
    env: Any, scratch: Any, index: int, seed: int, policies: list[Any], fair: bool
) -> tuple[dict[str, Any], list[str]]:
    env.reset(seed=seed, options={"scenario_index": index, "goal": [1, 1, 1, 1]})
    chosen, picks, d = policies[0], [], 0
    while env.agents:
        if d % acuity.CHOICE_EVERY == 0:
            blob = acuity._snapshot(env)
            if fair:
                scores = [
                    np.mean(
                        [
                            _rollout_score(scratch, blob, p, 1000 * env.sim.tick + k)
                            for k in range(N_FAIR_ROLLOUTS)
                        ]
                    )
                    for p in policies
                ]
            else:
                scores = [_rollout_score(scratch, blob, p, None) for p in policies]
            chosen = policies[int(np.argmin(scores))]
            picks.append(next(n for n in TRIGGERS if TRIGGERS[n] == chosen.trigger))
        env.step(chosen.act(env))
        d += 1
    return compute_episode_metrics(
        env._scenarios[index].scenario_id, "x", env.episode_trace
    ).model_dump(), picks


def run(part: int, n_parts: int) -> None:
    install()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    env, scratch = acuity._env(), acuity._env()
    params = RuleBasedStrongV3.frozen().params
    policies = [make(n, params) for n in TRIGGERS]
    for j, index in enumerate(acuity.test_scenarios(env)):
        if j % n_parts != part:
            continue
        sc = env._scenarios[index]
        out = RUN_DIR / f"{sc.scenario_id}.json"
        if out.exists():
            continue
        res: dict[str, Any] = {
            "seed": sc.seed,
            "fixed": {
                n: acuity._episode(env, index, sc.seed, p)
                for n, p in zip(TRIGGERS, policies, strict=True)
            },
        }
        res["fair"], res["fair_picks"] = _planned_episode(
            env, scratch, index, sc.seed, policies, fair=True
        )
        res["oracle"], res["oracle_picks"] = _planned_episode(
            env, scratch, index, sc.seed, policies, fair=False
        )
        out.write_text(json.dumps(res))
        print(f"{sc.scenario_id} done", flush=True)


def report() -> None:
    rows = [json.loads(f.read_text()) for f in sorted(RUN_DIR.glob("flood_v3_train_*.json"))]
    fixed = {n: float(np.mean([r["fixed"][n]["patient_deaths"] for r in rows])) for n in TRIGGERS}
    best_name = min(fixed, key=fixed.get)
    best = fixed[best_name]
    fair = float(np.mean([r["fair"]["patient_deaths"] for r in rows]))
    oracle = float(np.mean([r["oracle"]["patient_deaths"] for r in rows]))
    head = (best - fair) / best if best else 0.0
    verdict = (
        "HEADROOM for a learner (>= 15%)" if head >= 0.15 else "no meaningful headroom (< 15%)"
    )
    from collections import Counter

    fp = Counter(p for r in rows for p in r["fair_picks"])
    (OUT / "prototype_evacuation_results.json").write_text(
        json.dumps(
            {
                "fixed_means": fixed,
                "best_fixed": best_name,
                "fair": fair,
                "oracle": oracle,
                "headroom_share": head,
                "verdict": verdict,
            },
            indent=2,
        )
    )
    lines = [
        "# Feasibility test 4C: evacuation under uncertainty (prototype only)",
        "",
        f"Plan: prototype_hardq_plan.md. {len(rows)} moderate/severe train scenarios, "
        "seed k = 0. Mean deaths:",
        "",
        "| Policy | Mean deaths |",
        "|---|---|",
    ]
    lines += [
        f"| Fixed {n}{' (best fixed)' if n == best_name else ''} | {v:.2f} |"
        for n, v in fixed.items()
    ]
    lines += [
        f"| Fair planner (no foresight, 4 resampled rollouts) | {fair:.2f} |",
        f"| Clairvoyant oracle (reference only) | {oracle:.2f} |",
        "",
        f"Fair planner vs best fixed rule: {100 * head:+.1f}%; "
        f"clairvoyant vs best fixed: {100 * (best - oracle) / best:+.1f}%",
        "",
        f"**Reading (fixed in the plan): {verdict}**",
        "",
        "Fair planner's hourly choices: " + ", ".join(f"{k}: {v}" for k, v in fp.most_common()),
    ]
    (OUT / "prototype_evacuation_results.md").write_text("\n".join(lines) + "\n")
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
