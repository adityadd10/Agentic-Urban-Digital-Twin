#!/usr/bin/env python3
"""Evaluate every policy under ONE robustness condition (one process per condition).

Protocol: results/protocol/2026-09-28_robustness_study.md. Applies the condition
(`udt.robustness.apply`) before anything is built, then runs, on the frozen test
split (10 scenarios x 3 evaluation seeds, `scenario.seed + k * 100000`):
- do-nothing and rule-based, via the same harness as Experiment A;
- unless the condition is rule-based-only: PPO 2M (3 seeds), MAPPO 2M (5 seeds) and,
  as secondary, MAPPO baseline-300k (5 seeds), via their envs with default goals.
Episode reward is always scored with the nominal reward definition (configs/reward.yaml).
Writes <out-dir>/<condition>.json = {policy_label: [episode metrics, ...]}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "tests" / "fixtures"))

from udt import robustness  # noqa: E402

EVAL_SEEDS = 3
EVAL_SEED_STRIDE = 100_000
N_TICKS = 288
MODELS = {
    "ppo_2M": [f"results/full_budget_2M/models/ppo_seed{s}.zip" for s in range(3)],
    "mappo_2M": [f"results/full_budget_2M/models/mappo_seed{s}.pt" for s in range(5)],
    "mappo_300k": [f"results/baseline-300k/models/mappo_seed{s}.pt" for s in range(5)],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, choices=sorted(robustness.CONDITIONS))
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    transform = robustness.apply(args.condition)  # must precede building anything

    import torch
    from shapely.geometry import shape
    from stable_baselines3 import PPO

    from evaluate_policies import evaluate, run_mappo, run_ppo
    from runner import run_episode
    from udt.agents.marl.mappo import MAPPOTrainer
    from udt.agents.rule_based import DoNothingAgent, RuleBasedAgent
    from udt.common.models import DependencyGraph
    from udt.envs.multi_env import UDTMultiAgentEnv
    from udt.envs.reward import episode_reward, load_reward_normalisers
    from udt.envs.single_env import UDTSingleAgentEnv
    from udt.incidents.degradations.flood import SusceptibilityRaster, make_flood_degradation_fn
    from udt.logging.metrics import compute_episode_metrics
    from udt.scenarios.generator import apply_initial_conditions, onset_hour_of_day
    from udt.scenarios.suite import DEFAULT_FLOOD_SUITE_DIR, load_suite
    from udt.twin.road_network import RoadNetwork

    torch.set_num_threads(1)
    processed = REPO_ROOT / "data/processed"
    normalisers = load_reward_normalisers()
    scenarios = load_suite(DEFAULT_FLOOD_SUITE_DIR, "test")
    out: dict[str, list[dict[str, object]]] = {}

    # --- do-nothing and rule-based (Experiment A harness) ---
    base_graph = transform(
        DependencyGraph.model_validate_json((processed / "dependency_graph.json").read_text())
    )
    ward = json.loads((processed / "ward_boundary.geojson").read_text())
    ward_polygon = shape(ward["features"][0]["geometry"])
    with SusceptibilityRaster(processed / "flood_susceptibility.tif") as raster:
        road_network = RoadNetwork.load(processed / "roads_full.graphml", raster)
        for name, cls in (("do_nothing", DoNothingAgent), ("rule_based", RuleBasedAgent)):
            eps = []
            for scenario in scenarios:
                for k in range(EVAL_SEEDS):
                    seed = scenario.seed + k * EVAL_SEED_STRIDE
                    trace = run_episode(
                        apply_initial_conditions(base_graph, scenario),
                        cls(),
                        make_flood_degradation_fn(scenario.incident, raster),
                        N_TICKS,
                        seed=seed,
                        road_network=road_network,
                        incident=scenario.incident,
                        ward_polygon=ward_polygon,
                        onset_hour=onset_hour_of_day(scenario),
                    )
                    m = compute_episode_metrics(scenario.scenario_id, name, trace).model_dump()
                    m["episode_reward"] = episode_reward(trace, normalisers)
                    m["eval_seed"] = seed
                    eps.append(m)
            out[name] = eps

    # --- learned policies (not for rule-based-only conditions) ---
    if args.condition not in robustness.RULE_ONLY:
        env = UDTSingleAgentEnv(processed_dir=processed, scenario_split="test")
        env._base_graph = transform(env._base_graph)
        ids = [s.scenario_id for s in env._scenarios]
        seeds = [s.seed for s in env._scenarios]
        eps = []
        for path in MODELS["ppo_2M"]:
            model = PPO.load(str(REPO_ROOT / path))
            eps += evaluate(
                "ppo_2M",
                lambda seed, i, m=model: run_ppo(env, m, seed, i),
                ids,
                seeds,
                EVAL_SEEDS,
                normalisers,
            )
        env.close()
        out["ppo_2M"] = eps

        menv = UDTMultiAgentEnv(processed_dir=processed, scenario_split="test")
        menv._base_graph = transform(menv._base_graph)
        for label in ("mappo_2M", "mappo_300k"):
            eps = []
            for path in MODELS[label]:
                trainer = MAPPOTrainer(menv)
                trainer.load(REPO_ROOT / path)
                eps += evaluate(
                    label,
                    lambda seed, i, t=trainer: run_mappo(menv, t, seed, i),
                    ids,
                    seeds,
                    EVAL_SEEDS,
                    normalisers,
                )
            out[label] = eps
        menv.close()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.condition}.json").write_text(json.dumps(out, default=str))
    print(f"{args.condition}: done ({', '.join(f'{k}={len(v)}' for k, v in out.items())})")


if __name__ == "__main__":
    main()
