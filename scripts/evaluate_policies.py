#!/usr/bin/env python3
"""Experiments B (PPO) and C (MAPPO) on the frozen test split, paired against
the rule-based baseline from an Experiment A run (Master Doc §9).

Each trained policy (one per training seed) is run deterministically on
every test scenario for the same evaluation seeds Experiment A used
(`scenario.seed + k * EVAL_SEED_STRIDE`), through its Gym/PettingZoo env
with the §5.4 default goal weights. Per algorithm, each scenario's value is
the mean over training seeds x evaluation seeds. That per-scenario value is
compared with rule-based's per-scenario value from the Experiment A run:
95% bootstrap CI of the paired difference, two-sided Wilcoxon signed-rank,
alpha = 0.05 (declared in advance, no multiple-comparison correction).

Caveat: the envs and Experiment A's harness seed the twin identically but
consume its RNG slightly differently (the multi-agent env draws its goal
vector first), so patient arrivals aren't bit-identical between a policy
episode and the rule-based episode with the same evaluation seed. Pairing is
by scenario (flood, fragility, starting conditions), which is identical.

Usage:
  uv run python scripts/evaluate_policies.py \\
      --baseline runs/experiment_a/<run>/episodes.json \\
      --ppo runs/<id>/ppo_single_env_final.zip ... \\
      --mappo runs/<id>/mappo_final.pt ...
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import wilcoxon

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from stable_baselines3 import PPO  # noqa: E402

from _pipeline_common import configure_logging, load_config, resolve_path  # noqa: E402
from udt.agents.marl.mappo import MAPPOTrainer  # noqa: E402
from udt.common.versions import ENV_VERSION, SUITE_VERSION  # noqa: E402
from udt.envs.multi_env import UDTMultiAgentEnv  # noqa: E402
from udt.envs.reward import episode_reward, load_reward_normalisers  # noqa: E402
from udt.envs.single_env import UDTSingleAgentEnv  # noqa: E402
from udt.logging.metrics import compute_episode_metrics  # noqa: E402

EVAL_SEED_STRIDE = 100_000  # must match scripts/experiment_a.py
N_BOOTSTRAP = 10_000
ALPHA = 0.05
DEFAULT_GOAL = [1.0, 1.0, 1.0, 1.0]
METRICS = [
    ("patient_deaths", False),
    ("unmet_patient_hours", False),
    ("cascading_failure_count", False),
    ("mean_hospital_functional_level", True),
    ("mean_critical_functional_level", True),
    ("unserved_energy_mwh", False),
    ("mean_ambulance_response_delay_hours", False),
    ("requests_completed", True),
    ("episode_reward", True),
]


def run_ppo(env: UDTSingleAgentEnv, model: PPO, seed: int, index: int) -> list[Any]:
    obs, _ = env.reset(seed=seed, options={"scenario_index": index})
    done = False
    while not done:
        obs, _, terminated, truncated, _ = env.step(model.predict(obs, deterministic=True)[0])
        done = terminated or truncated
    return list(env.episode_trace)


def run_mappo(env: UDTMultiAgentEnv, trainer: MAPPOTrainer, seed: int, index: int) -> list[Any]:
    obs, _ = env.reset(seed=seed, options={"scenario_index": index, "goal": DEFAULT_GOAL})
    done = False
    while not done:
        actions = {}
        with torch.no_grad():
            for name in list(env.agents):
                o = torch.as_tensor(obs[name], dtype=torch.float32)
                actions[name] = trainer.actors[name].act(o, deterministic=True)[0].numpy()
        obs, _, terms, truncs, _ = env.step(actions)
        done = any(terms.values()) or any(truncs.values())
    return list(env.episode_trace)


def evaluate(
    label: str,
    runner: Callable[[int, int], list[Any]],
    scenario_ids: list[str],
    scenario_seeds: list[int],
    eval_seeds: int,
    normalisers: dict[str, float],
) -> list[dict[str, Any]]:
    out = []
    for index, (sid, sseed) in enumerate(zip(scenario_ids, scenario_seeds, strict=True)):
        for k in range(eval_seeds):
            seed = sseed + k * EVAL_SEED_STRIDE
            trace = runner(seed, index)
            m = compute_episode_metrics(sid, label, trace).model_dump()
            m["episode_reward"] = episode_reward(trace, normalisers)
            m["eval_seed"] = seed
            out.append(m)
    return out


def per_scenario(episodes: list[dict[str, Any]], ids: list[str], metric: str) -> np.ndarray:
    vals = []
    for sid in ids:
        v = [e[metric] for e in episodes if e["scenario_id"] == sid and e[metric] is not None]
        vals.append(float(np.mean(v)) if v else np.nan)
    return np.array(vals)


def compare(
    algo: np.ndarray, base: np.ndarray, higher_better: bool, rng: np.random.Generator
) -> dict[str, Any]:
    ok = ~(np.isnan(algo) | np.isnan(base))
    a, b = algo[ok], base[ok]
    diff = a - b

    def ci(x: np.ndarray) -> list[float]:
        m = x[rng.integers(0, len(x), size=(N_BOOTSTRAP, len(x)))].mean(axis=1)
        return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]

    p = 1.0 if np.allclose(diff, 0) else float(wilcoxon(a, b).pvalue)
    return {
        "n_scenarios": int(len(diff)),
        "policy_mean": float(a.mean()),
        "policy_ci95": ci(a),
        "rule_based_mean": float(b.mean()),
        "diff_mean_policy_minus_rule": float(diff.mean()),
        "diff_ci95": ci(diff),
        "wilcoxon_p": p,
        "significant_at_0.05": bool(p < ALPHA),
        "policy_better": bool(((diff.mean() > 0) == higher_better) and not np.allclose(diff, 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--baseline", required=True, help="Experiment A episodes.json")
    parser.add_argument("--ppo", nargs="*", default=[])
    parser.add_argument("--mappo", nargs="*", default=[])
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    log = configure_logging()

    processed = resolve_path(load_config(args.config)["paths"]["processed_dir"])
    baseline = json.loads(Path(args.baseline).read_text())
    rule = [e for e in baseline if e["agent_name"] == "rule_based"]
    eval_seeds = len({e["eval_seed"] for e in rule}) // len({e["scenario_id"] for e in rule})
    normalisers = load_reward_normalisers()

    episodes: dict[str, list[dict[str, Any]]] = {}
    if args.ppo:
        env = UDTSingleAgentEnv(processed_dir=processed, scenario_split=args.split)
        ids = [s.scenario_id for s in env._scenarios]
        seeds = [s.seed for s in env._scenarios]
        episodes["ppo"] = []
        for path in args.ppo:
            model = PPO.load(path)
            episodes["ppo"] += evaluate(
                "ppo",
                lambda seed, i, m=model: run_ppo(env, m, seed, i),
                ids,
                seeds,
                eval_seeds,
                normalisers,
            )
            log.info("ppo_policy_evaluated", model=path)
        env.close()
    if args.mappo:
        menv = UDTMultiAgentEnv(processed_dir=processed, scenario_split=args.split)
        ids = [s.scenario_id for s in menv._scenarios]
        seeds = [s.seed for s in menv._scenarios]
        episodes["mappo"] = []
        for path in args.mappo:
            trainer = MAPPOTrainer(menv)
            trainer.load(path)
            episodes["mappo"] += evaluate(
                "mappo",
                lambda seed, i, t=trainer: run_mappo(menv, t, seed, i),
                ids,
                seeds,
                eval_seeds,
                normalisers,
            )
            log.info("mappo_policy_evaluated", model=path)
        menv.close()

    rng = np.random.default_rng(0)
    ids = sorted({e["scenario_id"] for e in rule})
    results = {
        algo: {
            metric: compare(
                per_scenario(eps, ids, metric), per_scenario(rule, ids, metric), hb, rng
            )
            for metric, hb in METRICS
        }
        for algo, eps in episodes.items()
    }

    run_dir = (
        REPO_ROOT
        / "runs"
        / "experiment_bc"
        / (f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:6]}")
    )
    run_dir.mkdir(parents=True)
    meta = {
        "suite_version": SUITE_VERSION,
        "split": args.split,
        "env_version": ENV_VERSION,
        "baseline": args.baseline,
        "ppo_models": args.ppo,
        "mappo_models": args.mappo,
        "eval_seeds_per_scenario": eval_seeds,
    }
    (run_dir / "results.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2))
    (run_dir / "episodes.json").write_text(json.dumps(episodes, indent=2, default=str))
    lines = []
    for algo, res in results.items():
        n_models = len(args.ppo if algo == "ppo" else args.mappo)
        lines += [
            f"## {algo.upper()} vs rule-based ({args.split}, {n_models} training seeds × "
            f"{eval_seeds} eval seeds)",
            "",
            "| Metric | Policy | Rule-based | Difference (policy − rule) | p "
            "| Policy better & sig. |",
            "|---|---|---|---|---|---|",
        ]
        for metric, _ in METRICS:
            r = res[metric]
            lines.append(
                f"| {metric} | {r['policy_mean']:.2f} "
                f"[{r['policy_ci95'][0]:.2f}, {r['policy_ci95'][1]:.2f}] "
                f"| {r['rule_based_mean']:.2f} | {r['diff_mean_policy_minus_rule']:+.2f} "
                f"[{r['diff_ci95'][0]:+.2f}, {r['diff_ci95'][1]:+.2f}] | {r['wilcoxon_p']:.3f} "
                f"| {'yes' if r['policy_better'] and r['significant_at_0.05'] else 'no'} |"
            )
        lines.append("")
    (run_dir / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"written to {run_dir}")


if __name__ == "__main__":
    main()
