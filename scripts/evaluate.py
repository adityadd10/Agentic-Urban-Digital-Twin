#!/usr/bin/env python3
"""Policy evaluation (dev doc's own layout names `scripts/evaluate.py`;
module M5 slice 2 / dev doc §14 Phase 5's acceptance criterion: "trained
policy beats rule-based on >=1 val flood metric").

Runs a PPO policy (trained, via `--model`, or — with no `--model` — a
uniformly random one, useful as a pipeline check before any training)
against `envs.single_env.UDTSingleAgentEnv`, and the do-nothing/rule-
based baselines from `experiments/runner.py` against the *same* scenario
(same seed, same graph), then prints one comparison table using the
exact same `logging/metrics.py` metrics for all three rows.

**Disclosed caveat — read before reading the table as a verdict:**
- The PPO row's number depends entirely on how much training the loaded
  checkpoint received. `scripts/train.py`'s docstring is explicit that
  no run this session used the dev doc's real 2-5M-step budget — only a
  short pipeline-correctness smoke test. A `ppo_random`/lightly-trained
  row losing to `rule_based` here is expected, not evidence the approach
  doesn't work.
- **Decision cadence — FIXED, no longer an asymmetry:** `experiments/
  runner.py`'s harness now also decides once every `DECISION_INTERVAL_
  TICKS` (3) ticks, the same as `single_env`/`multi_env` (dev doc §5.1) —
  see that module's docstring for the fix. All rows in this table now
  get the same number of decision opportunities.
- n=1 scenario, not a CI — same disclosed limitation as `experiments/
  runner.py` (no frozen scenario suite exists yet, M3's own deferral).

Usage:
  uv run python scripts/evaluate.py --config configs/data.yaml
  uv run python scripts/evaluate.py --config configs/data.yaml \\
      --model runs/<run_id>/ppo_single_env_final.zip
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "tests" / "fixtures"))

from sample_graph import build_sample_dependency_graph  # noqa: E402
from shapely.geometry import shape  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from _pipeline_common import configure_logging, load_config, resolve_path  # noqa: E402
from runner import run_episode  # noqa: E402
from udt.agents.rule_based import DoNothingAgent, RuleBasedAgent  # noqa: E402
from udt.common.models import DependencyGraph, EpisodeMetrics  # noqa: E402
from udt.envs.single_env import UDTSingleAgentEnv  # noqa: E402
from udt.incidents.degradations.flood import (  # noqa: E402
    SusceptibilityRaster,
    make_flood_degradation_fn,
)
from udt.logging.metrics import compute_episode_metrics  # noqa: E402
from udt.scenarios.generator import generate_flood_scenario  # noqa: E402
from udt.twin.road_network import RoadNetwork  # noqa: E402


def evaluate_ppo_policy(
    env: UDTSingleAgentEnv, model: PPO | None, seed: int
) -> list[EpisodeMetrics]:
    obs, _ = env.reset(seed=seed)
    terminated = truncated = False
    while not (terminated or truncated):
        if model is not None:
            action = model.predict(obs, deterministic=True)[0]
        else:
            action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
    return env.episode_trace  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument(
        "--model", default=None, help="path to a trained SB3 .zip; omit for a random policy"
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-ticks", type=int, default=288)
    args = parser.parse_args()

    log = configure_logging()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]
    processed_dir = resolve_path(cfg["paths"]["processed_dir"])

    # --- PPO (trained or random), via the Gym env ---
    env = UDTSingleAgentEnv(processed_dir=processed_dir, n_ticks=args.n_ticks, base_seed=seed)
    model = PPO.load(args.model) if args.model else None
    ppo_trace = evaluate_ppo_policy(env, model, seed)
    ppo_name = "ppo_trained" if model is not None else "ppo_random"
    results = [compute_episode_metrics(f"eval_seed{seed}", ppo_name, ppo_trace)]
    env.close()
    log.info("ppo_episode_complete", **results[0].model_dump())

    # --- do_nothing / rule_based, via the exact harness M4 used, same seed ---
    dep_graph_path = processed_dir / "dependency_graph.json"
    if dep_graph_path.exists():
        with dep_graph_path.open() as f:
            base_graph = DependencyGraph.model_validate(json.load(f))
    else:
        base_graph = build_sample_dependency_graph()

    susceptibility_path = processed_dir / "flood_susceptibility.tif"
    with (processed_dir / "ward_boundary.geojson").open() as f:
        ward_boundary = json.load(f)
    ward_polygon = shape(ward_boundary["features"][0]["geometry"])

    scenario = generate_flood_scenario(
        scenario_id=f"eval_seed{seed}", ward_boundary_geojson=ward_boundary, seed=seed
    )

    with SusceptibilityRaster(susceptibility_path) as raster:
        degradation_fn = make_flood_degradation_fn(scenario.incident, raster)
        road_network = RoadNetwork.load(processed_dir / "roads_full.graphml", raster)
        for agent in (DoNothingAgent(), RuleBasedAgent()):
            graph_copy = base_graph.model_copy(deep=True)
            trace = run_episode(
                graph_copy,
                agent,
                degradation_fn,
                args.n_ticks,
                seed=seed,
                road_network=road_network,
                incident=scenario.incident,
                ward_polygon=ward_polygon,
            )
            metrics = compute_episode_metrics(scenario.scenario_id, agent.name, trace)
            results.append(metrics)
            log.info("baseline_episode_complete", **metrics.model_dump())

    print(
        "\nEvaluation vs. dev doc §14 Phase 5's 'beats rule-based' criterion "
        "(n=1 scenario, same decision cadence for every row)"
    )
    print(
        f"{'agent':<14} {'cascading_failures':>18} {'mean_hospital_fl':>17} "
        f"{'mean_critical_fl':>17} {'unmet_patient_hrs':>17} {'patient_deaths':>14} "
        f"{'unserved_mwh':>12} {'safety_viol':>11}"
    )
    for r in results:
        print(
            f"{r.agent_name:<14} {r.cascading_failure_count:>18} "
            f"{r.mean_hospital_functional_level:>17.3f} {r.mean_critical_functional_level:>17.3f} "
            f"{r.unmet_patient_hours:>17.1f} {r.patient_deaths:>14} {r.unserved_energy_mwh:>12.2f} "
            f"{r.safety_violations_attempted:>11}"
        )


if __name__ == "__main__":
    main()
