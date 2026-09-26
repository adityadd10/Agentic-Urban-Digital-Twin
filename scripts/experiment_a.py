#!/usr/bin/env python3
"""Experiment A on the frozen test split (Master Doc §9, dev doc §4.3/§15).

Rule-based baseline vs. do-nothing, run on every scenario of
`flood_suite_v1`'s **test** split, for `--eval-seeds` fixed evaluation seeds
each. The evaluation seed changes only the twin's own randomness (patient
arrivals, discharges, ambulance calls); the scenario (flood, fragility,
starting conditions) stays fixed.

Protocol (Master Doc §9, declared before running):
- unit of analysis = scenario: each agent's metric is averaged over the
  evaluation seeds per scenario, giving n = 10 paired observations;
- 95% bootstrap CI (10,000 resamples of scenarios) for each agent's mean and
  for the paired mean difference (rule_based − do_nothing);
- Wilcoxon signed-rank test on the paired per-scenario values, two-sided,
  alpha = 0.05. No correction across metrics is applied; the p-values are
  reported per metric and should be read as such.

`patient_deaths` is the queue-wait proxy (dev doc §3.8 item 8), not
clinical mortality. `episode_reward` is the §5.4 reward with the frozen
`configs/reward.yaml` normalisers and default goal weights.

Usage:
  uv run python scripts/experiment_a.py --eval-seeds 3
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon
from shapely.geometry import shape

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "tests" / "fixtures"))

from _pipeline_common import configure_logging, load_config, resolve_path  # noqa: E402
from runner import run_episode  # noqa: E402
from udt.agents.rule_based import DoNothingAgent, RuleBasedAgent  # noqa: E402
from udt.common.models import DependencyGraph  # noqa: E402
from udt.common.versions import ENV_VERSION, SUITE_VERSION  # noqa: E402
from udt.envs.reward import episode_reward, load_reward_normalisers  # noqa: E402
from udt.incidents.degradations.flood import (  # noqa: E402
    SusceptibilityRaster,
    make_flood_degradation_fn,
)
from udt.logging.metrics import compute_episode_metrics  # noqa: E402
from udt.scenarios.generator import apply_initial_conditions, onset_hour_of_day  # noqa: E402
from udt.scenarios.suite import DEFAULT_FLOOD_SUITE_DIR, load_suite  # noqa: E402
from udt.twin.road_network import RoadNetwork  # noqa: E402

N_TICKS = 288
EVAL_SEED_STRIDE = 100_000  # evaluation seed k of a scenario = scenario.seed + k * stride
N_BOOTSTRAP = 10_000
ALPHA = 0.05
# (metric, higher_is_better)
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


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    idx = rng.integers(0, len(values), size=(N_BOOTSTRAP, len(values)))
    means = values[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--eval-seeds", type=int, default=3)
    args = parser.parse_args()
    log = configure_logging()

    processed = resolve_path(load_config(args.config)["paths"]["processed_dir"])
    base_graph = DependencyGraph.model_validate(
        json.loads((processed / "dependency_graph.json").read_text())
    )
    ward = json.loads((processed / "ward_boundary.geojson").read_text())
    ward_polygon = shape(ward["features"][0]["geometry"])
    scenarios = load_suite(DEFAULT_FLOOD_SUITE_DIR, args.split)
    normalisers = load_reward_normalisers()

    agents = {"do_nothing": DoNothingAgent, "rule_based": RuleBasedAgent}
    episodes: list[dict[str, Any]] = []
    with SusceptibilityRaster(processed / "flood_susceptibility.tif") as raster:
        road_network = RoadNetwork.load(processed / "roads_full.graphml", raster)
        for scenario in scenarios:
            for k in range(args.eval_seeds):
                eval_seed = scenario.seed + k * EVAL_SEED_STRIDE
                for name, agent_cls in agents.items():
                    trace = run_episode(
                        apply_initial_conditions(base_graph, scenario),
                        agent_cls(),
                        make_flood_degradation_fn(scenario.incident, raster),
                        N_TICKS,
                        seed=eval_seed,
                        road_network=road_network,
                        incident=scenario.incident,
                        ward_polygon=ward_polygon,
                        onset_hour=onset_hour_of_day(scenario),
                    )
                    m = compute_episode_metrics(scenario.scenario_id, name, trace).model_dump()
                    m["episode_reward"] = episode_reward(trace, normalisers)
                    m["eval_seed"] = eval_seed
                    episodes.append(m)
            log.info("scenario_done", scenario=scenario.scenario_id)

    # per-scenario means (unit of analysis)
    ids = [s.scenario_id for s in scenarios]

    def per_scenario(agent: str, metric: str) -> np.ndarray:
        out = []
        for sid in ids:
            vals = [
                e[metric]
                for e in episodes
                if e["agent_name"] == agent and e["scenario_id"] == sid and e[metric] is not None
            ]
            out.append(float(np.mean(vals)) if vals else np.nan)
        return np.array(out)

    rng = np.random.default_rng(0)
    results: dict[str, Any] = {}
    for metric, higher_better in METRICS:
        a = per_scenario("do_nothing", metric)
        b = per_scenario("rule_based", metric)
        ok = ~(np.isnan(a) | np.isnan(b))
        a, b = a[ok], b[ok]
        diff = b - a
        if len(diff) == 0:
            results[metric] = {"n_scenarios": 0}
            continue
        if np.allclose(diff, 0):
            p = 1.0
        else:
            p = float(wilcoxon(b, a, alternative="two-sided", zero_method="wilcox").pvalue)
        better = (diff.mean() > 0) == higher_better and not np.allclose(diff, 0)
        results[metric] = {
            "n_scenarios": int(len(diff)),
            "do_nothing_mean": float(a.mean()),
            "do_nothing_ci95": bootstrap_ci(a, rng),
            "rule_based_mean": float(b.mean()),
            "rule_based_ci95": bootstrap_ci(b, rng),
            "diff_mean_rule_minus_nothing": float(diff.mean()),
            "diff_ci95": bootstrap_ci(diff, rng),
            "wilcoxon_p": p,
            "significant_at_0.05": bool(p < ALPHA),
            "rule_based_better": bool(better),
        }

    run_dir = (
        REPO_ROOT
        / "runs"
        / "experiment_a"
        / (f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:6]}")
    )
    run_dir.mkdir(parents=True)
    meta = {
        "suite_version": SUITE_VERSION,
        "split": args.split,
        "env_version": ENV_VERSION,
        "n_scenarios": len(scenarios),
        "eval_seeds_per_scenario": args.eval_seeds,
        "n_bootstrap": N_BOOTSTRAP,
        "alpha": ALPHA,
        "reward_normalisers": normalisers,
    }
    (run_dir / "results.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2))
    (run_dir / "episodes.json").write_text(json.dumps(episodes, indent=2, default=str))

    lines = [
        f"# Experiment A — rule-based vs do-nothing ({args.split} split, {SUITE_VERSION})",
        "",
        f"n = {len(scenarios)} scenarios × {args.eval_seeds} eval seeds; per-scenario means; "
        "95% bootstrap CIs; two-sided Wilcoxon signed-rank, α = 0.05.",
        "",
        "| Metric | Do-nothing | Rule-based | Difference (rule − nothing) | p "
        "| Rule better & significant |",
        "|---|---|---|---|---|---|",
    ]
    for metric, _ in METRICS:
        r = results[metric]
        if not r.get("n_scenarios"):
            lines.append(f"| {metric} | — | — | — | — | n/a |")
            continue
        lines.append(
            f"| {metric} | {r['do_nothing_mean']:.2f} "
            f"[{r['do_nothing_ci95'][0]:.2f}, {r['do_nothing_ci95'][1]:.2f}] "
            f"| {r['rule_based_mean']:.2f} "
            f"[{r['rule_based_ci95'][0]:.2f}, {r['rule_based_ci95'][1]:.2f}] "
            f"| {r['diff_mean_rule_minus_nothing']:+.2f} "
            f"[{r['diff_ci95'][0]:+.2f}, {r['diff_ci95'][1]:+.2f}] "
            f"| {r['wilcoxon_p']:.3f} "
            f"| {'yes' if r['rule_based_better'] and r['significant_at_0.05'] else 'no'} |"
        )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwritten to {run_dir}")


if __name__ == "__main__":
    main()
