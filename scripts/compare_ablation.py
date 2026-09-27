#!/usr/bin/env python3
"""Energy-floor ablation: condition A vs B on the validation split, with the
pre-declared decision rule (results/protocol/2026-09-27_energy_ablation_and_full_budget.md).

Runs every A and every B MAPPO policy deterministically on each val scenario
for 3 evaluation seeds (`scenario.seed + k * 100000`), default goal weights.
Per-scenario means over training seeds x evaluation seeds; A vs B paired by
scenario; 95% bootstrap CI of (B - A) and two-sided Wilcoxon. Both conditions
are scored with **A's reward definition** (`configs/reward.yaml`) for
`episode_reward`; the policies' behaviour doesn't depend on the scoring.

Decision rule (protocol §5), applied mechanically:
  USE B iff (a) mean unserved energy B <= 50% of A and Wilcoxon p < 0.05, and
  (b) for the death proxy and unmet patient-hours, the upper 95% CI bound of
  (B - A) <= +10% of A's mean. Otherwise USE A.

Usage:
  uv run python scripts/compare_ablation.py --a runs/.../mappo_final.pt ... \\
      --b runs/.../mappo_final.pt ... [--rule-episodes <experiment_a val episodes.json>]
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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _pipeline_common import configure_logging, load_config, resolve_path  # noqa: E402
from evaluate_policies import METRICS, evaluate, per_scenario, run_mappo  # noqa: E402
from udt.agents.marl.mappo import MAPPOTrainer  # noqa: E402
from udt.envs.multi_env import UDTMultiAgentEnv  # noqa: E402
from udt.envs.reward import REWARD_CONFIG_PATH, load_reward_normalisers  # noqa: E402

EVAL_SEEDS = 3
N_BOOTSTRAP = 10_000
ENERGY_REDUCTION = 0.50
NONINFERIORITY = 0.10
PRIMARY = ("patient_deaths", "unmet_patient_hours")


def run_condition(label: str, models: list[str], split: str) -> list[dict[str, Any]]:
    processed = resolve_path(load_config("configs/data.yaml")["paths"]["processed_dir"])
    # Env reward config doesn't affect trajectories (deterministic policy,
    # reward-independent dynamics); scoring below always uses A's normalisers.
    env = UDTMultiAgentEnv(processed_dir=processed, scenario_split=split)
    ids = [s.scenario_id for s in env._scenarios]
    seeds = [s.seed for s in env._scenarios]
    normalisers_a = load_reward_normalisers(REWARD_CONFIG_PATH)
    episodes: list[dict[str, Any]] = []
    for path in models:
        trainer = MAPPOTrainer(env)
        trainer.load(path)
        episodes += evaluate(
            label,
            lambda seed, i, t=trainer: run_mappo(env, t, seed, i),
            ids,
            seeds,
            EVAL_SEEDS,
            normalisers_a,
        )
    env.close()
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", nargs=5, required=True, help="condition A MAPPO models, seeds 0-4")
    parser.add_argument("--b", nargs=5, required=True, help="condition B MAPPO models, seeds 0-4")
    parser.add_argument("--split", default="val", choices=["val"], help="decisions use val only")
    parser.add_argument("--rule-episodes", default=None, help="experiment_a.py --split val output")
    args = parser.parse_args()
    log = configure_logging()

    eps_a = run_condition("A", args.a, args.split)
    log.info("condition_evaluated", condition="A")
    eps_b = run_condition("B", args.b, args.split)
    log.info("condition_evaluated", condition="B")
    rule = (
        [
            e
            for e in json.loads(Path(args.rule_episodes).read_text())
            if e["agent_name"] == "rule_based"
        ]
        if args.rule_episodes
        else []
    )
    ids = sorted({e["scenario_id"] for e in eps_a})
    rng = np.random.default_rng(0)

    def ci(x: np.ndarray) -> list[float]:
        m = x[rng.integers(0, len(x), size=(N_BOOTSTRAP, len(x)))].mean(axis=1)
        return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]

    results: dict[str, Any] = {}
    for metric, _ in METRICS:
        a, b = per_scenario(eps_a, ids, metric), per_scenario(eps_b, ids, metric)
        ok = ~(np.isnan(a) | np.isnan(b))
        a, b = a[ok], b[ok]
        diff = b - a
        p = 1.0 if np.allclose(diff, 0) else float(wilcoxon(b, a).pvalue)
        r = {
            "n_scenarios": int(len(diff)),
            "A_mean": float(a.mean()),
            "B_mean": float(b.mean()),
            "diff_B_minus_A": float(diff.mean()),
            "diff_ci95": ci(diff),
            "wilcoxon_p": p,
        }
        if rule:
            rb = per_scenario(rule, ids, metric)
            r["rule_based_mean"] = float(np.nanmean(rb))
        results[metric] = r

    e = results["unserved_energy_mwh"]
    cond_a = e["B_mean"] <= (1 - ENERGY_REDUCTION) * e["A_mean"] and e["wilcoxon_p"] < 0.05
    cond_b = {
        m: results[m]["diff_ci95"][1] <= NONINFERIORITY * results[m]["A_mean"] for m in PRIMARY
    }
    verdict = "USE B" if cond_a and all(cond_b.values()) else "USE A"
    decision = {
        "(a) energy: B <= 50% of A and p < 0.05": cond_a,
        "(b) non-inferior (upper CI of B-A <= +10% of A)": cond_b,
        "verdict": verdict,
    }

    run_dir = (
        REPO_ROOT
        / "runs"
        / "ablation_energy_floor"
        / (f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:6]}")
    )
    run_dir.mkdir(parents=True)
    meta = {"split": args.split, "a_models": args.a, "b_models": args.b, "eval_seeds": EVAL_SEEDS}
    (run_dir / "results.json").write_text(
        json.dumps({"meta": meta, "results": results, "decision": decision}, indent=2)
    )
    (run_dir / "episodes.json").write_text(
        json.dumps({"A": eps_a, "B": eps_b}, indent=2, default=str)
    )
    lines = [
        f"# Energy-floor ablation — A (floor) vs B (no floor), {args.split} split",
        "",
        "Scored with A's reward definition. Per-scenario means (5 training seeds × 3 eval seeds).",
        "",
        "| Metric | A | B | B − A [95% CI] | p |" + (" Rule-based |" if rule else ""),
        "|---|---|---|---|---|" + ("---|" if rule else ""),
    ]
    for metric, _ in METRICS:
        r = results[metric]
        lines.append(
            f"| {metric} | {r['A_mean']:.2f} | {r['B_mean']:.2f} | {r['diff_B_minus_A']:+.2f} "
            f"[{r['diff_ci95'][0]:+.2f}, {r['diff_ci95'][1]:+.2f}] | {r['wilcoxon_p']:.3f} |"
            + (f" {r['rule_based_mean']:.2f} |" if rule else "")
        )
    lines += [
        "",
        f"**Decision (protocol §5): {verdict}**",
        "",
        "```",
        json.dumps(decision, indent=2),
        "```",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"written to {run_dir}")


if __name__ == "__main__":
    main()
