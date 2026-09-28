#!/usr/bin/env python3
"""Run and analyse the pre-registered robustness study.

Protocol: results/protocol/2026-09-28_robustness_study.md.

`run`: evaluates every condition in its own process (scripts/robustness_eval.py),
`--jobs` at a time, skipping conditions whose output already exists.
`verify-nominal`: checks the nominal condition reproduces the reported results
bit-identically (Experiment A episodes for do-nothing/rule-based; the full-budget
2M test episodes for PPO/MAPPO) before any perturbed result is used.
`analyse`: applies the fixed pass rules (§4) and writes results/robustness/.

Usage:
  uv run python scripts/robustness_study.py verify-nominal
  uv run python scripts/robustness_study.py run --jobs 8
  uv run python scripts/robustness_study.py analyse
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr, wilcoxon

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from udt.robustness import CONDITIONS, RULE_ONLY  # noqa: E402

RUN_DIR = REPO_ROOT / "runs" / "robustness_study"
OUT_DIR = REPO_ROOT / "results" / "robustness"
PRIMARY = ("patient_deaths", "unmet_patient_hours")
SECONDARY = (
    "mean_hospital_functional_level",
    "unserved_energy_mwh",
    "requests_completed",
    "episode_reward",
)
LEARNED = ("ppo_2M", "mappo_2M")
ALPHA = 0.05
ROBUST_MIN = 18  # of 22 conditions (§4 rule 1)
KEYS = (
    "patient_deaths",
    "unmet_patient_hours",
    "cascading_failure_count",
    "mean_hospital_functional_level",
    "mean_critical_functional_level",
    "unserved_energy_mwh",
    "requests_completed",
    "episode_reward",
)


def cmd_run(jobs: int) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    todo = [c for c in CONDITIONS if not (RUN_DIR / f"{c}.json").exists()]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    running: list[tuple[str, subprocess.Popen[bytes]]] = []
    while todo or running:
        while todo and len(running) < jobs:
            c = todo.pop(0)
            log = open(RUN_DIR / f"{c}.log", "wb")  # noqa: SIM115
            p = subprocess.Popen(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts/robustness_eval.py"),
                    "--condition",
                    c,
                    "--out-dir",
                    str(RUN_DIR),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=REPO_ROOT,
            )
            running.append((c, p))
        for c, p in list(running):
            if p.poll() is not None:
                running.remove((c, p))
                print(f"{c}: exit {p.returncode}", flush=True)
        if running:
            time.sleep(5)


def _episodes(path: Path) -> Any:
    return json.loads(path.read_text())


def cmd_verify_nominal() -> None:
    nominal = _episodes(RUN_DIR / "nominal.json")
    exp_a = _episodes(REPO_ROOT / "results/experiment_a_flood_suite_v2/episodes.json")
    full = _episodes(REPO_ROOT / "results/full_budget_2M/test_episodes.json")
    refs = {
        "do_nothing": [e for e in exp_a if e["agent_name"] == "do_nothing"],
        "rule_based": [e for e in exp_a if e["agent_name"] == "rule_based"],
        "ppo_2M": full["ppo"],
        "mappo_2M": full["mappo"],
    }
    ok = True
    for label, ref in refs.items():
        got = nominal[label]
        same = len(got) == len(ref) and all(
            g["scenario_id"] == r["scenario_id"]
            and g["eval_seed"] == r["eval_seed"]
            and all(g[k] == r[k] for k in KEYS)
            for g, r in zip(got, ref, strict=True)
        )
        ok &= same
        print(f"{label}: {'IDENTICAL' if same else 'DIFFERENT'} ({len(got)} episodes)")
    print("NOMINAL CHECK:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def _per_scenario(eps: list[dict[str, Any]], ids: list[str], metric: str) -> np.ndarray:
    out = []
    for sid in ids:
        v = [e[metric] for e in eps if e["scenario_id"] == sid and e[metric] is not None]
        out.append(float(np.mean(v)) if v else np.nan)
    return np.array(out)


def _category(policy: np.ndarray, rule: np.ndarray, lower_is_better: bool) -> tuple[str, float]:
    d = policy - rule
    if np.allclose(d, 0):
        return "no difference", 1.0
    p = float(wilcoxon(policy, rule).pvalue)
    if p >= ALPHA:
        return "no difference", p
    favours_learned = (d.mean() < 0) if lower_is_better else (d.mean() > 0)
    return ("better" if favours_learned else "worse"), p


def cmd_analyse() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = {c: _episodes(RUN_DIR / f"{c}.json") for c in CONDITIONS}
    ids = sorted({e["scenario_id"] for e in data["nominal"]["rule_based"]})
    perturbed = [c for c in CONDITIONS if c != "nominal" and c not in RULE_ONLY]
    assert len(perturbed) == 22, len(perturbed)

    table: dict[str, Any] = {}
    for c in ["nominal", *perturbed]:
        rule = data[c]["rule_based"]
        table[c] = {}
        for algo in LEARNED:
            table[c][algo] = {}
            for metric in PRIMARY:
                cat, p = _category(
                    _per_scenario(data[c][algo], ids, metric),
                    _per_scenario(rule, ids, metric),
                    lower_is_better=True,
                )
                table[c][algo][metric] = {"category": cat, "p": p}

    verdicts: dict[str, Any] = {}
    for algo in LEARNED:
        for metric in PRIMARY:
            c0 = table["nominal"][algo][metric]["category"]
            flips = [c for c in perturbed if table[c][algo][metric]["category"] != c0]
            kept = len(perturbed) - len(flips)
            lost_families = sorted({CONDITIONS[c][0] for c in flips} if c0 == "better" else set())
            verdicts[f"{algo}/{metric}"] = {
                "nominal_category": c0,
                "conditions_unchanged": kept,
                "robust (>= 18 of 22)": kept >= ROBUST_MIN,
                "flipped_conditions": flips,
                "advantage_not_robust (rule 2)": c0 == "better" and len(lost_families) >= 2,
            }

    # Rule 3 (descriptive): ordering of 4 policies by mean death proxy
    policies = ("do_nothing", "rule_based", "ppo_2M", "mappo_2M")

    def means(c: str) -> list[float]:
        return [
            float(np.nanmean(_per_scenario(data[c][p], ids, "patient_deaths"))) for p in policies
        ]

    nominal_means = means("nominal")
    rhos = [float(spearmanr(nominal_means, means(c)).statistic) for c in perturbed]

    # Per-condition absolute means (all policies incl. rule-only conditions)
    absolute: dict[str, Any] = {}
    for c in CONDITIONS:
        absolute[c] = {
            label: {m: float(np.nanmean(_per_scenario(eps, ids, m))) for m in PRIMARY + SECONDARY}
            for label, eps in data[c].items()
        }

    result = {
        "verdicts": verdicts,
        "per_condition": table,
        "spearman_rho_median": float(np.median(rhos)),
        "spearman_rho_min": float(np.min(rhos)),
        "absolute_means": absolute,
    }
    (OUT_DIR / "results.json").write_text(json.dumps(result, indent=2))

    lines = [
        "# Robustness study — results",
        "",
        "Protocol: `results/protocol/2026-09-28_robustness_study.md`"
        " (pass rules fixed before any 2M result).",
        "Test split, 10 scenarios × 3 eval seeds; categories vs rule-based"
        " by two-sided Wilcoxon, α = 0.05.",
        "",
        "## Verdicts (pre-registered rules)",
        "",
        "| Algorithm / primary metric | Nominal | Unchanged in | Robust (≥ 18/22) | Flipped in |",
        "|---|---|---|---|---|",
    ]
    for k, v in verdicts.items():
        robust = "yes" if v["robust (>= 18 of 22)"] else "NO"
        flipped = ", ".join(v["flipped_conditions"]) or "—"
        lines.append(
            f"| {k} | {v['nominal_category']} | {v['conditions_unchanged']}/22 | "
            f"{robust} | {flipped} |"
        )
    lines += [
        "",
        "Rule 3 (ordering of do-nothing, rule-based, PPO, MAPPO by mean death proxy): "
        f"median Spearman ρ = {result['spearman_rho_median']:.2f} "
        f"(min {result['spearman_rho_min']:.2f}) across the 22 conditions.",
        "",
        "## Mean death proxy / unmet patient-hours per condition",
        "",
        "| Condition | Rule-based | PPO 2M | MAPPO 2M | MAPPO 300k | Do-nothing |",
        "|---|---|---|---|---|---|",
    ]

    def cell(a: dict[str, Any], label: str) -> str:
        if label not in a:
            return "—"
        return f"{a[label]['patient_deaths']:.1f} / {a[label]['unmet_patient_hours']:.0f}"

    for c, (_fam, desc) in CONDITIONS.items():
        a = absolute[c]

        lines.append(
            f"| {c}: {desc} | {cell(a, 'rule_based')} | {cell(a, 'ppo_2M')} | "
            f"{cell(a, 'mappo_2M')} | {cell(a, 'mappo_300k')} | {cell(a, 'do_nothing')} |"
        )
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--jobs", type=int, default=8)
    sub.add_parser("verify-nominal")
    sub.add_parser("analyse")
    args = parser.parse_args()
    if args.cmd == "run":
        cmd_run(args.jobs)
    elif args.cmd == "verify-nominal":
        cmd_verify_nominal()
    else:
        cmd_analyse()


if __name__ == "__main__":
    main()
