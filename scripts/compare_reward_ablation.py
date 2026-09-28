#!/usr/bin/env python3
"""Reward ablation C or D vs A on the validation split, with the pre-registered rule
(results/protocol/2026-09-28_reward_ablations.md §4).

Reuses `compare_ablation.run_condition` (same scenarios, evaluation seeds, default
goals, deterministic actions, and A's reward definition for `episode_reward`).
Adopt X iff (a) its mechanism metric is higher than A's with Wilcoxon p < 0.05, and
(b) for the death proxy and unmet patient-hours, the upper 95% CI bound of (X - A)
is <= +10% of A's mean. Validation only.

Usage:
  uv run python scripts/compare_reward_ablation.py --label C --mechanism requests_completed \\
      --a <A models x5> --x <X models x5> [--rule-episodes <experiment_a val episodes.json>]
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

from compare_ablation import run_condition  # noqa: E402
from evaluate_policies import per_scenario  # noqa: E402

PRIMARY = ("patient_deaths", "unmet_patient_hours")
REPORT = (
    *PRIMARY,
    "requests_completed",
    "requests_pending_at_end",
    "restoration",
    "mean_hospital_functional_level",
    "unserved_energy_mwh",
    "cascading_failure_count",
    "episode_reward",
)
N_BOOTSTRAP = 10_000
NONINFERIORITY = 0.10


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, choices=["C", "D"])
    parser.add_argument("--mechanism", required=True, choices=["requests_completed", "restoration"])
    parser.add_argument("--a", nargs=5, required=True)
    parser.add_argument("--x", nargs=5, required=True)
    parser.add_argument("--rule-episodes", default=None)
    args = parser.parse_args()

    eps_a = run_condition("A", args.a, "val")
    eps_x = run_condition(args.label, args.x, "val")
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

    results: dict[str, Any] = {}
    for metric in REPORT:
        a, x = per_scenario(eps_a, ids, metric), per_scenario(eps_x, ids, metric)
        ok = ~(np.isnan(a) | np.isnan(x))
        a, x = a[ok], x[ok]
        if len(a) == 0:
            results[metric] = {"n_scenarios": 0}
            continue
        d = x - a
        boot = d[rng.integers(0, len(d), size=(N_BOOTSTRAP, len(d)))].mean(axis=1)
        results[metric] = {
            "n_scenarios": int(len(d)),
            "A_mean": float(a.mean()),
            "X_mean": float(x.mean()),
            "diff_X_minus_A": float(d.mean()),
            "diff_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "wilcoxon_p": 1.0 if np.allclose(d, 0) else float(wilcoxon(x, a).pvalue),
        }
        if rule and metric in rule[0]:
            results[metric]["rule_based_mean"] = float(np.nanmean(per_scenario(rule, ids, metric)))

    m = results[args.mechanism]
    cond_a = bool(m.get("n_scenarios")) and m["X_mean"] > m["A_mean"] and m["wilcoxon_p"] < 0.05
    cond_b = {
        k: results[k]["diff_ci95"][1] <= NONINFERIORITY * results[k]["A_mean"] for k in PRIMARY
    }
    verdict = f"ADOPT {args.label}" if cond_a and all(cond_b.values()) else f"REJECT {args.label}"
    decision = {
        f"(a) {args.mechanism}: higher than A with p < 0.05": cond_a,
        "(b) non-inferior (upper CI of X-A <= +10% of A)": cond_b,
        "verdict": verdict,
    }

    run_dir = (
        REPO_ROOT
        / "runs"
        / f"reward_ablation_{args.label}"
        / (f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:6]}")
    )
    run_dir.mkdir(parents=True)
    meta = {
        "label": args.label,
        "mechanism": args.mechanism,
        "split": "val",
        "a": args.a,
        "x": args.x,
    }
    (run_dir / "results.json").write_text(
        json.dumps({"meta": meta, "results": results, "decision": decision}, indent=2)
    )
    (run_dir / "episodes.json").write_text(json.dumps({"A": eps_a, args.label: eps_x}, default=str))
    lines = [
        f"# Reward ablation {args.label} vs A — validation split",
        "",
        "Scored with A's reward definition. Per-scenario means (5 training seeds × 3 eval seeds).",
        "",
        "| Metric | A | "
        + args.label
        + " | "
        + args.label
        + " − A [95% CI] | p |"
        + (" Rule-based |" if rule else ""),
        "|---|---|---|---|---|" + ("---|" if rule else ""),
    ]
    for metric in REPORT:
        r = results[metric]
        if not r.get("n_scenarios"):
            lines.append(f"| {metric} | — | — | — | — |" + (" — |" if rule else ""))
            continue
        rb = (
            f" {r['rule_based_mean']:.2f} |" if "rule_based_mean" in r else (" — |" if rule else "")
        )
        lines.append(
            f"| {metric} | {r['A_mean']:.2f} | {r['X_mean']:.2f} | {r['diff_X_minus_A']:+.2f} "
            f"[{r['diff_ci95'][0]:+.2f}, {r['diff_ci95'][1]:+.2f}] | {r['wilcoxon_p']:.3f} |{rb}"
        )
    lines += [
        "",
        f"**Decision (protocol §4): {verdict}**",
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
