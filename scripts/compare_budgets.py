#!/usr/bin/env python3
"""Full budget (2M) vs baseline-300k, per algorithm, on the test split
(protocol 2026-09-27 §7: "evaluated on test once, against rule-based and
against baseline-300k, with this file's metrics and statistics").

Reads the per-episode results of two `evaluate_policies.py` runs, which used
the same test scenarios and evaluation seeds. Per-scenario means (over
training seeds x evaluation seeds), paired difference (2M - 300k), 95%
bootstrap CI (10,000 resamples), two-sided Wilcoxon, alpha = 0.05. Primary
metrics first.

Usage:
  uv run python scripts/compare_budgets.py \\
      --baseline results/experiment_bc_flood_suite_v2/episodes.json \\
      --full <runs/experiment_bc/.../episodes.json> \\
      --out results/full_budget_2M/vs_baseline_300k.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon

PRIMARY = ["patient_deaths", "unmet_patient_hours"]
SECONDARY = [
    "mean_hospital_functional_level",
    "mean_critical_functional_level",
    "unserved_energy_mwh",
    "mean_ambulance_response_delay_hours",
    "requests_completed",
    "episode_reward",
    "cascading_failure_count",
]
N_BOOTSTRAP = 10_000


def per_scenario(episodes: list[dict[str, Any]], ids: list[str], metric: str) -> np.ndarray:
    out = []
    for sid in ids:
        v = [e[metric] for e in episodes if e["scenario_id"] == sid and e[metric] is not None]
        out.append(float(np.mean(v)) if v else np.nan)
    return np.array(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--full", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    base = json.loads(Path(args.baseline).read_text())
    full = json.loads(Path(args.full).read_text())
    rng = np.random.default_rng(0)
    lines = [
        "# Full budget (2M) vs baseline-300k — test split (flood_suite_v2)",
        "",
        "Per-scenario means; paired difference (2M − 300k), 95% bootstrap CI, two-sided Wilcoxon.",
        "",
    ]
    results: dict[str, Any] = {}
    for algo in ("ppo", "mappo"):
        ids = sorted({e["scenario_id"] for e in full[algo]})
        lines += [
            f"## {algo.upper()}",
            "",
            "| Metric | 300k | 2M | 2M − 300k [95% CI] | p |",
            "|---|---|---|---|---|",
        ]
        results[algo] = {}
        for metric in PRIMARY + SECONDARY:
            a = per_scenario(base[algo], ids, metric)
            b = per_scenario(full[algo], ids, metric)
            ok = ~(np.isnan(a) | np.isnan(b))
            a, b = a[ok], b[ok]
            d = b - a
            m = d[rng.integers(0, len(d), size=(N_BOOTSTRAP, len(d)))].mean(axis=1)
            ci = [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
            p = 1.0 if np.allclose(d, 0) else float(wilcoxon(b, a).pvalue)
            results[algo][metric] = {
                "300k": float(a.mean()),
                "2M": float(b.mean()),
                "diff": float(d.mean()),
                "ci95": ci,
                "p": p,
            }
            tag = " **(primary)**" if metric in PRIMARY else ""
            lines.append(
                f"| {metric}{tag} | {a.mean():.2f} | {b.mean():.2f} | {d.mean():+.2f} "
                f"[{ci[0]:+.2f}, {ci[1]:+.2f}] | {p:.3f} |"
            )
        lines.append("")
    Path(args.out).write_text("\n".join(lines) + "\n")
    Path(args.out).with_suffix(".json").write_text(json.dumps(results, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
