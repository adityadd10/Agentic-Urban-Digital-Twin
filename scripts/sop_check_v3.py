#!/usr/bin/env python3
"""SOP headroom check (protocol 2026-10-02_sop_baseline §3): SOP vs frozen RB-S on train.

run --part i --n-parts k   SOP on train scenarios i, i+k, ... x seeds k = 0, 1, 2
analyse                    -> results/twin_v3_gates/sop_check.{json,md}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

RUN_DIR = REPO_ROOT / "runs" / "sop_check_v3"
OUT = REPO_ROOT / "results" / "twin_v3_gates"
REFERENCE = OUT / "rbs_v3_train_reference.json"
SEEDS, STRIDE = (0, 1, 2), 100_000


def run(part: int, n_parts: int) -> None:
    from udt.agents.sop_v3 import SOPBaselineV3
    from udt.envs.multi_env_v3 import UDTMultiAgentEnvV3
    from udt.logging.metrics import compute_episode_metrics

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    env = UDTMultiAgentEnvV3(
        processed_dir=REPO_ROOT / "data/processed",
        reward_config=REPO_ROOT / "configs/reward.yaml",
        suite_dir=REPO_ROOT / "data/scenarios/flood_v3",
        scenario_split="train",
    )
    sop = SOPBaselineV3()
    for index, sc in enumerate(env._scenarios):
        if index % n_parts != part:
            continue
        rows = []
        for k in SEEDS:
            env.reset(
                seed=sc.seed + k * STRIDE, options={"scenario_index": index, "goal": [1, 1, 1, 1]}
            )
            while env.agents:
                env.step(sop.act(env))
            rows.append(
                {
                    **compute_episode_metrics(
                        sc.scenario_id, "sop_v3", env.episode_trace
                    ).model_dump(),
                    "eval_seed": sc.seed + k * STRIDE,
                }
            )
        (RUN_DIR / f"{sc.scenario_id}.json").write_text(json.dumps(rows))


def _per_scenario(rows: list[dict[str, Any]], metric: str) -> dict[str, float]:
    out: dict[str, list[float]] = {}
    for r in rows:
        out.setdefault(r["scenario_id"], []).append(float(r[metric]))
    return {k: float(np.mean(v)) for k, v in out.items()}


def analyse() -> None:
    sop = [r for f in sorted(RUN_DIR.glob("*.json")) for r in json.loads(f.read_text())]
    rbs = json.loads(REFERENCE.read_text())
    res: dict[str, Any] = {}
    rng = np.random.default_rng(0)
    for metric in (
        "patient_deaths",
        "unmet_patient_hours",
        "mean_critical_functional_level",
        "jobs_completed",
    ):
        a, b = _per_scenario(sop, metric), _per_scenario(rbs, metric)
        ids = sorted(set(a) & set(b))
        d = np.array([a[i] - b[i] for i in ids])  # SOP - RB-S
        boot = d[rng.integers(0, len(d), size=(10_000, len(d)))].mean(axis=1)
        res[metric] = {
            "sop": float(np.mean([a[i] for i in ids])),
            "rbs": float(np.mean([b[i] for i in ids])),
            "diff": float(d.mean()),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "wilcoxon_p": 1.0 if np.allclose(d, 0) else float(wilcoxon(d).pvalue),
            "n": len(ids),
        }
    dd = res["patient_deaths"]
    share = dd["diff"] / dd["sop"] if dd["sop"] else 0.0
    passes = share >= 0.15 and dd["ci95"][0] > 0
    verdict = (
        "SOP leaves meaningful room for a learner (>= 15%)"
        if passes
        else "SOP already near-best (< 15%)"
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sop_check.json").write_text(
        json.dumps({"results": res, "gap_share_of_sop": share, "verdict": verdict}, indent=2)
    )
    lines = [
        "# SOP headroom check (train, 20 scenarios x 3 seeds)",
        "",
        "Protocol 2026-10-02_sop_baseline §3. Difference = SOP - RB-S "
        "(positive deaths = SOP worse).",
        "",
        "| Metric | SOP | RB-S | SOP - RB-S [95% CI] | p |",
        "|---|---|---|---|---|",
    ]
    for m, r in res.items():
        lines.append(
            f"| {m} | {r['sop']:.3f} | {r['rbs']:.3f} | {r['diff']:+.3f} "
            f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['wilcoxon_p']:.4f} |"
        )
    lines += [
        "",
        f"Gap in deaths as a share of SOP's mean: {100 * share:.1f}%",
        "",
        f"**Reading (fixed in the protocol): {verdict}**",
    ]
    (OUT / "sop_check.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--part", type=int, required=True)
    r.add_argument("--n-parts", type=int, required=True)
    sub.add_parser("analyse")
    a = ap.parse_args()
    run(a.part, a.n_parts) if a.cmd == "run" else analyse()
