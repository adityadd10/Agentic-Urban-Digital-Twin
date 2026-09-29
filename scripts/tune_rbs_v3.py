#!/usr/bin/env python3
"""Tune RB-S on the twin-v3 train split (protocol addendum 6), then freeze it.

  run    --worker i --n-workers k   evaluate configs i, i+k, ... of the 36-config grid
  select                            apply the pre-registered objective and write
                                    configs/rbs_v3.yaml (refuses to overwrite)

Every config runs the 20 train scenarios x evaluation seeds k = 0, 1, 2
(`scenario.seed + k * 100000`) through `UDTMultiAgentEnvV3` (full horizon,
default goal weights, metric-v2 on). Objective: lowest mean deaths, tie-break
lowest mean unmet patient-hours, then grid order. Train split only.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from udt.agents.rbs_v3 import RBSParams, RuleBasedStrongV3  # noqa: E402
from udt.envs.multi_env_v3 import UDTMultiAgentEnvV3  # noqa: E402
from udt.logging.metrics import compute_episode_metrics  # noqa: E402

GRID = {
    "transfer_buffer_h": [1.0, 2.0, 4.0],
    "transfer_queue_ratio": [0.1, 0.25],
    "divert_queue": [5, 10, 20],
    "surge_queue": [1, 5],
}
EVAL_SEEDS = 3
EVAL_SEED_STRIDE = 100_000
RUN_DIR = REPO_ROOT / "runs" / "rbs_v3_tuning"
OUT_DIR = REPO_ROOT / "results" / "rbs_v3_tuning"
CONFIG_OUT = REPO_ROOT / "configs" / "rbs_v3.yaml"
SUITE = REPO_ROOT / "data" / "scenarios" / "flood_v3"


def grid() -> list[dict[str, Any]]:
    keys = list(GRID)
    return [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*GRID.values())]


def evaluate(params: RBSParams, env: UDTMultiAgentEnvV3) -> list[dict[str, Any]]:
    rbs = RuleBasedStrongV3(params)
    rows = []
    for index, sc in enumerate(env._scenarios):
        for k in range(EVAL_SEEDS):
            env.reset(
                seed=sc.seed + k * EVAL_SEED_STRIDE,
                options={"scenario_index": index, "goal": [1, 1, 1, 1]},
            )
            done = False
            while not done:
                _, _, term, trunc, _ = env.step(rbs.act(env))
                done = any(term.values()) or any(trunc.values())
            m = compute_episode_metrics(sc.scenario_id, "rbs_v3", env.episode_trace).model_dump()
            m["eval_seed"] = sc.seed + k * EVAL_SEED_STRIDE
            rows.append(m)
    return rows


def run(worker: int, n_workers: int) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    env = UDTMultiAgentEnvV3(
        processed_dir=REPO_ROOT / "data/processed",
        reward_config=REPO_ROOT / "configs/reward.yaml",  # reward unused: RB-S is scored on metrics
        suite_dir=SUITE,
        scenario_split="train",
    )
    for i, cfg in enumerate(grid()):
        if i % n_workers != worker:
            continue
        out = RUN_DIR / f"config_{i:02d}.json"
        if out.exists():
            continue
        episodes = evaluate(RBSParams(**cfg), env)
        out.write_text(json.dumps({"index": i, "params": cfg, "episodes": episodes}))
        print(f"config {i:02d} done", flush=True)


def select() -> None:
    if CONFIG_OUT.exists():
        raise SystemExit(f"{CONFIG_OUT} exists: RB-S is frozen")
    results = []
    for i, cfg in enumerate(grid()):
        path = RUN_DIR / f"config_{i:02d}.json"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        eps = json.loads(path.read_text())["episodes"]
        results.append(
            {
                "index": i,
                "params": cfg,
                "mean_deaths": float(np.mean([e["patient_deaths"] for e in eps])),
                "mean_unmet_patient_hours": float(np.mean([e["unmet_patient_hours"] for e in eps])),
                "mean_uncollected_deaths": float(
                    np.mean([e["uncollected_casualty_deaths"] for e in eps])
                ),
                "mean_jobs_completed": float(np.mean([e["jobs_completed"] for e in eps])),
                "mean_transfers": float(np.mean([e["transfers_completed"] for e in eps])),
                "n_episodes": len(eps),
            }
        )
    ranked = sorted(
        results, key=lambda r: (r["mean_deaths"], r["mean_unmet_patient_hours"], r["index"])
    )
    best = ranked[0]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(
        json.dumps({"ranked": ranked, "code_commit": commit}, indent=2)
    )
    lines = [
        "# RB-S tuning (train split, 36 configs x 20 scenarios x 3 seeds)",
        "",
        "Objective (protocol addendum 6): lowest mean deaths, tie-break unmet patient-hours.",
        "",
        "| Rank | buffer_h | queue_ratio | divert_q | surge_q | deaths | unmet p-h | jobs "
        "| transfers |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for rank, r in enumerate(ranked, 1):
        p = r["params"]
        lines.append(
            f"| {rank} | {p['transfer_buffer_h']} | {p['transfer_queue_ratio']} | "
            f"{p['divert_queue']} | "
            f"{p['surge_queue']} | {r['mean_deaths']:.2f} | {r['mean_unmet_patient_hours']:.1f} | "
            f"{r['mean_jobs_completed']:.1f} | {r['mean_transfers']:.1f} |"
        )
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n")
    CONFIG_OUT.write_text(
        yaml.safe_dump(
            {
                "note": (
                    "RB-S for twin-v3, tuned on flood_suite_v3 train only (protocol addendum 6). "
                    "Frozen as git tag rbs-v3; never retune."
                ),
                "params": best["params"],
                "tuning": {
                    "objective": "min mean deaths, tie-break unmet patient-hours",
                    "mean_deaths": best["mean_deaths"],
                    "mean_unmet_patient_hours": best["mean_unmet_patient_hours"],
                    "code_commit": commit,
                    "selected_at": datetime.now(UTC).isoformat(),
                },
            },
            sort_keys=False,
        )
    )
    print("\n".join(lines[:9]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--worker", type=int, required=True)
    r.add_argument("--n-workers", type=int, required=True)
    sub.add_parser("select")
    a = ap.parse_args()
    run(a.worker, a.n_workers) if a.cmd == "run" else select()
