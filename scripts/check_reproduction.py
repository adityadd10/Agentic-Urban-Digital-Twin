#!/usr/bin/env python3
"""Reproduction check (protocol 2026-09-27 §6 items 1 and 3).

Rebuilds the MAPPO trainer exactly as `scripts/train_mappo.py` does for the
`baseline-300k` runs (default rollout length 256, unconstrained, same seed),
runs the first `--n` iterations, and compares every logged value with the
baseline run's `training_history.json`, requiring exact equality. Passes only
if every seed matches bit for bit.

Baseline run directories come from `results/baseline-300k/manifest.json`.

Usage:
  OMP_NUM_THREADS=1 uv run python scripts/check_reproduction.py --n 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from udt.agents.marl.mappo import MAPPOConfig, MAPPOTrainer  # noqa: E402
from udt.envs.multi_env import UDTMultiAgentEnv  # noqa: E402

KEYS = ("actor_loss", "value_loss", "entropy", "mean_episode_reward", "n_episodes_completed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=20, help="iterations to compare per seed")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    torch.set_num_threads(1)

    manifest = json.loads((REPO_ROOT / "results/baseline-300k/manifest.json").read_text())
    runs = {
        m["seed"]: REPO_ROOT / Path(m["model_path"]).parent
        for m in manifest["models"]
        if m["policy_type"] == "MAPPO"
    }
    all_ok = True
    for seed in args.seeds:
        baseline = json.loads((runs[seed] / "training_history.json").read_text())[: args.n]
        env = UDTMultiAgentEnv(
            processed_dir=REPO_ROOT / "data/processed", n_ticks=288, base_seed=seed
        )
        trainer = MAPPOTrainer(
            env,
            MAPPOConfig(rollout_length=256, use_action_masking=False, lagrangian_enabled=False),
            seed=seed,
        )
        mismatches = []
        for it in range(1, args.n + 1):
            losses = trainer.train_iteration()
            ref = baseline[it - 1]
            for k in KEYS:
                a, b = float(losses[k]), float(ref[k])
                if not (a == b or (a != a and b != b)):  # exact, NaN == NaN
                    mismatches.append((it, k, b, a))
        env.close()
        ok = not mismatches
        all_ok &= ok
        print(
            f"seed {seed}: {'IDENTICAL' if ok else 'MISMATCH'} over {args.n} iterations", flush=True
        )
        for it, k, b, a in mismatches[:5]:
            print(f"    iter {it} {k}: baseline {b!r} now {a!r}")
    print("REPRODUCTION CHECK:", "PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
