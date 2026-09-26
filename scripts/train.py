#!/usr/bin/env python3
"""PPO training (dev doc §5.5, Experiment B, module M5 slice 2).

Trains PPO (Stable-Baselines3) on `envs.single_env.UDTSingleAgentEnv`,
checkpointing periodically.

**Disclosed scope — read before trusting any trained model from this
script:** dev doc §5.5's training budget is "~2-5M env steps per
policy... checkpoint [...] every 50k steps". This script supports that
budget (`--total-timesteps`, default 2,000,000 — the low end of the
range) and implements the exact checkpoint cadence, but **no run in this
session has actually executed the real budget** — at this env's measured
throughput (~50 steps/sec on the real Kurla graph, dominated by real
Dijkstra routing + demand simulation, not neural-network overhead),
2-5M steps would take roughly 11-28 hours, well beyond a single
interactive turn. What this session verified instead: a short smoke-test
run (a few thousand steps, `--total-timesteps 5000` or similar) proving
the pipeline itself works — env/SB3 integration, checkpoint files
actually written, no crashes across a real rollout. Reaching dev doc
§14 Phase 5's actual acceptance criterion ("trained policy beats
rule-based on >=1 val metric") needs someone to run this script for the
real budget — `scripts/evaluate.py` is what checks that criterion once a
real checkpoint exists.

No MLflow yet (same disclosed deferral as `experiments/runner.py` — add
it when a real multi-seed sweep makes a tracking UI worth the
dependency). SB3's own Monitor CSV + TensorBoard logs, plus structlog to
stdout, cover this session's needs.

Usage:
  uv run python scripts/train.py --config configs/data.yaml --total-timesteps 5000
"""

from __future__ import annotations

import argparse
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import CheckpointCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402

from _pipeline_common import configure_logging, load_config, resolve_path  # noqa: E402
from udt.envs.single_env import UDTSingleAgentEnv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=2_000_000,
        help="dev doc §5.5: budget 2-5M, default is low end (not run this session)",
    )
    parser.add_argument(
        "--checkpoint-freq", type=int, default=50_000, help="dev doc §5.5 exact cadence"
    )
    parser.add_argument("--n-ticks", type=int, default=288)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    log = configure_logging()
    cfg = load_config(args.config)
    processed_dir = resolve_path(cfg["paths"]["processed_dir"])

    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
    run_dir = REPO_ROOT / "runs" / run_id
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env = UDTSingleAgentEnv(processed_dir=processed_dir, n_ticks=args.n_ticks, base_seed=args.seed)
    monitored_env = Monitor(env, filename=str(run_dir / "monitor.csv"))

    model = PPO(
        "MlpPolicy",
        monitored_env,
        seed=args.seed,
        verbose=1,
        tensorboard_log=str(run_dir / "tensorboard"),
    )

    # save_freq is counted in calls to the (single, non-vectorized) env's
    # step() -> equals env steps here, matching dev doc §5.5's cadence
    # directly (would need dividing by n_envs for a vectorized setup).
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(checkpoint_dir),
        name_prefix="ppo_single_env",
    )

    log.info(
        "training_started",
        total_timesteps=args.total_timesteps,
        checkpoint_freq=args.checkpoint_freq,
        run_dir=str(run_dir),
    )
    model.learn(
        total_timesteps=args.total_timesteps, callback=checkpoint_callback, progress_bar=False
    )

    final_path = run_dir / "ppo_single_env_final.zip"
    model.save(str(final_path))
    log.info("training_complete", final_model=str(final_path), run_dir=str(run_dir))
    env.close()


if __name__ == "__main__":
    main()
