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
import json
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
from udt.common.runinfo import machine_info  # noqa: E402
from udt.common.versions import ENV_VERSION, SUITE_VERSION  # noqa: E402
from udt.envs.single_env import UDTSingleAgentEnv  # noqa: E402


class _EpisodeAwareCheckpoint(CheckpointCallback):
    """SB3's CheckpointCallback, plus a `<checkpoint>.meta.json` sidecar with
    the env's episode counter at save time, so `--resume` continues the
    scenario sequence exactly where the checkpoint was taken."""

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.n_calls % self.save_freq == 0:
            path = Path(self._checkpoint_path(extension="zip"))
            env = self.training_env.envs[0].unwrapped  # type: ignore[attr-defined]
            path.with_suffix(".meta.json").write_text(
                json.dumps(
                    {"num_timesteps": int(self.num_timesteps), "episode_count": env._episode_count}
                )
            )
        return result


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
    parser.add_argument(
        "--resume",
        default=None,
        help="an SB3 checkpoint .zip from this run's checkpoints/ to continue from "
        "(approximate: the in-progress rollout/episode is lost; disclosed in protocol §7)",
    )
    args = parser.parse_args()

    log = configure_logging()
    cfg = load_config(args.config)
    processed_dir = resolve_path(cfg["paths"]["processed_dir"])

    run_config = {
        "seed": args.seed,
        "total_timesteps": args.total_timesteps,
        "checkpoint_freq": args.checkpoint_freq,
        "n_ticks": args.n_ticks,
        "env_version": ENV_VERSION,
        "suite_version": SUITE_VERSION,
        "machine": machine_info(),  # protocol addendum 2026-09-28 §2
    }
    if args.resume:
        checkpoint_dir = Path(args.resume).resolve().parent
        run_dir = checkpoint_dir.parent
        saved = json.loads((run_dir / "run_config.json").read_text())
        resumes = saved.pop("resumes", [])
        if saved != run_config:
            raise SystemExit(f"resume config mismatch: {saved} vs {run_config}")
    else:
        run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
        run_dir = REPO_ROOT / "runs" / run_id
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        resumes = []

    env = UDTSingleAgentEnv(processed_dir=processed_dir, n_ticks=args.n_ticks, base_seed=args.seed)
    if args.resume:
        # Continue the scenario sequence from where the checkpoint was taken:
        # the env's episode counter is saved next to every checkpoint
        # (`_EpisodeAwareCheckpoint`), so episodes run after the checkpoint
        # but before an interruption are not counted.
        sidecar = Path(args.resume).with_suffix(".meta.json")
        completed = int(json.loads(sidecar.read_text())["episode_count"])
        env._episode_count = completed
        monitor_path = run_dir / f"monitor_resume{len(resumes) + 1}.csv"
    else:
        monitor_path = run_dir / "monitor.csv"
    monitored_env = Monitor(env, filename=str(monitor_path))

    if args.resume:
        model = PPO.load(
            args.resume, env=monitored_env, tensorboard_log=str(run_dir / "tensorboard")
        )
        resumes.append(
            {
                "from_checkpoint": str(args.resume),
                "num_timesteps": int(model.num_timesteps),
                "episode_count": completed,
                "at": datetime.now(UTC).isoformat(),
            }
        )
        log.info("ppo_resumed", from_timesteps=int(model.num_timesteps), episodes=completed)
    else:
        model = PPO(
            "MlpPolicy",
            monitored_env,
            seed=args.seed,
            verbose=1,
            tensorboard_log=str(run_dir / "tensorboard"),
        )
    (run_dir / "run_config.json").write_text(
        json.dumps({**run_config, "resumes": resumes}, indent=2)
    )

    # save_freq is counted in calls to the (single, non-vectorized) env's
    # step() -> equals env steps here, matching dev doc §5.5's cadence
    # directly (would need dividing by n_envs for a vectorized setup).
    checkpoint_callback = _EpisodeAwareCheckpoint(
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
        total_timesteps=args.total_timesteps - int(model.num_timesteps),
        callback=checkpoint_callback,
        progress_bar=False,
        reset_num_timesteps=not args.resume,
    )

    final_path = run_dir / "ppo_single_env_final.zip"
    model.save(str(final_path))
    log.info("training_complete", final_model=str(final_path), run_dir=str(run_dir))
    env.close()


if __name__ == "__main__":
    main()
