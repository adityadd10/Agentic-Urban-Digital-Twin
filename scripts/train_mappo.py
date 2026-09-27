#!/usr/bin/env python3
"""MAPPO training (dev doc §5.5, module M6a slice 2, extended in M6b to
write a policy-registry entry).

Trains `agents.marl.mappo.MAPPOTrainer` on `envs.multi_env.
UDTMultiAgentEnv`, checkpointing periodically. Same disclosed-scope
discipline as `scripts/train.py` (M5's PPO script) — see that file's
docstring for the "not run at the real 2-5M-step budget this session"
disclosure, which applies here too (if anything, more so: this hand-
rolled trainer is a single non-vectorized rollout, see `agents/marl/
mappo.py`'s docstring, so its steps/sec is at or below `single_env`'s
already-measured ~46-50/sec).

**M6b addition:** after training, runs a few deterministic episodes
(different seeds from any training episode) and appends one entry to
`registry/policies.yaml` (`agents/marl/registry.py`, schema in `common/
models.py`'s `PolicyRegistryEntry`) — dev doc §5.5: "the ID/OOD router
reads this file" (M9b, not built yet, so nothing reads it back today).
`val_score` is explicitly *not* a true held-out validation score — no
train/val/test scenario suite exists yet (M3's own deferral) — just a
few fresh-seed deterministic episodes, flagged via `val_score_is_true_
holdout=False` in the written entry rather than implying more rigor
than exists.

No MLflow yet (same disclosed deferral project-wide) — structlog to
stdout + a plain JSON checkpoint of losses per iteration.

**M7 slice 2 addition:** `--constrained` turns on dev doc §5.5's
Experiment G as one unit — `MAPPOConfig.use_action_masking` +
`lagrangian_enabled` and `UDTMultiAgentEnv.constrained_reward` together
— rather than exposing each independently on this CLI; see `agents/marl/
mappo.py`'s module docstring for what each one actually does. Omit it
for the default unconstrained MAPPO (Experiments C/E/F).

Usage:
  uv run python scripts/train_mappo.py --config configs/data.yaml \\
      --n-iterations 5 --rollout-length 256
  uv run python scripts/train_mappo.py --config configs/data.yaml \\
      --n-iterations 5 --rollout-length 256 --constrained
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from _pipeline_common import configure_logging, load_config, resolve_path  # noqa: E402
from udt.agents.marl.mappo import MAPPOConfig, MAPPOTrainer  # noqa: E402
from udt.agents.marl.registry import append_entry  # noqa: E402
from udt.common.models import PolicyRegistryEntry  # noqa: E402
from udt.common.runinfo import machine_info  # noqa: E402
from udt.common.versions import SUITE_VERSION  # noqa: E402
from udt.envs.multi_env import UDTMultiAgentEnv  # noqa: E402

N_EVAL_EPISODES = 3
EVAL_SEED_OFFSET = 10_000  # kept clear of any training-episode seed


def _evaluate_deterministic(
    trainer: MAPPOTrainer,
    env: UDTMultiAgentEnv,
    n_episodes: int,
    base_seed: int,
    *,
    use_action_masking: bool = False,
    scenario_indices: list[int] | None = None,
) -> float:
    """Mean episode reward over `n_episodes` deterministic rollouts —
    disclosed as *not* a true held-out validation score, see module
    docstring.

    `use_action_masking` (M7 slice 2): a policy trained with masking on
    relies on the mask being applied at inference time too — evaluating
    it unmasked would score behavior the policy was never actually
    trained to produce on its own, not a faithful "how does this policy
    perform" check. Pass the same value `--constrained` set during
    training."""
    episode_rewards = []
    for i in range(n_episodes):
        options = {"scenario_index": scenario_indices[i]} if scenario_indices else None
        obs, _ = env.reset(seed=base_seed + EVAL_SEED_OFFSET + i, options=options)
        obs_t = {name: torch.as_tensor(obs[name], dtype=torch.float32) for name in env.agents}
        total_reward = 0.0
        done = False
        while not done:
            active_agents = list(env.agents)  # capture before step() may clear it on episode end
            actions = {}
            with torch.no_grad():
                for name in active_agents:
                    mask = None
                    if use_action_masking:
                        mask = [torch.as_tensor(m, dtype=torch.bool) for m in env.action_mask(name)]
                    action, _ = trainer.actors[name].act(obs_t[name], deterministic=True, mask=mask)
                    actions[name] = action.numpy()
            next_obs, rewards, terminations, truncations, _ = env.step(actions)
            total_reward += rewards[active_agents[0]]
            done = any(terminations.values()) or any(truncations.values())
            if not done:
                obs_t = {
                    name: torch.as_tensor(next_obs[name], dtype=torch.float32)
                    for name in env.agents
                }
        episode_rewards.append(total_reward)
    return float(np.mean(episode_rewards))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=200,
        help="each iteration = one rollout + PPO update; not run at a scale "
        "matching dev doc §5.5's 2-5M env steps this session, see docstring",
    )
    parser.add_argument("--rollout-length", type=int, default=256)
    parser.add_argument("--checkpoint-every", type=int, default=10, help="in iterations")
    parser.add_argument("--n-ticks", type=int, default=288)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--registry-path",
        default=str(REPO_ROOT / "registry" / "policies.yaml"),
        help="dev doc §5.5's registry/policies.yaml",
    )
    parser.add_argument(
        "--incident-type",
        default="flood",
        help="prototype-1 scope is flood-only (M3's own row) - the only value this has ever been",
    )
    parser.add_argument(
        "--constrained",
        action="store_true",
        help="dev doc §5.5's Experiment G, module M7 slice 2: turns on pre-hoc action "
        "masking (MAPPOConfig.use_action_masking), the PPO-Lagrangian penalty "
        "(lagrangian_enabled) and multi_env's constrained_reward together, as one unit "
        "- omit for the default unconstrained MAPPO (Experiments C/E/F)",
    )
    parser.add_argument(
        "--reward-config",
        default=None,
        help="reward normaliser YAML; default = the frozen configs/reward.yaml "
        "(protocol 2026-09-27: condition B uses configs/reward_no_energy_floor.yaml)",
    )
    parser.add_argument(
        "--resume-every",
        type=int,
        default=0,
        help="write resume_latest.pt (exact-resume state) every N iterations; 0 = off",
    )
    parser.add_argument(
        "--resume", default=None, help="resume_latest.pt to continue from (same args required)"
    )
    args = parser.parse_args()

    log = configure_logging()
    cfg = load_config(args.config)
    processed_dir = resolve_path(cfg["paths"]["processed_dir"])

    if args.resume:
        run_dir = Path(args.resume).resolve().parent  # continue in the same run dir
    else:
        run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
        run_dir = REPO_ROOT / "runs" / run_id
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # What exactly this run is, for the resume check and the record.
    run_config = {
        "seed": args.seed,
        "n_iterations": args.n_iterations,
        "rollout_length": args.rollout_length,
        "n_ticks": args.n_ticks,
        "constrained": args.constrained,
        "reward_config": args.reward_config,
        "env_version": UDTMultiAgentEnv.metadata["name"],
        "suite_version": SUITE_VERSION,
        # Protocol addendum 2026-09-28 §2: part of the resume check, so a
        # resume on another machine or changed code is refused.
        "machine": machine_info(),
    }

    env = UDTMultiAgentEnv(
        processed_dir=processed_dir,
        n_ticks=args.n_ticks,
        base_seed=args.seed,
        constrained_reward=args.constrained,
        reward_config=args.reward_config,
    )
    trainer = MAPPOTrainer(
        env,
        MAPPOConfig(
            rollout_length=args.rollout_length,
            use_action_masking=args.constrained,
            lagrangian_enabled=args.constrained,
        ),
        seed=args.seed,
    )

    log.info(
        "mappo_training_started",
        n_iterations=args.n_iterations,
        rollout_length=args.rollout_length,
        run_dir=str(run_dir),
    )

    history: list[dict[str, float]] = []
    start_iteration = 1
    if args.resume:
        resumed = trainer.load_resume(args.resume)
        if resumed["metadata"] != run_config:
            raise SystemExit(f"resume config mismatch: {resumed['metadata']} vs {run_config}")
        history = resumed["history"]
        start_iteration = resumed["iteration"] + 1
        log.info("mappo_resumed", from_iteration=resumed["iteration"], run_dir=str(run_dir))
    else:
        (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    for iteration in range(start_iteration, args.n_iterations + 1):
        losses = trainer.train_iteration()
        history.append({"iteration": iteration, **losses})
        log.info("mappo_iteration_complete", iteration=iteration, **losses)

        if args.resume_every and iteration % args.resume_every == 0:
            trainer.save_resume(
                run_dir / "resume_latest.pt",
                iteration=iteration,
                history=history,
                metadata=run_config,
            )

        if iteration % args.checkpoint_every == 0 or iteration == args.n_iterations:
            checkpoint_path = checkpoint_dir / f"mappo_iter{iteration}.pt"
            trainer.save(checkpoint_path)
            log.info("checkpoint_saved", path=str(checkpoint_path))

    (run_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    final_path = run_dir / "mappo_final.pt"
    trainer.save(final_path)
    log.info("mappo_training_complete", final_model=str(final_path), run_dir=str(run_dir))

    # 2026-09-27: validation on the frozen suite's held-out val split (dev doc
    # §4.3): one deterministic episode per val scenario, in a separate env.
    val_env = UDTMultiAgentEnv(
        processed_dir=processed_dir,
        n_ticks=args.n_ticks,
        base_seed=args.seed,
        constrained_reward=args.constrained,
        scenario_split="val",
        reward_config=args.reward_config,
    )
    n_val = len(val_env._scenarios)
    val_score = _evaluate_deterministic(
        trainer,
        val_env,
        n_val,
        args.seed,
        use_action_masking=args.constrained,
        scenario_indices=list(range(n_val)),
    )
    val_env.close()
    log.info("deterministic_eval_complete", val_score=val_score, n_episodes=n_val, split="val")

    entry = PolicyRegistryEntry(
        incident_type=args.incident_type,
        policy_path=str(final_path),
        env_version=UDTMultiAgentEnv.metadata["name"],
        suite_version=SUITE_VERSION,
        val_score=val_score,
        val_score_is_true_holdout=True,  # mean over the frozen suite's val split
        trained_date=datetime.now(UTC).isoformat(),
    )
    append_entry(args.registry_path, entry)
    log.info("registry_entry_appended", path=args.registry_path, **entry.model_dump())

    env.close()


if __name__ == "__main__":
    main()
