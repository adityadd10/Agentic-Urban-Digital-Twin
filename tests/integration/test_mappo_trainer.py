"""Phase 6 acceptance test: `agents.marl.mappo.MAPPOTrainer` runs a real
rollout + PPO update against the real Kurla graph without crashing or
producing NaNs, and checkpoint save/load round-trips correctly.

Not a claim about learning quality — dev doc §5.5's real training budget
(2-5M env steps) isn't run here, see `scripts/train_mappo.py`'s
docstring. This is the same "pipeline correctness, not scale" bar
`tests/integration/test_single_env.py` set for M5.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from udt.agents.marl.mappo import MAPPOConfig, MAPPOTrainer
from udt.envs.multi_env import UDTMultiAgentEnv

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
REQUIRED_FILES = (
    "dependency_graph.json",
    "flood_susceptibility.tif",
    "ward_boundary.geojson",
    "roads_full.graphml",
)

pytestmark = pytest.mark.skipif(
    not all((PROCESSED_DIR / f).exists() for f in REQUIRED_FILES),
    reason="needs data/processed/{dependency_graph.json,flood_susceptibility.tif,"
    "ward_boundary.geojson,roads_full.graphml} (run the 01-06 pipeline scripts)",
)


@pytest.fixture
def trainer() -> MAPPOTrainer:
    env = UDTMultiAgentEnv(processed_dir=PROCESSED_DIR, n_ticks=30, base_seed=0)
    t = MAPPOTrainer(env, MAPPOConfig(rollout_length=32, n_epochs=2), seed=0)
    yield t
    env.close()


@pytest.mark.phase6
def test_train_iteration_runs_without_nan(trainer: MAPPOTrainer) -> None:
    losses = trainer.train_iteration()
    for key, value in losses.items():
        assert value == value, f"{key} is NaN"  # NaN != NaN
    assert trainer.total_steps == 32


@pytest.mark.phase6
def test_multiple_iterations_keep_running(trainer: MAPPOTrainer) -> None:
    for _ in range(3):
        losses = trainer.train_iteration()
        assert all(v == v for v in losses.values())
    assert trainer.total_steps == 96


@pytest.mark.phase6
def test_checkpoint_round_trip(trainer: MAPPOTrainer, tmp_path: Path) -> None:
    trainer.train_iteration()
    checkpoint_path = tmp_path / "mappo_test.pt"
    trainer.save(checkpoint_path)
    assert checkpoint_path.exists()

    # A fresh trainer with different random init, then loaded, must
    # reproduce the saved trainer's action distribution exactly.
    env2 = UDTMultiAgentEnv(processed_dir=PROCESSED_DIR, n_ticks=30, base_seed=0)
    fresh = MAPPOTrainer(env2, MAPPOConfig(rollout_length=32), seed=999)
    fresh.load(checkpoint_path)
    env2.close()

    for name in trainer.agent_names:
        # Each agent's actor has its own obs dim (18/12/23 on the real
        # Kurla graph) - generate per-agent, not one shared vector.
        obs = torch.randn(trainer.actors[name].backbone[0].in_features)
        a1, _ = trainer.actors[name].act(obs, deterministic=True)
        a2, _ = fresh.actors[name].act(obs, deterministic=True)
        assert torch.equal(a1, a2), f"{name}'s loaded weights don't match the saved ones"


# ---------------------------------------------------------------------------
# M7 slice 2: action masking + PPO-Lagrangian (dev doc §5.5's Experiment
# G), real smoke run against the real Kurla graph.
# ---------------------------------------------------------------------------
@pytest.fixture
def constrained_trainer() -> MAPPOTrainer:
    env = UDTMultiAgentEnv(
        processed_dir=PROCESSED_DIR, n_ticks=30, base_seed=0, constrained_reward=True
    )
    t = MAPPOTrainer(
        env,
        MAPPOConfig(
            rollout_length=32, n_epochs=2, use_action_masking=True, lagrangian_enabled=True
        ),
        seed=0,
    )
    yield t
    env.close()


@pytest.mark.phase7
def test_constrained_train_iteration_runs_without_nan(constrained_trainer: MAPPOTrainer) -> None:
    losses = constrained_trainer.train_iteration()
    for key, value in losses.items():
        assert value == value, f"{key} is NaN"  # NaN != NaN
    assert "mean_violations_per_decision" in losses
    assert "lagrange_lambda" in losses
    assert constrained_trainer.total_steps == 32


@pytest.mark.phase7
def test_action_masking_alone_runs_without_lagrangian() -> None:
    """`use_action_masking` and `lagrangian_enabled` are independent
    flags — confirm masking works with the Lagrangian penalty off too
    (the config every `MAPPOConfig` default leaves it at)."""
    env = UDTMultiAgentEnv(processed_dir=PROCESSED_DIR, n_ticks=30, base_seed=0)
    trainer = MAPPOTrainer(
        env, MAPPOConfig(rollout_length=32, n_epochs=2, use_action_masking=True), seed=0
    )
    losses = trainer.train_iteration()
    for key, value in losses.items():
        assert value == value, f"{key} is NaN"
    assert "lagrange_lambda" not in losses  # Lagrangian off -> no dual-variable logging
    env.close()


@pytest.mark.phase7
def test_lagrange_lambda_dual_ascent_matches_hand_computed_update(
    constrained_trainer: MAPPOTrainer,
) -> None:
    """Direct check of the dual-ascent formula itself, not dependent on
    whether masking happens to let any real violation through in this
    particular 32-step rollout (with masking this well-behaved, it
    usually doesn't — a good sign for masking, but it means a rollout
    driven purely by the real env can't reliably exercise lambda moving
    at all). Collects one real rollout for realistic buffer shapes, then
    overwrites `buffer.costs` with a known nonzero value so the *update
    rule* — not the env's actual violation rate — is what's under test."""
    buffer, last_value, last_cost_value, _ = constrained_trainer.collect_rollout()
    forced_mean_cost = 3.0
    buffer.costs = [forced_mean_cost] * len(buffer)
    lambda_before = constrained_trainer.lagrange_lambda

    constrained_trainer.update(buffer, last_value, last_cost_value)

    expected = max(
        0.0,
        lambda_before
        + constrained_trainer.config.lagrange_lr
        * (forced_mean_cost - constrained_trainer.config.violation_target),
    )
    assert constrained_trainer.lagrange_lambda == pytest.approx(expected)
    assert constrained_trainer.lagrange_lambda > lambda_before  # a real increase, not a no-op


@pytest.mark.phase7
def test_constrained_checkpoint_round_trip_preserves_cost_critic_and_lambda(
    constrained_trainer: MAPPOTrainer, tmp_path: Path
) -> None:
    constrained_trainer.train_iteration()
    checkpoint_path = tmp_path / "mappo_constrained_test.pt"
    constrained_trainer.save(checkpoint_path)

    env2 = UDTMultiAgentEnv(
        processed_dir=PROCESSED_DIR, n_ticks=30, base_seed=0, constrained_reward=True
    )
    fresh = MAPPOTrainer(
        env2, MAPPOConfig(rollout_length=32, lagrangian_enabled=True), seed=999
    )
    fresh.load(checkpoint_path)
    env2.close()

    assert fresh.lagrange_lambda == pytest.approx(constrained_trainer.lagrange_lambda)
    v1 = constrained_trainer.cost_critic(
        torch.cat(
            [
                torch.zeros(constrained_trainer.actors[name].backbone[0].in_features)
                for name in constrained_trainer.agent_names
            ]
        ).unsqueeze(0)
    )
    v2 = fresh.cost_critic(
        torch.cat(
            [
                torch.zeros(fresh.actors[name].backbone[0].in_features)
                for name in fresh.agent_names
            ]
        ).unsqueeze(0)
    )
    assert torch.allclose(v1, v2)
