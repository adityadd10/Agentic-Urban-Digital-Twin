"""Exact MAPPO resume (protocol 2026-09-27 §6.2 acceptance test).

N iterations uninterrupted must equal N/2 -> save_resume -> fresh env and
trainer -> load_resume -> N/2, bit for bit: every logged loss and every final
parameter. Uses the real training config (256-step rollouts). The checkpoint
after 512 steps lands mid-episode (episodes are 96 decision steps), and the
resumed half crosses an episode boundary, so env state really is exercised.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from udt.agents.marl.mappo import MAPPOConfig, MAPPOTrainer
from udt.envs.multi_env import UDTMultiAgentEnv

REPO_ROOT = Path(__file__).resolve().parents[2]
SEED = 3
N = 4


def _trainer() -> MAPPOTrainer:
    env = UDTMultiAgentEnv(processed_dir=REPO_ROOT / "data/processed", n_ticks=288, base_seed=SEED)
    return MAPPOTrainer(env, MAPPOConfig(rollout_length=256), seed=SEED)


def _params(t: MAPPOTrainer) -> dict[str, torch.Tensor]:
    out = {f"critic.{k}": v for k, v in t.critic.state_dict().items()}
    for name, actor in t.actors.items():
        out.update({f"{name}.{k}": v for k, v in actor.state_dict().items()})
    return out


@pytest.mark.phase6
def test_resume_is_bit_identical_to_uninterrupted(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    straight = _trainer()
    straight_losses = [straight.train_iteration() for _ in range(N)]

    first = _trainer()
    for _ in range(N // 2):
        first.train_iteration()
    ckpt = tmp_path / "resume.pt"
    first.save_resume(ckpt, iteration=N // 2, history=[], metadata={"seed": SEED})
    assert first.env.sim is not None and first.env.sim.tick % 288 != 0  # mid-episode
    first.env.close()

    resumed = _trainer()  # fresh env + trainer, as after a crash
    info = resumed.load_resume(ckpt)
    assert info["iteration"] == N // 2 and info["metadata"] == {"seed": SEED}
    resumed_losses = [resumed.train_iteration() for _ in range(N - N // 2)]

    assert resumed_losses == straight_losses[N // 2 :]
    a, b = _params(straight), _params(resumed)
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), k
    assert resumed.total_steps == straight.total_steps == N * 256


@pytest.mark.phase6
def test_resume_refuses_a_different_config(tmp_path: Path) -> None:
    t = _trainer()
    t.save_resume(tmp_path / "r.pt", iteration=0, history=[], metadata={})
    env = UDTMultiAgentEnv(processed_dir=REPO_ROOT / "data/processed", n_ticks=288, base_seed=SEED)
    other = MAPPOTrainer(env, MAPPOConfig(rollout_length=128), seed=SEED)
    with pytest.raises(ValueError, match="config mismatch"):
        other.load_resume(tmp_path / "r.pt")
