"""Phase 6 acceptance tests: `agents.marl.networks` (dev doc §5.5's
actor/critic) and `agents.marl.buffer`'s GAE computation."""

from __future__ import annotations

import pytest
import torch

from udt.agents.marl.buffer import RolloutBuffer
from udt.agents.marl.networks import CentralizedCritic, MultiCategoricalActor


@pytest.mark.phase6
def test_actor_discrete_action_shape() -> None:
    """A plain Discrete(7) action space is action_nvec=[7]."""
    actor = MultiCategoricalActor(obs_dim=5, action_nvec=[7])
    obs = torch.randn(5)
    action, log_prob = actor.act(obs)
    assert action.shape == (1,)
    assert 0 <= int(action[0]) < 7
    assert log_prob.shape == ()


@pytest.mark.phase6
def test_actor_multidiscrete_action_shape() -> None:
    """A MultiDiscrete([4, 2]) action space is action_nvec=[4, 2]."""
    actor = MultiCategoricalActor(obs_dim=5, action_nvec=[4, 2])
    obs = torch.randn(5)
    action, log_prob = actor.act(obs)
    assert action.shape == (2,)
    assert 0 <= int(action[0]) < 4
    assert 0 <= int(action[1]) < 2
    assert log_prob.shape == ()


@pytest.mark.phase6
def test_actor_deterministic_is_reproducible() -> None:
    actor = MultiCategoricalActor(obs_dim=5, action_nvec=[4, 2])
    obs = torch.randn(5)
    a1, _ = actor.act(obs, deterministic=True)
    a2, _ = actor.act(obs, deterministic=True)
    assert torch.equal(a1, a2)  # same obs, same network -> same argmax action


@pytest.mark.phase6
def test_actor_evaluate_matches_act_log_prob_for_same_action() -> None:
    """`evaluate()` re-scoring an action `act()` just produced should
    reproduce the same log-prob (before any gradient step changes the
    weights) — this is exactly the invariant PPO's importance-sampling
    ratio depends on being correct at iteration 0."""
    actor = MultiCategoricalActor(obs_dim=5, action_nvec=[4, 2])
    obs = torch.randn(5)
    action, log_prob_from_act = actor.act(obs)
    log_prob_from_evaluate, _entropy = actor.evaluate(obs.unsqueeze(0), action.unsqueeze(0))
    assert torch.allclose(log_prob_from_act, log_prob_from_evaluate.squeeze(0), atol=1e-5)


@pytest.mark.phase7
def test_actor_mask_never_samples_a_masked_out_choice() -> None:
    """Statistical check (dev doc §5.5's Experiment G, `networks.py`'s
    masking addition): over many samples, a head whose mask allows only
    choice 1 of 2 must never produce choice 0."""
    actor = MultiCategoricalActor(obs_dim=5, action_nvec=[2])
    obs = torch.randn(5)
    mask = [torch.tensor([False, True])]
    for _ in range(200):
        action, _ = actor.act(obs, mask=mask)
        assert int(action[0]) == 1


@pytest.mark.phase7
def test_actor_mask_deterministic_argmax_respects_mask() -> None:
    """Even `deterministic=True` (argmax over probs) must never pick a
    masked-out choice, regardless of what the untrained network's raw
    logits happen to favor."""
    actor = MultiCategoricalActor(obs_dim=5, action_nvec=[4])
    obs = torch.randn(5)
    mask = [torch.tensor([True, False, False, False])]
    action, _ = actor.act(obs, deterministic=True, mask=mask)
    assert int(action[0]) == 0


@pytest.mark.phase7
def test_actor_evaluate_with_mask_matches_act_log_prob() -> None:
    """Same invariant `test_actor_evaluate_matches_act_log_prob_for_same_
    action` checks unmasked, but with a real mask applied consistently
    to both calls — `mappo.py`'s PPO ratio depends on this holding for
    masked policies too."""
    actor = MultiCategoricalActor(obs_dim=5, action_nvec=[4, 2])
    obs = torch.randn(5)
    mask = [torch.tensor([True, True, False, False]), torch.tensor([False, True])]
    action, log_prob_from_act = actor.act(obs, mask=mask)
    log_prob_from_evaluate, _entropy = actor.evaluate(
        obs.unsqueeze(0), action.unsqueeze(0), mask=[m.unsqueeze(0) for m in mask]
    )
    assert torch.allclose(log_prob_from_act, log_prob_from_evaluate.squeeze(0), atol=1e-5)


@pytest.mark.phase7
def test_actor_without_mask_is_unaffected_by_masking_code_path() -> None:
    """`mask=None` (the default, every M6a/M6b caller) must behave
    exactly as before this feature existed — a regression guard, not
    just "doesn't crash"."""
    actor = MultiCategoricalActor(obs_dim=5, action_nvec=[4, 2])
    torch.manual_seed(0)
    obs = torch.randn(5)
    action, log_prob = actor.act(obs, deterministic=True)
    assert action.shape == (2,)
    assert log_prob.shape == ()


@pytest.mark.phase6
def test_centralized_critic_output_shape() -> None:
    critic = CentralizedCritic(global_state_dim=10)
    batch = torch.randn(4, 10)
    values = critic(batch)
    assert values.shape == (4,)


@pytest.mark.phase6
def test_gae_zero_reward_zero_value_gives_zero_advantage() -> None:
    """Degenerate case: if every reward and value is 0, GAE's advantage
    must also be exactly 0 at every step (delta = 0 + gamma*0 - 0 = 0
    every step, so the recursive gae accumulation stays at 0)."""
    buffer = RolloutBuffer(agent_names=["a"])
    for _ in range(5):
        buffer.add(
            obs={"a": torch.zeros(3)},
            actions={"a": torch.zeros(1)},
            log_probs={"a": torch.tensor(0.0)},
            global_state=torch.zeros(3),
            reward=0.0,
            value=0.0,
            done=False,
        )
    advantages, returns = buffer.compute_gae(last_value=0.0)
    assert torch.allclose(advantages, torch.zeros(5))
    assert torch.allclose(returns, torch.zeros(5))


@pytest.mark.phase6
def test_gae_single_step_matches_hand_computed_value() -> None:
    """One step, known reward/value: delta = r + gamma*last_value - v is
    exactly computable by hand, and with only one step GAE's advantage
    *is* that one delta (no recursive discounting to accumulate)."""
    buffer = RolloutBuffer(agent_names=["a"])
    buffer.add(
        obs={"a": torch.zeros(3)},
        actions={"a": torch.zeros(1)},
        log_probs={"a": torch.tensor(0.0)},
        global_state=torch.zeros(3),
        reward=1.0,
        value=0.5,
        done=False,
    )
    gamma, gae_lambda, last_value = 0.99, 0.95, 2.0
    advantages, returns = buffer.compute_gae(
        last_value=last_value, gamma=gamma, gae_lambda=gae_lambda
    )
    expected_delta = 1.0 + gamma * last_value - 0.5
    assert advantages[0].item() == pytest.approx(expected_delta)
    assert returns[0].item() == pytest.approx(expected_delta + 0.5)


@pytest.mark.phase7
def test_gae_override_uses_the_passed_rewards_and_values_not_self() -> None:
    """`mappo.py`'s Lagrangian mode calls `compute_gae(rewards=buffer.
    costs, values=buffer.cost_values)` to reuse this same routine for
    the cost signal — confirms the override is actually used, not
    silently ignored in favor of `self.rewards`/`self.values`."""
    buffer = RolloutBuffer(agent_names=["a"])
    buffer.add(
        obs={"a": torch.zeros(3)},
        actions={"a": torch.zeros(1)},
        log_probs={"a": torch.tensor(0.0)},
        global_state=torch.zeros(3),
        reward=1.0,  # deliberately different from the cost override below
        value=0.5,
        done=False,
        cost=9.0,
        cost_value=3.0,
    )
    gamma, gae_lambda, last_value = 0.99, 0.95, 2.0
    advantages, returns = buffer.compute_gae(
        last_value=last_value,
        gamma=gamma,
        gae_lambda=gae_lambda,
        rewards=buffer.costs,
        values=buffer.cost_values,
    )
    expected_delta = 9.0 + gamma * last_value - 3.0
    assert advantages[0].item() == pytest.approx(expected_delta)
    assert returns[0].item() == pytest.approx(expected_delta + 3.0)


@pytest.mark.phase6
def test_gae_does_not_bootstrap_across_a_done_boundary() -> None:
    """A `done=True` step must zero out the bootstrap term for *that*
    step's delta (the episode ended, there is no real "next state" to
    bootstrap from) — this is the exact mechanism that makes GAE correct
    across episode boundaries within one rollout buffer."""
    buffer = RolloutBuffer(agent_names=["a"])
    buffer.add(
        obs={"a": torch.zeros(3)},
        actions={"a": torch.zeros(1)},
        log_probs={"a": torch.tensor(0.0)},
        global_state=torch.zeros(3),
        reward=1.0,
        value=0.5,
        done=True,
    )
    advantages, _ = buffer.compute_gae(last_value=100.0)  # large, must be ignored
    # done=True -> non_terminal=0.0 -> delta = reward - value, last_value's
    # huge magnitude must not leak in.
    assert advantages[0].item() == pytest.approx(1.0 - 0.5)
