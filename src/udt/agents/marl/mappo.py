"""MAPPO trainer (dev doc §5.5, module M6a slice 2 — the actual learning
part; `envs/multi_env.py`, M6a slice 1, was the arena only).

**Disclosed simplifications, same discipline as `scripts/train.py`'s
SB3/PPO docstring:**
- Single (non-vectorized) environment rollouts — matches `multi_env.py`'s
  own measured throughput (~46-50 steps/sec) being the real bottleneck,
  not the buffer/update step. `MAPPOTrainer.collect_rollout` resets the
  environment mid-rollout on episode end.
- No minibatching or shuffling across the rollout — each PPO epoch is one
  full-batch gradient step over the whole rollout. Standard PPO uses
  minibatches to reduce gradient variance/memory at scale; for this
  environment's rollout sizes (hundreds, not the tens-of-thousands a
  production MARL benchmark would use), full-batch updates are a
  reasonable, disclosed simplification for a first correct
  implementation, not a shortcut that changes what the algorithm *does*.
- One combined optimizer over every actor's parameters *and* the shared
  critic's — simpler to implement correctly than separate per-network
  optimizers, and a legitimate, common choice when actor/critic don't
  share a backbone (they don't here — see `networks.py`).
- No learning-rate schedule / no observation normalization — dev doc
  §5.5 doesn't specify either; both are common real-world additions
  worth doing before trusting a long real training run, not before a
  correctness smoke test.

See `networks.py`'s module docstring for the "independent actors, shared
critic" design decision (dev doc §5.5 literally says "parameter-shared
actors with agent-ID embedding" — not what's implemented, and why).

**M7 slice 2 additions (dev doc §5.5's Experiment G, RQ3/H3):**
- `MAPPOConfig.use_action_masking`: when `True`, `collect_rollout` fetches
  `env.action_mask(name)` (`envs/multi_env.py`, M7 slice 2) before each
  actor samples, and `update()` re-scores actions under the *same*
  masks (stored in the rollout buffer) — see `networks.py`'s
  `evaluate()` docstring for why the mask has to match exactly.
- `MAPPOConfig.lagrangian_enabled`: **PPO-Lagrangian** (Ray, Achiam &
  Amodei 2019, "Benchmarking Safe Exploration in Deep Reinforcement
  Learning" — OpenAI Safety Gym's own baseline algorithm for exactly
  this "PPO + a dual-ascent-updated penalty on a separate cost signal"
  recipe): a second `CentralizedCritic` (`self.cost_critic`, reused
  as-is — it's just a scalar value function, agnostic to what the
  scalar means) estimates the *cost* value (per-decision constraint-
  violation count, from `multi_env.py`'s `infos[...]["safety_
  violations_attempted"]`), GAE'd with the same `RolloutBuffer.
  compute_gae` routine `rewards`/`values` already use (now
  parameterized to accept an override, so cost and reward share one GAE
  implementation). The actor loss adds `self.lagrange_lambda *
  (ratio * cost_advantage).mean()` — an **unclipped** cost surrogate,
  a disclosed simplification against the reward objective's PPO-
  clipped one (clipping the cost term too is a legitimate refinement
  Ray et al.'s own follow-up work does; the unclipped version is still
  a real, correctly-derived policy-gradient term, not a placeholder).
  `self.lagrange_lambda` (a scalar dual variable, initialized 0.0) is
  updated once per `train_iteration`, *after* the policy update, via
  plain dual ascent against `MAPPOConfig.violation_target` (dev doc's
  own "~0 executed violations" acceptance bar, default 0.0).
- Both default to `False`/off — every M6a/M6b caller (unmasked,
  unconstrained MAPPO) is unaffected; `scripts/train_mappo.py` gained a
  `--constrained` flag that turns both on together, plus `multi_env.py`'s
  `constrained_reward=True`, matching dev doc Table row G as one unit.
- **Not done, disclosed rather than faked:** running this at a scale
  that could actually validate Phase 7's own acceptance criterion
  ("constrained policy: ~0 executed violations, >=90% of unconstrained
  reward on val") — same "mechanism verified, not validated at scale"
  status every RL module this session has been built at (M5/M6a/M6b's
  own rows say the same about their own training runs).
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from torch import nn

from udt.agents.marl.buffer import RolloutBuffer
from udt.agents.marl.networks import CentralizedCritic, MultiCategoricalActor
from udt.envs.multi_env import UDTMultiAgentEnv


def build_global_state(obs: dict[str, torch.Tensor], agent_order: list[str]) -> torch.Tensor:
    """Concatenates every agent's local observation into one global-state
    vector for the centralized critic (dev doc §5.5). Disclosed
    redundancy: `multi_env.py`'s common observation block (critical-asset
    levels, incident severity, tick-of-day) is repeated once per agent
    inside each local obs, so it appears `len(agent_order)` times here —
    harmless (the critic just sees the same numbers more than once), not
    worth a separate "common-block-once" code path for a first
    implementation."""
    return torch.cat([obs[name] for name in agent_order])


def _action_nvec(env: UDTMultiAgentEnv, agent: str) -> list[int]:
    space = env.action_space(agent)
    if isinstance(space, spaces.MultiDiscrete):
        return [int(n) for n in space.nvec]
    if isinstance(space, spaces.Discrete):
        return [int(space.n)]
    raise TypeError(f"unsupported action space type for agent {agent!r}: {type(space)}")


def _obs_dim(env: UDTMultiAgentEnv, agent: str) -> int:
    """`Space.shape` is typed as `tuple[int, ...] | None` in general
    (some Gymnasium space types have no shape) — every agent's
    observation space here is a `Box`, which always has one, but this
    makes that assumption explicit instead of indexing a possibly-`None`
    value directly."""
    shape = env.observation_space(agent).shape
    assert shape is not None, f"agent {agent!r}'s observation space has no shape"
    return int(shape[0])


@dataclass
class MAPPOConfig:
    rollout_length: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 3e-4
    n_epochs: int = 4
    max_grad_norm: float = 0.5
    # M7 slice 2 (dev doc §5.5's Experiment G) — see module docstring.
    use_action_masking: bool = False
    lagrangian_enabled: bool = False
    lagrange_lr: float = 0.01
    violation_target: float = 0.0


class MAPPOTrainer:
    """Owns the actors/critic and the rollout-then-update loop. One
    `train_iteration()` = collect `config.rollout_length` steps, then
    `config.n_epochs` full-batch PPO update passes."""

    def __init__(
        self,
        env: UDTMultiAgentEnv,
        config: MAPPOConfig | None = None,
        *,
        seed: int = 0,
    ) -> None:
        self.env = env
        self.config = config or MAPPOConfig()
        self.agent_names = list(env.possible_agents)
        torch.manual_seed(seed)

        self.actors = {
            name: MultiCategoricalActor(
                obs_dim=_obs_dim(env, name),
                action_nvec=_action_nvec(env, name),
            )
            for name in self.agent_names
        }
        global_state_dim = sum(_obs_dim(env, name) for name in self.agent_names)
        self.critic = CentralizedCritic(global_state_dim)
        # M7 slice 2: the Lagrangian penalty's own value function, over
        # the cost signal (per-decision violation count) rather than the
        # team reward — same architecture, different target, see module
        # docstring's PPO-Lagrangian citation.
        self.cost_critic = CentralizedCritic(global_state_dim)
        self.lagrange_lambda = 0.0

        params: list[torch.nn.Parameter] = list(self.critic.parameters())
        for actor in self.actors.values():
            params += list(actor.parameters())
        if self.config.lagrangian_enabled:
            params += list(self.cost_critic.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.config.learning_rate)

        self._obs: dict[str, torch.Tensor] | None = None
        self.total_steps = 0

    def _reset(self, seed: int | None = None) -> None:
        obs, _ = self.env.reset(seed=seed)
        self._obs = {
            name: torch.as_tensor(obs[name], dtype=torch.float32) for name in self.agent_names
        }

    def collect_rollout(self) -> tuple[RolloutBuffer, float, float, list[float]]:
        if self._obs is None:
            self._reset()
        buffer = RolloutBuffer(self.agent_names)
        episode_rewards: list[float] = []
        current_episode_reward = 0.0

        for _ in range(self.config.rollout_length):
            assert self._obs is not None
            actions: dict[str, torch.Tensor] = {}
            log_probs: dict[str, torch.Tensor] = {}
            masks: dict[str, list[torch.Tensor]] | None = (
                {} if self.config.use_action_masking else None
            )
            with torch.no_grad():
                for name in self.agent_names:
                    mask = None
                    if masks is not None:
                        mask = [
                            torch.as_tensor(m, dtype=torch.bool) for m in self.env.action_mask(name)
                        ]
                        masks[name] = mask
                    action, log_prob = self.actors[name].act(self._obs[name], mask=mask)
                    actions[name] = action
                    log_probs[name] = log_prob
                global_state = build_global_state(self._obs, self.agent_names)
                value = self.critic(global_state.unsqueeze(0)).item()
                cost_value = (
                    self.cost_critic(global_state.unsqueeze(0)).item()
                    if self.config.lagrangian_enabled
                    else 0.0
                )

            env_actions = {name: actions[name].numpy() for name in self.agent_names}
            next_obs, rewards, terminations, truncations, infos = self.env.step(env_actions)
            reward = rewards[self.agent_names[0]]  # shared team reward, dev doc §5.4
            # M7 slice 2: the Lagrangian cost signal — same value for
            # every agent's info (one joint-action check per decision),
            # so any agent's entry works, same convention `reward` above
            # already uses for the shared team reward.
            cost = float(infos[self.agent_names[0]].get("safety_violations_attempted", 0))
            done = any(terminations.values()) or any(truncations.values())

            buffer.add(
                obs=self._obs,
                actions=actions,
                log_probs=log_probs,
                global_state=global_state,
                reward=reward,
                value=value,
                done=done,
                masks=masks,
                cost=cost,
                cost_value=cost_value,
            )
            self.total_steps += 1
            current_episode_reward += reward

            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0.0
                self._reset()
            else:
                self._obs = {
                    name: torch.as_tensor(next_obs[name], dtype=torch.float32)
                    for name in self.agent_names
                }

        with torch.no_grad():
            if buffer.dones[-1]:
                last_value = 0.0
                last_cost_value = 0.0
            else:
                assert self._obs is not None
                final_global_state = build_global_state(self._obs, self.agent_names).unsqueeze(0)
                last_value = self.critic(final_global_state).item()
                last_cost_value = (
                    self.cost_critic(final_global_state).item()
                    if self.config.lagrangian_enabled
                    else 0.0
                )

        return buffer, last_value, last_cost_value, episode_rewards

    def update(
        self, buffer: RolloutBuffer, last_value: float, last_cost_value: float = 0.0
    ) -> dict[str, float]:
        advantages, returns = buffer.compute_gae(
            last_value=last_value, gamma=self.config.gamma, gae_lambda=self.config.gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # M7 slice 2: the Lagrangian penalty's own GAE pass, over
        # `buffer.costs`/`buffer.cost_values` instead of the team reward
        # — same routine, same normalization convention as `advantages`
        # above (see module docstring's PPO-Lagrangian citation).
        cost_advantages: torch.Tensor | None = None
        cost_returns: torch.Tensor | None = None
        if self.config.lagrangian_enabled:
            cost_advantages, cost_returns = buffer.compute_gae(
                last_value=last_cost_value,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
                rewards=buffer.costs,
                values=buffer.cost_values,
            )
            cost_advantages = (cost_advantages - cost_advantages.mean()) / (
                cost_advantages.std() + 1e-8
            )

        global_states = torch.stack(buffer.global_states)
        old_log_probs = {name: torch.stack(buffer.log_probs[name]) for name in self.agent_names}
        obs_batch = {name: torch.stack(buffer.obs[name]) for name in self.agent_names}
        action_batch = {name: torch.stack(buffer.actions[name]) for name in self.agent_names}

        # M7 slice 2: reconstruct each agent's per-head mask batch,
        # shape (rollout_length, n_categories) per head, from the
        # buffer's per-timestep, per-head list-of-tensors — `None` for
        # any agent the rollout recorded no masks for (masking off, or
        # an agent with nothing to mask, e.g. `power`, still gets a
        # real all-True mask from `action_mask()`, so this is really
        # just "masking was off entirely" here).
        mask_batches: dict[str, list[torch.Tensor] | None] = {}
        for name in self.agent_names:
            agent_masks = buffer.masks.get(name)
            if agent_masks:
                n_heads = len(agent_masks[0])
                mask_batches[name] = [
                    torch.stack([step_masks[h] for step_masks in agent_masks])
                    for h in range(n_heads)
                ]
            else:
                mask_batches[name] = None

        params: list[torch.nn.Parameter] = list(self.critic.parameters())
        for actor in self.actors.values():
            params += list(actor.parameters())
        if self.config.lagrangian_enabled:
            params += list(self.cost_critic.parameters())

        last_losses: dict[str, float] = {}
        for _ in range(self.config.n_epochs):
            values = self.critic(global_states)
            value_loss = nn.functional.mse_loss(values, returns)
            cost_value_loss = torch.tensor(0.0)
            if self.config.lagrangian_enabled:
                assert cost_returns is not None
                cost_value_loss = nn.functional.mse_loss(
                    self.cost_critic(global_states), cost_returns
                )

            total_actor_loss = torch.tensor(0.0)
            total_cost_surrogate = torch.tensor(0.0)
            total_entropy = torch.tensor(0.0)
            for name in self.agent_names:
                new_log_probs, entropy = self.actors[name].evaluate(
                    obs_batch[name], action_batch[name], mask=mask_batches[name]
                )
                ratio = torch.exp(new_log_probs - old_log_probs[name].detach())
                surr1 = ratio * advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range)
                    * advantages
                )
                total_actor_loss = total_actor_loss + -torch.min(surr1, surr2).mean()
                total_entropy = total_entropy + entropy.mean()
                if self.config.lagrangian_enabled:
                    assert cost_advantages is not None
                    # Unclipped cost surrogate — disclosed simplification
                    # against the reward objective's PPO-clipped one, see
                    # module docstring.
                    total_cost_surrogate = total_cost_surrogate + (ratio * cost_advantages).mean()

            loss = (
                total_actor_loss
                + self.config.value_loss_coef * value_loss
                - self.config.entropy_coef * total_entropy
            )
            if self.config.lagrangian_enabled:
                loss = (
                    loss
                    + self.lagrange_lambda * total_cost_surrogate
                    + self.config.value_loss_coef * cost_value_loss
                )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, self.config.max_grad_norm)
            self.optimizer.step()

            last_losses = {
                "actor_loss": total_actor_loss.item(),
                "value_loss": value_loss.item(),
                "entropy": total_entropy.item(),
            }
            if self.config.lagrangian_enabled:
                last_losses["cost_surrogate"] = total_cost_surrogate.item()
                last_losses["cost_value_loss"] = cost_value_loss.item()

        # M7 slice 2: dual ascent on lambda, *after* the policy update,
        # against this rollout's realized mean cost (the raw per-decision
        # violation count, not its advantage) — plain PPO-Lagrangian dual
        # ascent (module docstring's citation), pushing lambda up when the
        # policy is violating more than `violation_target` and down
        # (never below 0) once it isn't.
        if self.config.lagrangian_enabled:
            mean_violations = float(np.mean(buffer.costs)) if buffer.costs else 0.0
            self.lagrange_lambda = max(
                0.0,
                self.lagrange_lambda
                + self.config.lagrange_lr * (mean_violations - self.config.violation_target),
            )
            last_losses["mean_violations_per_decision"] = mean_violations
            last_losses["lagrange_lambda"] = self.lagrange_lambda

        return last_losses

    def train_iteration(self) -> dict[str, float]:
        buffer, last_value, last_cost_value, episode_rewards = self.collect_rollout()
        losses = self.update(buffer, last_value, last_cost_value)
        losses["mean_episode_reward"] = (
            float(np.mean(episode_rewards)) if episode_rewards else float("nan")
        )
        losses["n_episodes_completed"] = float(len(episode_rewards))
        losses["total_steps"] = float(self.total_steps)
        return losses

    def save(self, path: str | Path) -> None:
        checkpoint: dict[str, Any] = {
            "actors": {name: actor.state_dict() for name, actor in self.actors.items()},
            "critic": self.critic.state_dict(),
            "total_steps": self.total_steps,
        }
        # M7 slice 2: only present when the Lagrangian mode was actually
        # used — a checkpoint from an unconstrained run has no cost
        # critic to save, and `load` treats their absence as "this
        # checkpoint predates/doesn't use the Lagrangian mode", not an
        # error.
        if self.config.lagrangian_enabled:
            checkpoint["cost_critic"] = self.cost_critic.state_dict()
            checkpoint["lagrange_lambda"] = self.lagrange_lambda
        torch.save(checkpoint, path)

    RESUME_FORMAT = "mappo_resume_v1"

    def save_resume(
        self,
        path: str | Path,
        *,
        iteration: int,
        history: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> None:
        """Full training state for an exact resume (protocol 2026-09-27 §6.2):
        networks, optimiser, counters, current observation, torch/numpy/python
        RNG states, the env's in-progress state, config and metadata. Written
        atomically (temp file + rename) so an interruption mid-save can't
        corrupt the previous resume point. `save()` (final models) is unchanged."""
        state = {
            "format": self.RESUME_FORMAT,
            "config": asdict(self.config),
            "actors": {name: actor.state_dict() for name, actor in self.actors.items()},
            "critic": self.critic.state_dict(),
            "cost_critic": self.cost_critic.state_dict(),
            "lagrange_lambda": self.lagrange_lambda,
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "iteration": iteration,
            "history": history,
            "obs": self._obs,
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "env_state": self.env.resume_state(),
            "metadata": metadata,
        }
        tmp = Path(f"{path}.tmp")
        torch.save(state, tmp)
        os.replace(tmp, path)

    def load_resume(self, path: str | Path) -> dict[str, Any]:
        """Restore `save_resume` state into this trainer (built with the same
        config and a freshly constructed env). Returns iteration, history and
        metadata so the caller continues from `iteration + 1`."""
        state: dict[str, Any] = torch.load(path, weights_only=False)
        if state.get("format") != self.RESUME_FORMAT:
            raise ValueError(f"{path} is not a {self.RESUME_FORMAT} checkpoint")
        if state["config"] != asdict(self.config):
            raise ValueError(f"config mismatch: saved {state['config']} vs {asdict(self.config)}")
        for name, actor in self.actors.items():
            actor.load_state_dict(state["actors"][name])
        self.critic.load_state_dict(state["critic"])
        self.cost_critic.load_state_dict(state["cost_critic"])
        self.lagrange_lambda = state["lagrange_lambda"]
        self.optimizer.load_state_dict(state["optimizer"])
        self.total_steps = state["total_steps"]
        self.env.load_resume_state(state["env_state"])
        self._obs = state["obs"]
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        return {k: state[k] for k in ("iteration", "history", "metadata")}

    def load(self, path: str | Path) -> None:
        checkpoint: dict[str, Any] = torch.load(path, weights_only=True)
        for name, actor in self.actors.items():
            actor.load_state_dict(checkpoint["actors"][name])
        self.critic.load_state_dict(checkpoint["critic"])
        self.total_steps = checkpoint.get("total_steps", 0)
        if "cost_critic" in checkpoint:
            self.cost_critic.load_state_dict(checkpoint["cost_critic"])
            self.lagrange_lambda = checkpoint.get("lagrange_lambda", 0.0)
