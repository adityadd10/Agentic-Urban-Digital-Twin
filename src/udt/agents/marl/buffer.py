"""On-policy rollout buffer + GAE (module M6a slice 2).

**Disclosed simplification:** a single (non-vectorized) environment's
trajectory, not the parallel-envs batch a production MAPPO
implementation would use — matches this session's env throughput
(~46-50 steps/sec, `envs/multi_env.py`'s own measured rate) being the
real bottleneck, not the buffer. `MAPPOTrainer` (`mappo.py`) resets the
environment mid-rollout if an episode ends before the rollout length is
reached, and GAE correctly treats that boundary (bootstraps from 0, not
from a value estimate past the episode's end).

**M7 slice 2 additions (dev doc §5.5's Experiment G):**
- `masks`: the per-head feasibility mask active at each step, when
  `MAPPOConfig.use_action_masking` is on (empty per-agent lists
  otherwise) — stored so `mappo.py`'s `update()` can re-score each
  action under the *same* mask it was sampled under (see `networks.py`'s
  `evaluate()` docstring for why that has to match exactly).
- `costs`/`cost_values`: the per-step constraint-violation count and the
  cost-critic's value estimate for it, for the Lagrangian penalty's own
  GAE pass (`MAPPOConfig.lagrangian_enabled`) — `compute_gae` below now
  takes an explicit `rewards`/`values` override so the *same* GAE
  routine computes both the reward-advantage and the cost-advantage
  (Ray, Achiam & Amodei 2019, "Benchmarking Safe Exploration in Deep
  Reinforcement Learning" — OpenAI Safety Gym's PPO-Lagrangian
  baseline, the algorithm `mappo.py`'s Lagrangian mode implements).
  Default 0.0 when unused, kept in sync with `len(self)` unconditionally
  (simpler than letting these lists go missing/short when masking or
  the Lagrangian penalty is off) — see `mappo.py`'s module docstring for
  the "unclipped cost surrogate" simplification this feeds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class RolloutBuffer:
    """Stores one rollout's transitions. Per-agent action/log-prob/obs
    lists are keyed by agent name; reward/value/done are shared (dev doc
    §5.4: one team reward, one centralized value estimate)."""

    agent_names: list[str]
    obs: dict[str, list[torch.Tensor]] = field(default_factory=dict)
    actions: dict[str, list[torch.Tensor]] = field(default_factory=dict)
    log_probs: dict[str, list[torch.Tensor]] = field(default_factory=dict)
    masks: dict[str, list[list[torch.Tensor]]] = field(default_factory=dict)
    global_states: list[torch.Tensor] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    costs: list[float] = field(default_factory=list)
    cost_values: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        for name in self.agent_names:
            self.obs.setdefault(name, [])
            self.actions.setdefault(name, [])
            self.log_probs.setdefault(name, [])
            self.masks.setdefault(name, [])

    def add(
        self,
        *,
        obs: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor],
        log_probs: dict[str, torch.Tensor],
        global_state: torch.Tensor,
        reward: float,
        value: float,
        done: bool,
        masks: dict[str, list[torch.Tensor]] | None = None,
        cost: float = 0.0,
        cost_value: float = 0.0,
    ) -> None:
        for name in self.agent_names:
            self.obs[name].append(obs[name])
            self.actions[name].append(actions[name])
            self.log_probs[name].append(log_probs[name])
            if masks is not None:
                self.masks[name].append(masks[name])
        self.global_states.append(global_state)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.costs.append(cost)
        self.cost_values.append(cost_value)

    def __len__(self) -> int:
        return len(self.rewards)

    def compute_gae(
        self,
        *,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        rewards: list[float] | None = None,
        values: list[float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Standard GAE(lambda) (Schulman et al. 2015). `last_value` is
        the critic's estimate for the state *after* the rollout's final
        step (0.0 if that step ended an episode — the trainer is
        responsible for passing that correctly, not this function).
        Returns `(advantages, returns)`, each shape `(len(self),)`.

        `rewards`/`values` default to `self.rewards`/`self.values` (the
        team reward + its critic) — passed explicitly by `mappo.py`'s
        Lagrangian mode to run the *same* GAE math over `self.costs`/
        `self.cost_values` instead, rather than duplicating this
        function's recursion for a second signal."""
        rewards = rewards if rewards is not None else self.rewards
        values = values if values is not None else self.values
        n = len(self)
        advantages = torch.zeros(n)
        gae = 0.0
        next_value = last_value
        for t in reversed(range(n)):
            non_terminal = 0.0 if self.dones[t] else 1.0
            delta = rewards[t] + gamma * next_value * non_terminal - values[t]
            gae = delta + gamma * gae_lambda * non_terminal * gae
            advantages[t] = gae
            next_value = values[t]
        values_tensor = torch.tensor(values, dtype=torch.float32)
        returns = advantages + values_tensor
        return advantages, returns
