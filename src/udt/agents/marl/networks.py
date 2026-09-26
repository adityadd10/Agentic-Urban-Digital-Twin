"""MAPPO networks (dev doc §5.5: "MAPPO (CTDE: shared critic sees global
state + g; parameter-shared actors with agent-ID embedding)", module
M6a slice 2).

**Design decision made and disclosed — read before assuming this
matches §5.5 literally:** "parameter-shared actors with agent-ID
embedding" assumes agents homogeneous enough to share one network,
distinguished only by an ID embedding — the usual case in symmetric
settings (e.g. identical unit types in SMAC). Our 3 agents
(`health`/`power`/`transport`, `envs/multi_env.py`) are genuinely
heterogeneous: different observation dimensions (18/12/23 on the real
Kurla graph) and different action spaces (`Discrete(7)` vs.
`MultiDiscrete([4,2])` vs. `MultiDiscrete([2,2,2,2])`). Forcing one
shared network across mismatched input/output shapes would need padding
and masking that adds real complexity for a benefit ("literal" spec
compliance) that doesn't hold in a genuinely heterogeneous team. Resolved:
**independent per-agent actors, one shared centralized critic** — the
critic is where the CTDE property (and dev doc §5.4's "shared team
reward") actually lives (all agents are trained against the *same* value
baseline/advantage signal), and independent actors for heterogeneous
agents is itself a recognized, legitimate MAPPO variant, just not
literally "one shared network".

`g` (dev doc §7.3's goal vector) is omitted from the critic's input, not
padded in as a constant — M6a's own scope note is "fixed goal weights,
no conditioning yet" (that's M6b), so `g` is always the same fixed
vector this module ever sees; a constant input carries no signal for a
network to condition on.

**M7 slice 2 addition (dev doc §5.5's Experiment G: "MAPPO + action
masking"):** `MultiCategoricalActor.forward`/`act`/`evaluate` all take
an optional `mask` — one boolean tensor per action head, `True` = that
discrete choice is feasible under `constraints/engine.py`'s registered
rules (computed by `envs/multi_env.py`'s `action_mask()`, not by this
class — this class just applies whatever mask it's handed). Masked-out
logits are pushed to `_MASK_FILL_VALUE`, not `-inf`: every head's mask
is constructed by the caller to always leave at least one choice
feasible (the "do nothing" option, dev doc §8's action space always
allows that), so `-inf` would be safe too, but a large finite value is
the standard defensive choice — it doesn't rely on that invariant
holding for `softmax` to stay finite.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

_MASK_FILL_VALUE = -1e8


class MultiCategoricalActor(nn.Module):
    """One agent's policy network. `action_nvec` is a list of category
    counts, one per discrete sub-action — a plain `Discrete(n)` action
    space (e.g. `health`) is just `action_nvec=[n]`; a `MultiDiscrete`
    space (e.g. `power`, `transport`) is `action_nvec` verbatim. This
    lets one class handle every agent's action shape uniformly, without
    needing per-agent-type subclasses."""

    def __init__(self, obs_dim: int, action_nvec: list[int], hidden_dim: int = 64) -> None:
        super().__init__()
        self.action_nvec = list(action_nvec)
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, n) for n in self.action_nvec])

    def forward(
        self, obs: torch.Tensor, mask: list[torch.Tensor] | None = None
    ) -> list[Categorical]:
        """Returns one `Categorical` distribution per sub-action head —
        callers combine per-head log-probs by summing them (the joint
        action's log-prob is the sum of independent categoricals' log-
        probs), matching how `MultiDiscrete.sample()` already treats
        each sub-action as independent.

        `mask[i]` (if given), a bool tensor shaped like head `i`'s
        logits (`(n_i,)` unbatched or `(T, n_i)` batched — same shape
        `obs` itself is batched at), zeroes out that head's infeasible
        choices' probability mass before sampling (M7 slice 2, dev doc
        §5.5's Experiment G)."""
        features = self.backbone(obs)
        dists = []
        for i, head in enumerate(self.heads):
            logits = head(features)
            if mask is not None:
                logits = logits.masked_fill(~mask[i], _MASK_FILL_VALUE)
            dists.append(Categorical(logits=logits))
        return dists

    def act(
        self,
        obs: torch.Tensor,
        *,
        deterministic: bool = False,
        mask: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns `(action, log_prob)` — `action` has shape
        `(len(action_nvec),)`, `log_prob` is the joint log-prob (summed
        across sub-action heads), a scalar."""
        dists = self.forward(obs, mask=mask)
        if deterministic:
            actions = [d.probs.argmax(dim=-1) for d in dists]
        else:
            actions = [d.sample() for d in dists]
        log_prob = sum(d.log_prob(a) for d, a in zip(dists, actions, strict=True))
        return torch.stack(actions), log_prob

    def evaluate(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        *,
        mask: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns `(log_prob, entropy)` for an already-taken `action` —
        used during PPO updates to compute the new policy's probability
        ratio against the old one stored in the rollout buffer. `mask`
        must be the *same* per-head masks that were active when `action`
        was sampled (`agents/marl/mappo.py`'s `update()` reconstructs
        this from the rollout buffer) — re-scoring under a different
        mask would compute a log-prob for a distribution the action
        wasn't actually drawn from, corrupting PPO's importance-sampling
        ratio."""
        dists = self.forward(obs, mask=mask)
        log_prob = sum(d.log_prob(action[..., i]) for i, d in enumerate(dists))
        entropy = sum(d.entropy() for d in dists)
        return log_prob, entropy


class CentralizedCritic(nn.Module):
    """The one shared value network (dev doc §5.5's CTDE critic) — takes
    the *global* state (all agents' observations concatenated, see
    `agents/marl/mappo.py`'s `build_global_state`), outputs one scalar
    value shared by every agent, matching dev doc §5.4's shared team
    reward."""

    def __init__(self, global_state_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.net(global_state).squeeze(-1)
