"""Multi-agent PettingZoo environment (dev doc §5.1-§5.4, module M6a,
extended in M6b with goal-conditioning).

PettingZoo `ParallelEnv`, 3 agents: `health`, `power`, `transport` (water
folded into `power`, dev doc §5.1's own note: water's only lever is
power allocation + reserve usage — a real architectural reading of the
dev doc's text, not a workaround for the real Kurla graph's water-facility
count, which used to be 0 and is now 2, see `data/manual_facilities.yaml`
/ `MTP_Module_Planner.md`'s M2 revision). Wraps the *same* twin `single_env.py` (M5)
does — `Simulator`, `RoadNetwork`, `twin/ambulances.py`, `twin/demand.py`,
`twin/power.py`, all M1-M4 code, unchanged — just observed/acted on by 3
agents instead of 1.

**This module deliberately does not share code with `single_env.py`.**
Refactoring both into a shared base mid-session risked destabilizing
`single_env.py`'s already-tested behavior for a modest benefit; the
duplication here is the twin-setup boilerplate only (loading the graph/
raster/road network/ward polygon), not the twin itself.

**Disclosed scope, same discipline as `single_env.py`:**
- **Reward is the same scalar for every agent, every step** — dev doc
  §5.4 explicitly: "shared team reward; identical for single- and
  multi-agent". `resource_cost` stays a disclosed 0.0 (no cost model
  exists). `water_shortage_hours` also stays 0.0, but no longer because
  of a missing water facility (M2 revision added 2, see
  `data/manual_facilities.yaml`) — it's blocked on `TwinState` not
  carrying per-edge `EdgeRuntimeState`/satisfaction in its tick trace,
  needed to isolate the water-specific term from a hospital's combined
  cascade product (same gap `single_env.py`'s own docstring now
  describes in full) — everything else is now weighted by the sampled
  goal vector `g`, not fixed at 1.0.
- **`safety_violations` (M7 addition, dev doc §8/§5.4):** the joint
  action decoded from all 3 agents' choices is checked against
  `constraints/engine.py`'s registered rules *before* being applied —
  the repaired (clipped/dropped) action is what actually executes. By
  default (`constrained_reward=False`, the "UNCONSTRAINED variant" per
  dev doc §5.4's own footnote) the count of violations caught is
  penalized in the reward at the dev doc's exact `-20.0` coefficient.
- **`action_mask()` + `constrained_reward` (M7 slice 2, dev doc §5.5's
  Experiment G — "MAPPO + action masking [...] + Lagrangian penalty on
  residual soft constraints"):** `action_mask(agent)` returns, per
  action head, which discrete choices are feasible under the same
  registered rules — `agents/marl/mappo.py`'s trainer applies it *before*
  sampling when `MAPPOConfig.use_action_masking` is set, so a masked
  policy essentially never proposes what `check(...)` above would have
  had to repair. `constrained_reward=True` then drops the `-20.0`
  reward term (dev doc's footnote: a constrained variant "never
  executes violations [...] so there the count of attempted violations
  is logged as a metric instead") — the `episode_trace`/H3 metric keeps
  counting them regardless of this flag, only the reward differs. Both
  default to their unconstrained M7-slice-1 behavior (`False`) unless a
  caller opts in.
- **Goal-conditioning (M6b, dev doc §5.4/§5.5):** `g = (g_health,
  g_power, g_transport, g_cost)` is sampled uniformly in `[0.5, 2.0]`
  (dev doc's exact box) once per episode (`reset()`), included in every
  agent's observation (dev doc §5.2's common block already lists it;
  M6a simply had nothing to put there yet), and used to weight the
  corresponding reward terms (`_tick_reward`). `g_cost` is sampled and
  observed like the other three, but currently has zero effect on the
  reward — it multiplies `resource_cost`, which is still 0.0 — disclosed
  as a wired-but-currently-inert component, not hidden. `new_cascade_
  failures`/`safety_violations`/`water_shortage_hours` keep their fixed
  dev-doc coefficients (5.0/20.0/1.0) — only the four `g_*`-named terms
  are goal-weighted, per §5.4's own formula.
- **Action split follows dev doc §5.3's literal per-agent assignment,
  narrower than M4/M5's single-agent generalization:** §5.3 lists
  repair-crew target under `power` ("none, S1, S2, S3" — substations),
  not under `health`. M4's `RuleBasedAgent` (one unified agent, only 1
  real substation to work with) generalized repair to also cover
  hospitals; here, `power` is the only agent with a repair lever, and it
  targets substations/water only. Hospitals still recover automatically
  via the cascade formula once their dependencies are restored — they
  just have no *direct* repair action in this split, matching spec.
- **Omitted, not faked as no-ops:** dev doc §5.3's "per hospital:
  {normal, surge-mode}" and "per water facility: {run, reserve-only}" —
  no surge-mode consequence exists in the twin (the same category of gap
  load-shedding was in before `twin/power.py` existed), and no
  reserve-only-mode consequence exists for water facilities either
  (having 2 real, M2-revision water assets now, see
  `data/manual_facilities.yaml`, doesn't by itself give this lever
  anything to do — that needs its own twin mechanic, still unbuilt).
- MAPPO training itself lives in `agents/marl/` — this file only adds
  the goal-conditioning *interface* (sampling, observing, weighting);
  `agents/marl/mappo.py`'s critic already sees `g` for free, since it's
  now part of every agent's local observation (dev doc §5.5: "shared
  critic sees global state + g" — the global state IS the concatenated
  local observations, which now include `g`).
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium import spaces
from pettingzoo import ParallelEnv
from shapely.geometry import shape

from udt.common.models import AgentAction, AssetType, DependencyGraph, Incident, TwinState
from udt.common.versions import ENV_VERSION
from udt.constraints.engine import check
from udt.incidents.degradations.flood import SusceptibilityRaster, make_flood_degradation_fn
from udt.scenarios.generator import (
    apply_initial_conditions,
    generate_flood_scenario,
    onset_hour_of_day,
)
from udt.twin.ambulances import generate_requests, spawn_ambulances
from udt.twin.graph import dependency_edges_of
from udt.twin.power import SHED_FRACTION_BY_TIER, update_substation_load
from udt.twin.road_network import RoadNetwork
from udt.twin.simulator import Simulator

DECISION_INTERVAL_TICKS = 3
MAX_TICKS = 288
# dev doc §5.4 exactly: g = (g_health, g_power, g_transport, g_cost),
# "defaults 1.0, bounded [0.5, 2.0]" — M6b samples uniformly in this box
# each episode instead of fixing g at 1.0 (M6a's own scope note).
GOAL_LOW = 0.5
GOAL_HIGH = 2.0
GOAL_DIM = 4  # health, power, transport, cost — dev doc §5.4's own order
STABILIZATION_TICKS = 12
STABILIZATION_LEVEL = 0.9
N_AMBULANCES_PER_HOSPITAL = 2

AGENT_HEALTH = "health"
AGENT_POWER = "power"
AGENT_TRANSPORT = "transport"
POSSIBLE_AGENTS = (AGENT_HEALTH, AGENT_POWER, AGENT_TRANSPORT)

# power agent's repair lever — substations + water (dev doc §5.3's own
# "none, S1, S2, S3" wording), deliberately NOT hospitals — see module
# docstring's "Action split" note.
REPAIRABLE_POWER_TYPES = (AssetType.SUBSTATION, AssetType.WATER)
TRANSFER_TIERS = (0, 2, 5, 10)  # dev doc §5.3 exactly
MAX_HOSPITAL_PAIRS_FOR_TRANSFER = 3  # dev doc §5.3's own cap


class UDTMultiAgentEnv(
    ParallelEnv[str, npt.NDArray[np.float32], npt.NDArray[np.integer[Any]]]  # type: ignore[misc]
):
    """dev doc §5.1's 3-agent split. Construct once per process (loads
    the road network once, same reasoning as `single_env.py`); call
    `reset()` per episode."""

    metadata: dict[str, Any] = {"name": ENV_VERSION, "render_modes": []}

    def __init__(
        self,
        *,
        processed_dir: str | Path,
        n_ticks: int = MAX_TICKS,
        base_seed: int = 0,
        constrained_reward: bool = False,
    ) -> None:
        # M7 slice 2: dev doc §5.4's own footnote — the fixed
        # `-20.0 x safety_violations` reward term is for "UNCONSTRAINED
        # variants only"; a constrained variant (action masking active,
        # `MAPPOConfig.use_action_masking`) "never executes violations
        # (they're masked/repaired), so there the count of *attempted*
        # violations is logged as a metric instead" — `_tick_reward`
        # drops that term when this is `True`. Defaults `False` so every
        # M4-M6b/M7-slice-1 caller's reward is unchanged.
        self.constrained_reward = constrained_reward
        processed_dir = Path(processed_dir)
        with (processed_dir / "dependency_graph.json").open() as f:
            self._base_graph = DependencyGraph.model_validate(json.load(f))

        self._susceptibility_path = processed_dir / "flood_susceptibility.tif"
        with (processed_dir / "ward_boundary.geojson").open() as f:
            ward_boundary = json.load(f)
        self._ward_boundary = ward_boundary
        self._ward_polygon = shape(ward_boundary["features"][0]["geometry"])

        # One raster handle for the env's whole lifetime — same reasoning
        # as `single_env.py`'s `__init__`.
        self._raster = SusceptibilityRaster(self._susceptibility_path)
        self._road_network = RoadNetwork.load(processed_dir / "roads_full.graphml", self._raster)

        self.n_ticks = n_ticks
        self.base_seed = base_seed
        self._episode_count = 0
        self._prev_patient_deaths = 0

        self._hospital_ids = [
            a.asset_id for a in self._base_graph.assets if a.asset_type == AssetType.HOSPITAL
        ]
        self._substation_ids = [
            a.asset_id for a in self._base_graph.assets if a.asset_type == AssetType.SUBSTATION
        ]
        self._repairable_power_ids = [
            a.asset_id for a in self._base_graph.assets if a.asset_type in REPAIRABLE_POWER_TYPES
        ]
        self._critical_ids = [
            a.asset_id
            for a in self._base_graph.assets
            if a.asset_type in (AssetType.HOSPITAL, AssetType.SUBSTATION, AssetType.WATER)
        ]
        self._n_ambulances = len(self._hospital_ids) * N_AMBULANCES_PER_HOSPITAL
        self._transfer_pairs = self._build_transfer_pairs(self._hospital_ids)

        self.possible_agents: list[str] = list(POSSIBLE_AGENTS)
        self.agents: list[str] = []

        self.sim: Simulator | None = None
        self.incident: Incident | None = None
        self._stable_streak = 0
        self._prev_cascading_count = 0
        self._safety_violations_attempted = 0
        self.episode_trace: list[TwinState] = []
        # M6b: overwritten with a fresh uniform-[0.5,2.0] sample every
        # `reset()` — this default (all 1.0, dev doc §5.4's own default)
        # only matters before the first `reset()` call.
        self.goal: npt.NDArray[np.float32] = np.ones(GOAL_DIM, dtype=np.float32)

    # ------------------------------------------------------------------
    # Space construction (PettingZoo convention: methods, not attributes
    # — spaces can differ per agent name)
    # ------------------------------------------------------------------
    @staticmethod
    def _build_transfer_pairs(hospital_ids: list[str]) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for i, from_id in enumerate(hospital_ids):
            for to_id in hospital_ids[i + 1 :]:
                pairs.append((from_id, to_id))
                pairs.append((to_id, from_id))
        return pairs[: MAX_HOSPITAL_PAIRS_FOR_TRANSFER * 2]

    def _common_obs_dim(self) -> int:
        # levels, severity+onset_age, sin/cos, goal vector g (M6b)
        return 2 * len(self._critical_ids) + 2 + 2 + GOAL_DIM

    @functools.cache  # noqa: B019 - PettingZoo's own required pattern; env is a process-lifetime singleton
    def observation_space(self, agent: str) -> spaces.Space[Any]:
        """PettingZoo's own convention (its `parallel_api_test` asserts
        on it): must return the *same* Space object on repeated calls
        for the same agent, not a fresh one — hence the cache, not just
        a plain computed return."""
        common = self._common_obs_dim()
        if agent == AGENT_HEALTH:
            dim = common + 4 * len(self._hospital_ids)
        elif agent == AGENT_POWER:
            dim = common + 2 * len(self._substation_ids)
        elif agent == AGENT_TRANSPORT:
            dim = common + 3 * self._n_ambulances + 1
        else:
            raise ValueError(f"unknown agent {agent!r}")
        return spaces.Box(low=-1.0, high=2.0, shape=(dim,), dtype=np.float32)

    @functools.cache  # noqa: B019 - PettingZoo's own required pattern; env is a process-lifetime singleton
    def action_space(self, agent: str) -> spaces.Space[Any]:
        """Same identity-caching requirement as `observation_space`."""
        if agent == AGENT_HEALTH:
            n_transfer_options = 1 + len(self._transfer_pairs) * (len(TRANSFER_TIERS) - 1)
            return spaces.Discrete(n_transfer_options)
        if agent == AGENT_POWER:
            n_repair_options = 1 + len(self._repairable_power_ids)
            return spaces.MultiDiscrete([4] * len(self._substation_ids) + [n_repair_options])
        if agent == AGENT_TRANSPORT:
            return spaces.MultiDiscrete([2] * self._n_ambulances)
        raise ValueError(f"unknown agent {agent!r}")

    # ------------------------------------------------------------------
    # PettingZoo API
    # ------------------------------------------------------------------
    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, npt.NDArray[np.float32]], dict[str, dict[str, Any]]]:
        episode_seed = seed if seed is not None else self.base_seed + self._episode_count
        self._episode_count += 1

        scenario = generate_flood_scenario(
            scenario_id=f"multi_env_ep{self._episode_count}",
            ward_boundary_geojson=self._ward_boundary,
            seed=episode_seed,
        )
        self.incident = scenario.incident
        self._degradation_fn = make_flood_degradation_fn(scenario.incident, self._raster)

        # Dev doc §4.3 (2026-09-26): scenario's bed occupancy + onset hour.
        graph_copy = apply_initial_conditions(self._base_graph, scenario)
        self.sim = Simulator(
            graph_copy,
            seed=episode_seed,
            road_network=self._road_network,
            onset_hour_of_day=onset_hour_of_day(scenario),
        )
        spawn_ambulances(self.sim.graph, self._road_network, N_AMBULANCES_PER_HOSPITAL)

        # M6b: sampled from the twin's own rng, right after construction —
        # deterministic given episode_seed (consumes GOAL_DIM draws before
        # any tick's own randomness), same disclosed convention as every
        # other rng use in this codebase (dev doc §3.6).
        self.goal = self.sim.rng.uniform(GOAL_LOW, GOAL_HIGH, size=GOAL_DIM).astype(np.float32)

        self.agents = list(self.possible_agents)
        self._stable_streak = 0
        self._prev_cascading_count = 0
        self._prev_patient_deaths = 0
        # M7: same cumulative-counter convention as `_prev_cascading_
        # count` — read off `trace[-1]` by `logging/metrics.py`.
        self._safety_violations_attempted = 0
        self.episode_trace = []

        obs = {a: self._build_observation(a) for a in self.agents}
        infos: dict[str, dict[str, Any]] = {a: {} for a in self.agents}
        return obs, infos

    def close(self) -> None:
        self._raster.close()

    def step(
        self, actions: dict[str, npt.NDArray[np.integer[Any]]]
    ) -> tuple[
        dict[str, npt.NDArray[np.float32]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        assert self.sim is not None and self.incident is not None, "call reset() first"
        patient_transfer = self._decode_health_action(actions[AGENT_HEALTH])
        shed_tier, repair_target = self._decode_power_action(actions[AGENT_POWER])
        ambulance_assignment = self._decode_transport_action(actions[AGENT_TRANSPORT])

        # M7 slice 1: post-hoc validation of the *joint* action across
        # all 3 agents (dev doc §8 usage (2)) — always runs, regardless
        # of whether the caller is also using slice 2's pre-hoc masking
        # (`action_mask()`, below): masking makes a violation *unlikely*
        # for the two rules it covers, this check is what guarantees
        # nothing genuinely unsafe ever executes.
        joint_action = AgentAction(
            repair_target=repair_target,
            ambulance_assignment=ambulance_assignment,
            patient_transfer=patient_transfer,
            shed_tier=shed_tier,
        )
        report = check(self.sim.graph, joint_action, road_network=self._road_network)
        repair_target = report.repaired_action.repair_target
        ambulance_assignment = report.repaired_action.ambulance_assignment
        patient_transfer = report.repaired_action.patient_transfer
        shed_tier = report.repaired_action.shed_tier
        violations_this_decision = len(report.violations)
        self._safety_violations_attempted += violations_this_decision

        total_reward = 0.0
        for tick_in_interval in range(DECISION_INTERVAL_TICKS):
            update_substation_load(
                self.sim.graph, self.sim.tick, self.incident, self.sim.dt_minutes
            )
            self._road_network.update_for_tick(self.incident, self.sim.tick, self.sim.dt_minutes)
            generate_requests(
                self.sim.graph,
                self.sim.tick,
                self.sim.dt_hours,
                self.incident,
                self._ward_polygon,
                self.sim.rng,
            )
            snapshot = self.sim.step(
                degradation_fn=self._degradation_fn,
                repair_target=repair_target if tick_in_interval == 0 else None,
                ambulance_assignment=ambulance_assignment if tick_in_interval == 0 else None,
                patient_transfer=patient_transfer if tick_in_interval == 0 else None,
                shed_tier=shed_tier if tick_in_interval == 0 else None,
            )
            snapshot.safety_violations_attempted_cumulative = self._safety_violations_attempted
            # `constrained_reward=True` (M7 slice 2): drop the fixed
            # -20/violation term from the reward — see `__init__`'s
            # docstring note. The `episode_trace`/H3 metric still counts
            # every attempted violation regardless (`_safety_violations_
            # attempted` above), only the *reward* signal differs.
            reward_violations = 0 if self.constrained_reward else violations_this_decision
            total_reward += self._tick_reward(
                snapshot, reward_violations if tick_in_interval == 0 else 0
            )
            self.episode_trace.append(snapshot)
            if self.sim.tick >= self.n_ticks:
                break

        terminated = self._stable_streak >= STABILIZATION_TICKS
        truncated = self.sim.tick >= self.n_ticks and not terminated

        obs = {a: self._build_observation(a) for a in self.agents}
        rewards = {a: total_reward for a in self.agents}  # dev doc §5.4: shared team reward
        terminations = {a: terminated for a in self.agents}
        truncations = {a: truncated for a in self.agents}
        # `safety_violations_attempted` (M7 slice 2): this *decision's*
        # count (not the cumulative), for `agents/marl/mappo.py`'s
        # Lagrangian-penalty cost signal to read per step without
        # needing to diff two cumulative snapshots itself.
        infos: dict[str, dict[str, Any]] = {
            a: {"tick": self.sim.tick, "safety_violations_attempted": violations_this_decision}
            for a in self.agents
        }

        if terminated or truncated:
            self.agents = []  # PettingZoo convention: no active agents once the episode ends

        return obs, rewards, terminations, truncations, infos

    # ------------------------------------------------------------------
    # Action decoding (per agent) — same encodings `single_env.py` uses
    # for the corresponding sub-actions, so a rollout is directly
    # comparable at the mechanism level.
    # ------------------------------------------------------------------
    def _decode_health_action(
        self, action: npt.NDArray[np.integer[Any]] | int
    ) -> tuple[str, str, int] | None:
        # `Discrete`'s own `.sample()` gives a bare scalar, but a caller
        # driving this env from a uniform per-agent policy (e.g.
        # `agents/marl/mappo.py`'s `MultiCategoricalActor`, which always
        # emits a length-1 array for a single-head action space) may
        # legitimately hand in a length-1 array instead — accept both
        # rather than `int()`-ing a non-scalar array (deprecated in
        # NumPy, and an outright error in future versions).
        choice = int(np.asarray(action).reshape(-1)[0])
        if choice == 0 or not self._transfer_pairs:
            return None
        pair_idx, tier_idx = divmod(choice - 1, len(TRANSFER_TIERS) - 1)
        if pair_idx >= len(self._transfer_pairs):
            return None
        from_id, to_id = self._transfer_pairs[pair_idx]
        return (from_id, to_id, TRANSFER_TIERS[tier_idx + 1])

    def _decode_power_action(
        self, action: npt.NDArray[np.integer[Any]]
    ) -> tuple[dict[str, int] | None, str | None]:
        shed_choices = [int(x) for x in action[: len(self._substation_ids)]]
        repair_choice = int(action[len(self._substation_ids)])
        shed_tier = (
            dict(zip(self._substation_ids, shed_choices, strict=True))
            if self._substation_ids
            else None
        )
        repair_target = self._repairable_power_ids[repair_choice - 1] if repair_choice > 0 else None
        return shed_tier, repair_target

    def _decode_transport_action(
        self, action: npt.NDArray[np.integer[Any]]
    ) -> dict[str, str] | None:
        assert self.sim is not None
        pending: list[dict[str, Any]] = self.sim.graph.graph.get("pending_requests", [])
        if not pending:
            return None
        oldest_request = min(pending, key=lambda r: r["requested_at_tick"])
        ambulance_ids = [
            asset_id
            for asset_id in self.sim.graph.nodes
            if self.sim.graph.nodes[asset_id]["asset"].asset_type == AssetType.AMBULANCE
        ]
        for ambulance_id, choice in zip(ambulance_ids, (int(x) for x in action), strict=False):
            asset = self.sim.graph.nodes[ambulance_id]["asset"]
            if choice == 1 and asset.attributes.get("status") == "idle":
                return {ambulance_id: str(oldest_request["request_id"])}
        return None

    # ------------------------------------------------------------------
    # Pre-hoc action masking (M7 slice 2, dev doc §5.5's Experiment G:
    # "MAPPO + action masking (mask infeasible actions from the
    # constraint engine pre-hoc)"). Callers (`agents/marl/mappo.py`'s
    # `MAPPOTrainer.collect_rollout`, when `MAPPOConfig.
    # use_action_masking` is set) call this once per agent right before
    # sampling that agent's next action — the mask reflects whatever
    # state `self.sim.graph` is in *right now*, which is exactly the
    # state the observation just returned by `reset()`/`step()` also
    # reflects (nothing mutates `self.sim.graph` in between in this
    # env's own call sequence).
    # ------------------------------------------------------------------
    def action_mask(self, agent: str) -> list[npt.NDArray[np.bool_]]:
        """One boolean array per action head (matching `action_space
        (agent)`'s shape), `True` = this discrete choice passes every
        constraint rule that touches this agent's action field.

        Only `health` (`bed_capacity`) and `transport` (`route_flood_
        safety`) have any rule to mask against — every one of that
        head's possible discrete values is decoded and run through
        `constraints/engine.py`'s `check(...)`, so this is exhaustive
        for those two rules, not a heuristic approximation. `power`'s
        action space isn't touched by any *implemented* rule (`power_
        balance` is a disclosed no-op, see `engine.py`), so its mask is
        always all-`True`. The "do nothing" choice (transfer-choice 0,
        dispatch-choice 0 per ambulance) is never masked — every head
        always has at least one feasible choice, which is also what
        lets `networks.py`'s masked softmax stay well-defined.

        An ambulance whose dispatch would be a structural no-op (not
        currently idle, or no pending request exists at all) is left
        feasible (`True`) rather than masked — that's not a constraint
        violation, just an action with no effect, and masking it would
        conflate "unsafe" with "pointless" (§8's rules are only about
        the former)."""
        assert self.sim is not None

        if agent == AGENT_HEALTH:
            health_space = self.action_space(AGENT_HEALTH)
            assert isinstance(health_space, spaces.Discrete)
            n = int(health_space.n)
            mask = np.ones(n, dtype=bool)
            for choice in range(1, n):
                transfer_candidate = self._decode_health_action(np.array([choice]))
                report = check(self.sim.graph, AgentAction(patient_transfer=transfer_candidate))
                mask[choice] = report.passed
            return [mask]

        if agent == AGENT_TRANSPORT:
            masks = [np.ones(2, dtype=bool) for _ in range(self._n_ambulances)]
            for i in range(self._n_ambulances):
                probe = np.zeros(self._n_ambulances, dtype=np.int64)
                probe[i] = 1
                dispatch_candidate = self._decode_transport_action(probe)
                if dispatch_candidate is None:
                    continue  # structural no-op (busy ambulance / no request) - feasible
                report = check(
                    self.sim.graph,
                    AgentAction(ambulance_assignment=dispatch_candidate),
                    road_network=self._road_network,
                )
                masks[i][1] = report.passed
            return masks

        if agent == AGENT_POWER:
            power_space = self.action_space(AGENT_POWER)
            assert isinstance(power_space, spaces.MultiDiscrete)
            nvec = [int(n) for n in power_space.nvec]
            return [np.ones(n, dtype=bool) for n in nvec]

        raise ValueError(f"unknown agent {agent!r}")

    # ------------------------------------------------------------------
    # Reward — dev doc §5.4, identical formula/scope to `single_env.py`
    # (see that module's docstring for the disclosed defaults).
    # ------------------------------------------------------------------
    def _tick_reward(self, snapshot: TwinState, violations_attempted: int = 0) -> float:
        assert self.sim is not None
        queued_this_tick = sum(
            int(a.attributes.get("patient_queue", 0))
            for a in snapshot.assets
            if a.asset_type == AssetType.HOSPITAL
        )
        unmet_patient_hours = queued_this_tick * self.sim.dt_hours
        patient_deaths_delta = snapshot.patient_deaths_cumulative - self._prev_patient_deaths
        self._prev_patient_deaths = snapshot.patient_deaths_cumulative

        new_cascade_failures = max(0, snapshot.cascading_failure_count - self._prev_cascading_count)
        self._prev_cascading_count = snapshot.cascading_failure_count

        unserved_energy_mwh = 0.0
        for asset in snapshot.assets:
            if asset.asset_type == AssetType.SUBSTATION:
                shed_fraction = SHED_FRACTION_BY_TIER[int(asset.attributes.get("shed_tier", 0))]
                unserved_energy_mwh += (
                    float(asset.attributes.get("load_mw", 0.0)) * shed_fraction * self.sim.dt_hours
                )

        ambulance_response_delay_hours = sum(snapshot.ambulance_response_times_this_tick)

        all_healthy = all(a.functional_level >= STABILIZATION_LEVEL for a in snapshot.assets)
        self._stable_streak = self._stable_streak + 1 if all_healthy else 0

        # dev doc §5.4's exact formula: only the four g_*-named terms are
        # goal-weighted (M6b); new_cascade_failures/safety_violations/
        # water_shortage_hours keep their fixed coefficients regardless
        # of g. g_cost (self.goal[3]) has nothing to multiply yet —
        # resource_cost is still a disclosed 0.0 — so it's sampled/
        # observed but currently inert, not omitted.
        g_health, g_power, g_transport, _g_cost = self.goal
        return -(
            float(g_health) * (unmet_patient_hours + 10.0 * patient_deaths_delta)
            + float(g_power) * unserved_energy_mwh
            + float(g_transport) * ambulance_response_delay_hours
            + 5.0 * new_cascade_failures
            + 20.0 * violations_attempted  # dev doc §5.4's exact coefficient (M7)
        )

    # ------------------------------------------------------------------
    # Observation — common block (all agents) + per-agent additions,
    # dev doc §5.2's structure, reduced scope (see module docstring).
    # ------------------------------------------------------------------
    def _common_obs(self) -> list[float]:
        assert self.sim is not None and self.incident is not None
        g = self.sim.graph
        parts: list[float] = []
        for asset_id in self._critical_ids:
            asset = g.nodes[asset_id]["asset"]
            parts.extend([asset.functional_level, asset.intrinsic_level])
        parts.append(self.incident.severity)
        onset_age_hours = (self.sim.tick - self.incident.onset_tick) * self.sim.dt_hours
        parts.append(min(1.0, onset_age_hours / 24.0))
        hour_of_day = (self.sim.tick * self.sim.dt_hours) % 24.0
        parts.append(float(np.sin(2 * np.pi * hour_of_day / 24.0)))
        parts.append(float(np.cos(2 * np.pi * hour_of_day / 24.0)))
        parts.extend(float(x) for x in self.goal)  # M6b: dev doc §5.2's "goal vector g"
        return parts

    def _build_observation(self, agent: str) -> npt.NDArray[np.float32]:
        assert self.sim is not None
        g = self.sim.graph
        parts = self._common_obs()

        if agent == AGENT_HEALTH:
            for hospital_id in self._hospital_ids:
                attrs = g.nodes[hospital_id]["asset"].attributes
                beds_total = max(1, int(attrs.get("beds_total", 1)))
                icu_total = max(1, int(attrs.get("icu_total", 1)))
                power_edges = [e for e in dependency_edges_of(g, hospital_id) if e.kind == "power"]
                if power_edges:
                    state = self.sim.edge_states.get(power_edges[0].edge_id)
                    buffer_frac = (
                        (state.remaining_hours / state.capacity_hours)
                        if state and state.capacity_hours > 0
                        else 1.0
                    )
                else:
                    buffer_frac = 1.0
                parts.extend(
                    [
                        int(attrs.get("beds_occupied", 0)) / beds_total,
                        int(attrs.get("icu_occupied", 0)) / icu_total,
                        min(1.0, int(attrs.get("patient_queue", 0)) / 20.0),
                        buffer_frac,
                    ]
                )
        elif agent == AGENT_POWER:
            for substation_id in self._substation_ids:
                attrs = g.nodes[substation_id]["asset"].attributes
                capacity = max(1e-6, float(attrs.get("capacity_mw", 1.0)))
                parts.append(min(2.0, float(attrs.get("load_mw", 0.0)) / capacity))
                parts.append(int(attrs.get("shed_tier", 0)) / 3.0)
        elif agent == AGENT_TRANSPORT:
            ambulance_ids = [
                asset_id
                for asset_id in g.nodes
                if g.nodes[asset_id]["asset"].asset_type == AssetType.AMBULANCE
            ]
            status_index = {"idle": 0, "enroute": 1, "returning": 2}
            for ambulance_id in ambulance_ids[: self._n_ambulances]:
                one_hot = [0.0, 0.0, 0.0]
                status = g.nodes[ambulance_id]["asset"].attributes.get("status", "idle")
                one_hot[status_index.get(status, 0)] = 1.0
                parts.extend(one_hot)
            parts.append(min(1.0, len(g.graph.get("pending_requests", [])) / 20.0))

        return np.array(parts, dtype=np.float32)
