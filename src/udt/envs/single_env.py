"""Single-agent Gymnasium environment (dev doc §5.1-§5.4, module M5 —
"Experiment B and the MARL sanity fallback").

Wraps the twin (M1-M4: `Simulator`, `RoadNetwork`, `twin/ambulances.py`,
`twin/demand.py`, `twin/power.py`) as one flattened-action RL agent
standing in for the dev doc's three cooperating roles (health/power/
transport) that `multi_env.py` (M6a, not built yet) will eventually
split apart.

**Disclosed prototype-1 scope — read before training anything:**

- **Decision interval = 3 ticks (15 min), dev doc §5.1 exactly.** One
  `step()` call decodes the action *once*, applies it on the interval's
  first tick, then lets the twin evolve for the remaining
  `DECISION_INTERVAL_TICKS - 1` ticks with no new decision (repair/shed/
  transfer/dispatch commands aren't re-issued — re-issuing the *same*
  transfer/repair command every tick would double- or triple-apply
  something meant to happen once per interval). This is a disclosed
  reading of "world evolves every tick, agents act every 3rd" — the dev
  doc doesn't spell out what happens to a not-re-issued action, and
  "nothing new is commanded, physics continues" is the natural one.
- **Observation space is a real but reduced subset of dev doc §5.2.**
  Included: per-critical-asset (hospital/substation) functional_level +
  intrinsic_level, per-hospital power-edge buffer fraction, per-hospital
  bed/ICU/queue occupancy, per-substation load/capacity ratio + shed
  tier, per-ambulance status one-hot, pending-request count, incident
  severity + onset-age, tick-of-day sin/cos. **Deferred:** the goal
  vector `g` (dev doc §7.3 — no LLM planner exists yet to produce one,
  M9a/M9b), per-individual-road blockage (156 real roads is far more
  than the "60-90 dims" the dev doc's own sizing note anticipated; each
  road's effect is already folded into the hospital/substation
  functional_level it feeds, which *is* observed).
- **Action space is built dynamically from whatever graph is loaded**
  (2 hospitals / 1 substation / 4 ambulances for the real Kurla data),
  not from a static `configs/env.yaml` enumeration — the dev doc expects
  the latter, but a static file would just restate this env's own
  dynamic derivation for this prototype's one fixed graph. Test
  coverage asserts the expected shape for the real graph instead (dev
  doc §13's "obs/action shapes match configs/env.yaml" intent, same
  purpose, different mechanism).
- **`resource_cost` is not modeled** (no real repair-crew/ambulance-fuel
  cost data exists) — always 0.0 in the reward, disclosed rather than
  invented. `water_shortage_hours` is also always 0.0 — **not** because
  there are no water facilities anymore (M2 revision closed that gap,
  see `data/manual_facilities.yaml`), but because isolating the
  water-specific term from a hospital's combined
  `functional_level = intrinsic × Π sat(d)` product needs per-edge
  `EdgeRuntimeState`/satisfaction in the tick trace, which `TwinState`
  doesn't carry yet (only asset-level functional/intrinsic levels) — a
  real, still-open, disclosed gap of its own, not silently reused as the
  old one. Goal weights `g_*` (§5.4) default to 1.0
  (the dev doc's own stated default) since goal-conditioning doesn't
  exist yet either.
- **`safety_violations` (M7 addition, dev doc §8/§5.4):** every decoded
  action is checked against `constraints/engine.py`'s registered rules
  *before* being applied — whatever the checker repairs (clips/drops) is
  what actually executes, so the twin never sees a genuinely unsafe
  action. The count of violations caught this way is penalized in the
  reward at the dev doc's exact `-20.0` coefficient. This env has no
  pre-hoc action masking yet (M7 slice 2, not built) — dev doc §5.4's own
  footnote calls this the "UNCONSTRAINED variant" case: violations are
  attempted (decoded, then caught and repaired) rather than never
  proposed in the first place, which is exactly why they're still
  penalized here instead of just logged.
- **No frozen scenario suite exists yet** (§4.3, M3's own deferral) —
  `reset()` generates a fresh random flood scenario each episode
  (severity resampled) rather than drawing from train/val/test splits,
  so a policy can't just memorize one scenario.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces
from shapely.geometry import shape

from udt.common.models import (
    AgentAction,
    AssetType,
    DependencyGraph,
    Incident,
    TwinState,
)
from udt.constraints.engine import check
from udt.envs.reward import load_reward_normalisers, tick_terms
from udt.incidents.degradations.flood import SusceptibilityRaster, make_flood_degradation_fn
from udt.scenarios.generator import (
    apply_initial_conditions,
    generate_flood_scenario,
    onset_hour_of_day,
)
from udt.scenarios.suite import DEFAULT_FLOOD_SUITE_DIR, load_suite
from udt.twin.ambulances import generate_requests, spawn_ambulances
from udt.twin.graph import dependency_edges_of
from udt.twin.modes import require_twin_v2
from udt.twin.power import update_substation_load
from udt.twin.road_network import RoadNetwork
from udt.twin.simulator import Simulator

DECISION_INTERVAL_TICKS = 3  # dev doc §5.1 exactly (15 simulated minutes)
MAX_TICKS = 288  # dev doc §3.1: T_max = 24h at dt=5min
STABILIZATION_TICKS = 12  # dev doc §3.1: 12 consecutive ticks >= 90% ends the episode
STABILIZATION_LEVEL = 0.9
N_AMBULANCES_PER_HOSPITAL = 2  # matches experiments/runner.py's disclosed placeholder

# Same set `agents/rule_based.py` uses for the repair rule — duplicated
# rather than imported, to keep envs/ from depending on agents/ (twin/ is
# the only package both depend on; the dependency direction stays
# one-way: twin <- agents, twin <- envs).
REPAIRABLE_TYPES = (AssetType.HOSPITAL, AssetType.SUBSTATION, AssetType.WATER)
TRANSFER_TIERS = (0, 2, 5, 10)  # dev doc §5.3 exactly (0 folded in as "no transfer")
MAX_HOSPITAL_PAIRS_FOR_TRANSFER = 3  # dev doc §5.3's own cap ("<=3 pairs")


class UDTSingleAgentEnv(gym.Env[npt.NDArray[np.float32], npt.NDArray[np.integer[Any]]]):
    """dev doc §5.1's single super-agent env. Construct once per process
    (loads the road network once — expensive, and its per-edge
    susceptibility cache doesn't depend on the episode); call `reset()`
    per episode."""

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        *,
        processed_dir: str | Path,
        n_ticks: int = MAX_TICKS,
        base_seed: int = 0,
        scenario_split: str | None = "train",
        suite_dir: str | Path | None = None,
    ) -> None:
        require_twin_v2()  # twin-v2 physics only; see udt.twin.modes
        processed_dir = Path(processed_dir)
        dep_graph_path = processed_dir / "dependency_graph.json"
        with dep_graph_path.open() as f:
            self._base_graph = DependencyGraph.model_validate(json.load(f))

        self._susceptibility_path = processed_dir / "flood_susceptibility.tif"
        with (processed_dir / "ward_boundary.geojson").open() as f:
            ward_boundary = json.load(f)
        self._ward_boundary = ward_boundary
        self._ward_polygon = shape(ward_boundary["features"][0]["geometry"])

        # One raster handle for the env's whole lifetime — reused by both
        # the road network (built once, below) and every episode's
        # degradation function (`reset()`), instead of opening a fresh
        # rasterio handle per episode and never closing it. Closed by
        # `close()` (Gymnasium's own convention).
        self._raster = SusceptibilityRaster(self._susceptibility_path)

        roads_full_path = processed_dir / "roads_full.graphml"
        self._road_network = RoadNetwork.load(roads_full_path, self._raster)

        self.n_ticks = n_ticks
        self.base_seed = base_seed
        self._episode_count = 0
        # Dev doc §4.3 (2026-09-27): episodes draw from the frozen suite's
        # `scenario_split` (train for training; val/test only for evaluation),
        # verified on load. `None` = fresh random scenarios, for smoke tests only.
        self._reward_normalisers = load_reward_normalisers()
        self.scenario_split = scenario_split
        self._scenarios = (
            load_suite(suite_dir or DEFAULT_FLOOD_SUITE_DIR, scenario_split)
            if scenario_split is not None
            else []
        )
        self._prev_patient_deaths = 0

        # Static counts, derived once from the loaded graph — fixed for
        # this prototype's one real dependency graph (dev doc §12.2:
        # "Asset"/"DependencyGraph" don't change shape mid-run).
        self._hospital_ids = [
            a.asset_id for a in self._base_graph.assets if a.asset_type == AssetType.HOSPITAL
        ]
        self._substation_ids = [
            a.asset_id for a in self._base_graph.assets if a.asset_type == AssetType.SUBSTATION
        ]
        self._critical_ids = [
            a.asset_id for a in self._base_graph.assets if a.asset_type in REPAIRABLE_TYPES
        ]
        self._n_ambulances = len(self._hospital_ids) * N_AMBULANCES_PER_HOSPITAL

        self._transfer_pairs = self._build_transfer_pairs(self._hospital_ids)

        self.action_space = spaces.MultiDiscrete(self._action_nvec())
        # Every feature `_build_observation` emits is normalized/clipped
        # into [-1, 2] by construction (sin/cos reach -1; the load/
        # capacity ratio is clipped at 2.0 - see there) - a real bounded
        # space, not a lazy unbounded one.
        self.observation_space = spaces.Box(
            low=-1.0, high=2.0, shape=(self._obs_dim(),), dtype=np.float32
        )

        self.sim: Simulator | None = None
        self.incident: Incident | None = None
        self._stable_streak = 0
        self._prev_cascading_count = 0

    # ------------------------------------------------------------------
    # Space construction
    # ------------------------------------------------------------------
    @staticmethod
    def _build_transfer_pairs(hospital_ids: list[str]) -> list[tuple[str, str]]:
        """Ordered (from, to) pairs, capped at dev doc §5.3's own "<=3
        pairs" note — for the real Kurla graph (2 hospitals) this is
        exactly 1 unordered pair -> 2 ordered directions."""
        pairs: list[tuple[str, str]] = []
        for i, from_id in enumerate(hospital_ids):
            for to_id in hospital_ids[i + 1 :]:
                pairs.append((from_id, to_id))
                pairs.append((to_id, from_id))
        return pairs[: MAX_HOSPITAL_PAIRS_FOR_TRANSFER * 2]

    def _action_nvec(self) -> list[int]:
        n_transfer_options = 1 + len(self._transfer_pairs) * (len(TRANSFER_TIERS) - 1)
        n_repair_options = 1 + len(self._critical_ids)
        return (
            [n_transfer_options]
            + [4] * len(self._substation_ids)  # shed tier 0-3 per substation
            + [n_repair_options]
            + [2] * self._n_ambulances  # dispatch-or-not per ambulance
        )

    def _obs_dim(self) -> int:
        n_critical = len(self._critical_ids)
        n_hospitals = len(self._hospital_ids)
        n_substations = len(self._substation_ids)
        # 4 per hospital: beds_occupied frac, icu_occupied frac, queue
        # (norm), water_reserve frac (real, M2 revision — was a hardcoded
        # 0.0 placeholder before the real graph had any water facilities).
        return (
            2 * n_critical  # functional_level, intrinsic_level
            + n_hospitals  # power-edge buffer fraction
            + 4 * n_hospitals
            + 2 * n_substations  # load/capacity ratio, shed_tier normalized
            + 3 * self._n_ambulances  # status one-hot (idle/enroute/returning)
            + 1  # pending_requests_count (normalized)
            + 2  # incident severity, onset-age normalized
            + 2  # tick-of-day sin/cos
        )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[npt.NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        episode_seed = seed if seed is not None else self.base_seed + self._episode_count
        self._episode_count += 1

        if self._scenarios:
            index = (options or {}).get("scenario_index", episode_seed % len(self._scenarios))
            scenario = self._scenarios[int(index)]
        else:
            scenario = generate_flood_scenario(
                scenario_id=f"single_env_ep{self._episode_count}",
                ward_boundary_geojson=self._ward_boundary,
                seed=episode_seed,
            )
        self.scenario = scenario
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

        self._stable_streak = 0
        self._prev_cascading_count = 0
        self._prev_patient_deaths = 0
        # M7: cumulative count of constraint violations caught (and
        # repaired) so far this episode — same "cumulative counter set
        # on the returned snapshot" convention `cascading_failure_count`
        # already uses, so `logging/metrics.py` can read it off
        # `trace[-1]` unchanged.
        self._safety_violations_attempted = 0
        # Full per-tick trace of this episode, for `logging/metrics.py`'s
        # `compute_episode_metrics` to consume after the episode ends —
        # `step()` only returns the *last* of each interval's ticks to
        # the RL caller (Gymnasium's own contract), so this is the only
        # place a caller can get the same metrics `experiments/runner.py`
        # reports for the rule-based/do-nothing baselines, for a fair
        # side-by-side comparison (`scripts/evaluate.py`).
        self.episode_trace: list[TwinState] = []

        return self._build_observation(), {}

    def close(self) -> None:
        self._raster.close()

    def step(
        self, action: npt.NDArray[np.integer[Any]]
    ) -> tuple[npt.NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        assert self.sim is not None and self.incident is not None, "call reset() first"
        agent_action = self._decode_action(action)

        # M7: post-hoc validation, dev doc §8 usage (2) — "validation of
        # every action before execution (all experiments, all paths)".
        # No pre-hoc masking exists yet (M7 slice 2), so the *repaired*
        # action is what actually executes; whatever got clipped/dropped
        # is counted and penalized below (dev doc §5.4's "UNCONSTRAINED
        # variant" case — see class docstring).
        report = check(self.sim.graph, agent_action, road_network=self._road_network)
        agent_action = report.repaired_action
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
                repair_target=agent_action.repair_target if tick_in_interval == 0 else None,
                ambulance_assignment=(
                    agent_action.ambulance_assignment if tick_in_interval == 0 else None
                ),
                ambulance_destination=(
                    agent_action.ambulance_destination if tick_in_interval == 0 else None
                ),
                patient_transfer=agent_action.patient_transfer if tick_in_interval == 0 else None,
                shed_tier=agent_action.shed_tier if tick_in_interval == 0 else None,
                divert=agent_action.divert if tick_in_interval == 0 else None,
                surge=agent_action.surge if tick_in_interval == 0 else None,
                transfer_requests=(
                    agent_action.transfer_requests if tick_in_interval == 0 else None
                ),
            )
            snapshot.safety_violations_attempted_cumulative = self._safety_violations_attempted
            total_reward += self._tick_reward(
                snapshot, violations_this_decision if tick_in_interval == 0 else 0
            )
            self.episode_trace.append(snapshot)
            if self.sim.tick >= self.n_ticks:
                break

        terminated = self._stable_streak >= STABILIZATION_TICKS
        truncated = self.sim.tick >= self.n_ticks and not terminated
        info: dict[str, Any] = {"tick": self.sim.tick}
        return self._build_observation(), total_reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Action decoding
    # ------------------------------------------------------------------
    def _decode_action(self, action: npt.NDArray[np.integer[Any]]) -> AgentAction:
        assert self.sim is not None
        idx = 0
        transfer_choice = int(action[idx])
        idx += 1
        shed_choices = [int(x) for x in action[idx : idx + len(self._substation_ids)]]
        idx += len(self._substation_ids)
        repair_choice = int(action[idx])
        idx += 1
        ambulance_choices = [int(x) for x in action[idx : idx + self._n_ambulances]]

        patient_transfer = None
        if transfer_choice > 0 and self._transfer_pairs:
            pair_idx, tier_idx = divmod(transfer_choice - 1, len(TRANSFER_TIERS) - 1)
            if pair_idx < len(self._transfer_pairs):
                from_id, to_id = self._transfer_pairs[pair_idx]
                patient_transfer = (from_id, to_id, TRANSFER_TIERS[tier_idx + 1])

        shed_tier = (
            {sid: tier for sid, tier in zip(self._substation_ids, shed_choices, strict=True)}
            if self._substation_ids
            else None
        )

        repair_target = self._critical_ids[repair_choice - 1] if repair_choice > 0 else None

        ambulance_assignment = None
        pending = self.sim.graph.graph.get("pending_requests", [])
        if pending:
            oldest_request = min(pending, key=lambda r: r["requested_at_tick"])
            ambulance_ids = [
                asset_id
                for asset_id in self.sim.graph.nodes
                if self.sim.graph.nodes[asset_id]["asset"].asset_type == AssetType.AMBULANCE
            ]
            for ambulance_id, choice in zip(ambulance_ids, ambulance_choices, strict=False):
                asset = self.sim.graph.nodes[ambulance_id]["asset"]
                if choice == 1 and asset.attributes.get("status") == "idle":
                    ambulance_assignment = {ambulance_id: str(oldest_request["request_id"])}
                    break  # one dispatch per tick, same convention as the rule-based agent

        return AgentAction(
            repair_target=repair_target,
            ambulance_assignment=ambulance_assignment,
            patient_transfer=patient_transfer,
            shed_tier=shed_tier,
        )

    # ------------------------------------------------------------------
    # Reward (dev doc §5.4, g_* defaulted to 1.0 - see class docstring)
    # ------------------------------------------------------------------
    def _tick_reward(self, snapshot: TwinState, violations_attempted: int = 0) -> float:
        assert self.sim is not None
        # Dev doc §5.4 (normalisers implemented 2026-09-27, `envs/reward.py`):
        # each raw term is divided by its fitted normaliser before the
        # §5.4 coefficients apply.
        raw = tick_terms(
            snapshot, self.sim.dt_hours, self._prev_patient_deaths, self._prev_cascading_count
        )
        self._prev_patient_deaths = snapshot.patient_deaths_cumulative
        self._prev_cascading_count = snapshot.cascading_failure_count
        t = {k: v / self._reward_normalisers[k] for k, v in raw.items()}

        all_healthy = all(a.functional_level >= STABILIZATION_LEVEL for a in snapshot.assets)
        self._stable_streak = self._stable_streak + 1 if all_healthy else 0

        return -(
            1.0 * (t["unmet_patient_hours"] + 10.0 * t["patient_deaths"])
            + 1.0 * t["unserved_energy_mwh"]
            + 1.0 * t["ambulance_response_delay_hours"]
            + 5.0 * t["new_cascade_failures"]
            + 20.0 * violations_attempted  # dev doc §5.4's exact coefficient (M7)
        )

    # ------------------------------------------------------------------
    # Observation (dev doc §5.2, reduced scope - see class docstring)
    # ------------------------------------------------------------------
    def _edge_buffer_fraction(self, consumer_id: str, kind: str) -> float:
        """Remaining-buffer fraction (0..1) of `consumer_id`'s first `kind`
        dependency edge — 1.0 if it has none (never buffered against that
        kind, e.g. a hospital with no water edge) or its buffer is
        unbounded/exhausted-tracking-not-yet-started. Shared by the power
        and (M2 revision — see module docstring) water reserve
        observation slots below; same formula either way."""
        assert self.sim is not None
        edges = [e for e in dependency_edges_of(self.sim.graph, consumer_id) if e.kind == kind]
        if not edges:
            return 1.0
        state = self.sim.edge_states.get(edges[0].edge_id)
        if state and state.capacity_hours > 0:
            return state.remaining_hours / state.capacity_hours
        return 1.0

    def _build_observation(self) -> npt.NDArray[np.float32]:
        assert self.sim is not None and self.incident is not None
        g = self.sim.graph
        parts: list[float] = []

        for asset_id in self._critical_ids:
            asset = g.nodes[asset_id]["asset"]
            parts.extend([asset.functional_level, asset.intrinsic_level])

        for hospital_id in self._hospital_ids:
            parts.append(self._edge_buffer_fraction(hospital_id, "power"))

        for hospital_id in self._hospital_ids:
            attrs = g.nodes[hospital_id]["asset"].attributes
            beds_total = max(1, int(attrs.get("beds_total", 1)))
            icu_total = max(1, int(attrs.get("icu_total", 1)))
            parts.extend(
                [
                    int(attrs.get("beds_occupied", 0)) / beds_total,
                    int(attrs.get("icu_occupied", 0)) / icu_total,
                    min(1.0, int(attrs.get("patient_queue", 0)) / 20.0),  # disclosed norm cap
                    # Real water-reserve buffer fraction (M2 revision closed
                    # the "0 water facilities in the real data" gap this
                    # slot used to be a hardcoded 0.0 placeholder for —
                    # dev doc §5.2's own "water/gen reserves" wording).
                    self._edge_buffer_fraction(hospital_id, "water"),
                ]
            )

        for substation_id in self._substation_ids:
            attrs = g.nodes[substation_id]["asset"].attributes
            capacity = max(1e-6, float(attrs.get("capacity_mw", 1.0)))
            # Clipped, not just normalized: `twin/power.py`'s load surge
            # can genuinely push raw load past capacity (that's what
            # "overloaded" means) — clip at 2x rather than leaving the
            # observation space unbounded for one feature.
            parts.append(min(2.0, float(attrs.get("load_mw", 0.0)) / capacity))
            parts.append(int(attrs.get("shed_tier", 0)) / 3.0)

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

        parts.append(
            min(1.0, len(g.graph.get("pending_requests", [])) / 20.0)
        )  # disclosed norm cap
        parts.append(self.incident.severity)
        onset_age_hours = (self.sim.tick - self.incident.onset_tick) * self.sim.dt_hours
        parts.append(min(1.0, onset_age_hours / 24.0))

        hour_of_day = (self.sim.tick * self.sim.dt_hours) % 24.0
        parts.append(float(np.sin(2 * np.pi * hour_of_day / 24.0)))
        parts.append(float(np.cos(2 * np.pi * hour_of_day / 24.0)))

        return np.array(parts, dtype=np.float32)
