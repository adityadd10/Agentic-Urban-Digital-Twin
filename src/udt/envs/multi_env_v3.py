"""Twin-v3 multi-agent environment (dev doc §3.9, "Twin-v3 environment
specification"; `ENV_VERSION_V3`).

A PettingZoo `ParallelEnv` with the same three agents as v2 (`health`, `power`,
`transport`) on the twin-v3 network (3 hospitals, 3 substations, 3 pumps,
6 ambulances, 1 repair crew). `envs/multi_env.py` (v2) is unchanged; this env
switches the twin-v3 mechanics on for its process (`udt.twin.modes`).

- **Timing.** Physics every 5 min; one decision per 15 min. One-shot actions
  (dispatches, transfer requests, crew orders) apply on the first tick of the
  interval; divert/surge/shed settings persist until changed. Reward is summed
  over the interval; observations are taken at its end. Jobs created during
  an interval (including transfers health requests now) are seen next decision.
- **Same path as the rule baseline.** Actions are decoded into an
  `AgentAction`, the form RB-S emits, checked by `constraints.check`, and
  applied by `Simulator.step`.
- **Transport.** It sees up to `N_JOB_SLOTS` jobs (street calls and transfer
  requests) in a fixed order: least slack before the patient's 4 h deadline
  first, then urgent before routine, then oldest. For each slot it chooses
  "defer" or a destination hospital. Selected slots are served in slot order;
  each gets the nearest idle ambulance whose three legs (to the pickup, to the
  destination, back home) are all reachable now. A job is deferred, never
  cancelled or rerouted, if no such ambulance is left, the destination is
  unreachable, or a transfer would go back to its own source. Every decision's
  served and deferred jobs are reported in `infos`.
- **Health.** Per hospital: transfer request {none, 2 routine, 2 urgent,
  5 urgent} x divert {accept, divert} x surge {off, on}.
- **Power.** The crew's target {keep current, one of the facilities}. The shed
  tier head was removed by the pre-registered pre-check (protocol addendum 2,
  `results/twin_v3_gates/shedding_precheck.md`: no causal effect on patient
  outcomes). `shedding_enabled=True` restores it for diagnostics only; it is
  not part of the frozen twin-v3 action space.

Episodes always run the full horizon unless `terminate_on_stabilization` is
set (the v2 env's early stop made learned-policy evaluations shorter than the
rule baseline's; see the constructor note).

Reward: reward A's form (`envs/reward.py`), with twin-v3 normalisers supplied
explicitly via `reward_config` (the v3 normalisers are fitted on RB-S later;
there is deliberately no default).
"""

from __future__ import annotations

import functools
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium import spaces
from pettingzoo import ParallelEnv
from shapely.geometry import shape

from udt.common.models import (
    AgentAction,
    AssetType,
    DependencyGraph,
    Incident,
    Scenario,
    TwinState,
)
from udt.common.versions import ENV_VERSION_V3
from udt.constraints.engine import check
from udt.envs.reward import load_reward_normalisers, load_reward_options, tick_terms
from udt.incidents.degradations.flood import (
    MAX_DEPTH_AT_SEVERITY_1_M,
    SusceptibilityRaster,
    flood_phase,
    make_flood_degradation_fn,
)
from udt.scenarios.generator import (
    apply_initial_conditions,
    generate_flood_scenario,
    onset_hour_of_day,
)
from udt.scenarios.suite import load_suite
from udt.twin import crew as crew_mod
from udt.twin import demand
from udt.twin.ambulances import generate_requests, spawn_ambulances
from udt.twin.demand import SURGE_MAX_HOURS, effective_free_beds
from udt.twin.graph import dependency_edges_of
from udt.twin.modes import enable_twin_v3
from udt.twin.power import update_substation_load
from udt.twin.road_network import RoadNetwork
from udt.twin.simulator import Simulator

DECISION_INTERVAL_TICKS = 3
MAX_TICKS = 288
GOAL_LOW, GOAL_HIGH, GOAL_DIM = 0.5, 2.0, 4
STABILIZATION_TICKS = 12
STABILIZATION_LEVEL = 0.9
N_AMBULANCES_PER_HOSPITAL = 2
N_JOB_SLOTS = 6
# Health transfer options: (count, urgency); index 0 = no request.
TRANSFER_OPTIONS: tuple[tuple[int, str] | None, ...] = (
    None,
    (2, "routine"),
    (2, "urgent"),
    (5, "urgent"),
)
CREW_STATUSES = ("idle", "travelling", "working")
AMBULANCE_STATUSES = ("idle", "enroute", "returning")
UNREACHABLE_MIN = 120.0  # travel-time observation value for "no route now"
GRAPH_FILE = "dependency_graph_v3.json"

AGENT_HEALTH, AGENT_POWER, AGENT_TRANSPORT = "health", "power", "transport"
POSSIBLE_AGENTS = (AGENT_HEALTH, AGENT_POWER, AGENT_TRANSPORT)
FACILITY_TYPES = (AssetType.HOSPITAL, AssetType.SUBSTATION, AssetType.WATER)


def job_slack_hours(graph: Any, job: dict[str, Any], tick: int, dt_hours: float) -> float:
    """Hours left before the job's patient reaches the wait deadline. A transfer
    uses its source's longest-waiting queued patient; `inf` if nobody is queued
    there (it would move an admitted patient)."""
    deadline = demand.PATIENT_WAIT_DEADLINE_HOURS
    if job.get("kind") == "transfer":
        queue = graph.nodes[job["from_hospital_id"]]["asset"].attributes.get("queue_arrivals", [])
        if not queue:
            return float("inf")
        return float(deadline - (tick - min(queue)) * dt_hours)
    return float(deadline - (tick - job["requested_at_tick"]) * dt_hours)


def job_priority_order(graph: Any, tick: int, dt_hours: float) -> list[dict[str, Any]]:
    """All pending jobs in the fixed slot order: slack, urgency, age."""
    jobs = list(graph.graph.get("pending_requests", [])) + list(
        graph.graph.get("transfer_requests", [])
    )

    def key(job: dict[str, Any]) -> tuple[float, int, int]:
        urgent = job.get("kind") != "transfer" or job.get("urgency") == "urgent"
        return (
            job_slack_hours(graph, job, tick, dt_hours),
            0 if urgent else 1,
            int(job["requested_at_tick"]),
        )

    return sorted(jobs, key=key)


class UDTMultiAgentEnvV3(
    ParallelEnv[str, npt.NDArray[np.float32], npt.NDArray[np.integer[Any]]]  # type: ignore[misc]
):
    metadata: dict[str, Any] = {"name": ENV_VERSION_V3, "render_modes": []}

    def __init__(
        self,
        *,
        processed_dir: str | Path,
        reward_config: str | Path,
        scenarios: list[Scenario] | None = None,
        suite_dir: str | Path | None = None,
        scenario_split: str | None = None,
        n_ticks: int = MAX_TICKS,
        base_seed: int = 0,
        shedding_enabled: bool = False,
        terminate_on_stabilization: bool = False,
    ) -> None:
        enable_twin_v3()
        processed_dir = Path(processed_dir)
        self._base_graph = DependencyGraph.model_validate_json(
            (processed_dir / GRAPH_FILE).read_text()
        )
        with (processed_dir / "ward_boundary.geojson").open() as f:
            self._ward_boundary = json.load(f)
        self._ward_polygon = shape(self._ward_boundary["features"][0]["geometry"])
        self._raster = SusceptibilityRaster(processed_dir / "flood_susceptibility.tif")
        self._road_network = RoadNetwork.load(processed_dir / "roads_full.graphml", self._raster)

        self.reward_config = str(reward_config)
        self._reward_normalisers = load_reward_normalisers(reward_config)
        self._reward_options = load_reward_options(reward_config)
        if scenarios is not None:
            self._scenarios = list(scenarios)
        elif suite_dir is not None and scenario_split is not None:
            self._scenarios = load_suite(suite_dir, scenario_split, env_version=ENV_VERSION_V3)
        else:
            self._scenarios = []  # fresh random scenarios: smoke tests only
        self.n_ticks = n_ticks
        self.base_seed = base_seed
        self.shedding_enabled = shedding_enabled
        # Off by default: every episode runs the full horizon, exactly like the
        # rule-based harness (experiments/runner.py), so learned policies and
        # RB-S are scored over the same 24 h. The v2 env ended episodes early
        # after 12 healthy ticks, which made v2 learned-policy evaluations
        # shorter than rule-based ones (found 2026-09-29).
        self.terminate_on_stabilization = terminate_on_stabilization
        self._episode_count = 0

        assets = self._base_graph.assets
        self._hospital_ids = [a.asset_id for a in assets if a.asset_type == AssetType.HOSPITAL]
        self._substation_ids = [a.asset_id for a in assets if a.asset_type == AssetType.SUBSTATION]
        self._facility_ids = [a.asset_id for a in assets if a.asset_type in FACILITY_TYPES]
        self._n_ambulances = len(self._hospital_ids) * N_AMBULANCES_PER_HOSPITAL

        self.possible_agents: list[str] = list(POSSIBLE_AGENTS)
        self.agents: list[str] = []
        self.sim: Simulator | None = None
        self.incident: Incident | None = None
        self.goal: npt.NDArray[np.float32] = np.ones(GOAL_DIM, dtype=np.float32)
        self.episode_trace: list[TwinState] = []
        self._slots: list[str] = []
        self._prev_patient_deaths = 0
        self._prev_cascading_count = 0
        self._stable_streak = 0
        self._safety_violations_attempted = 0
        self.last_action: AgentAction | None = None

    # ------------------------------------------------------------------ spaces
    def _obs_dims(self) -> dict[str, int]:
        n_h, n_s, n_f = len(self._hospital_ids), len(self._substation_ids), len(self._facility_ids)
        common = 3 * n_f + 8 * n_h + 6 + 3 + (len(CREW_STATUSES) + n_f + 1) + 2 + GOAL_DIM
        return {
            AGENT_HEALTH: common + 2 * n_h,
            AGENT_POWER: common + 2 * n_s + 2 * n_f,
            AGENT_TRANSPORT: common
            + N_JOB_SLOTS * (5 + 3 * n_h)
            + len(AMBULANCE_STATUSES) * self._n_ambulances
            + 6 * n_h,
        }

    @functools.cache  # noqa: B019 - PettingZoo requires the same Space object per agent
    def observation_space(self, agent: str) -> spaces.Space[Any]:
        return spaces.Box(low=-1.0, high=10.0, shape=(self._obs_dims()[agent],), dtype=np.float32)

    @functools.cache  # noqa: B019
    def action_space(self, agent: str) -> spaces.Space[Any]:
        if agent == AGENT_HEALTH:
            return spaces.MultiDiscrete([len(TRANSFER_OPTIONS), 2, 2] * len(self._hospital_ids))
        if agent == AGENT_POWER:
            shed = [4] * len(self._substation_ids) if self.shedding_enabled else []
            return spaces.MultiDiscrete([*shed, 1 + len(self._facility_ids)])
        if agent == AGENT_TRANSPORT:
            return spaces.MultiDiscrete([1 + len(self._hospital_ids)] * N_JOB_SLOTS)
        raise ValueError(f"unknown agent {agent!r}")

    def action_mask(self, agent: str) -> list[npt.NDArray[np.bool_]]:
        """All choices allowed (infeasible ones become no-ops or deferrals)."""
        nvec = self.action_space(agent).nvec  # type: ignore[attr-defined]
        return [np.ones(int(n), dtype=bool) for n in nvec]

    # ------------------------------------------------------------ resume
    _RESUME_EXCLUDE = ("_raster", "_road_network")

    def resume_state(self) -> bytes:
        fn = getattr(self, "_degradation_fn", None)
        sim = self.sim
        saved_rn = sim.road_network if sim is not None else None
        saved_raster = fn.raster if fn is not None else None
        try:
            if sim is not None:
                sim.road_network = None
            if fn is not None:
                fn.raster = None
            state = {k: v for k, v in self.__dict__.items() if k not in self._RESUME_EXCLUDE}
            state["_road_network_state"] = self._road_network.mutable_state()
            return pickle.dumps(state)
        finally:
            if sim is not None:
                sim.road_network = saved_rn
            if fn is not None:
                fn.raster = saved_raster

    def load_resume_state(self, blob: bytes) -> None:
        state = pickle.loads(blob)
        rn_state = state.pop("_road_network_state")
        self.__dict__.update(state)
        if self.sim is not None:
            self.sim.road_network = self._road_network
        if getattr(self, "_degradation_fn", None) is not None:
            self._degradation_fn.raster = self._raster
        self._road_network.load_mutable_state(rn_state, getattr(self, "incident", None))

    # -------------------------------------------------------------- episode
    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, npt.NDArray[np.float32]], dict[str, dict[str, Any]]]:
        episode_seed = seed if seed is not None else self.base_seed + self._episode_count
        self._episode_count += 1
        if self._scenarios:
            index = (options or {}).get("scenario_index", episode_seed % len(self._scenarios))
            scenario = self._scenarios[int(index)]
        else:
            scenario = generate_flood_scenario(
                scenario_id=f"multi_env_v3_ep{self._episode_count}",
                ward_boundary_geojson=self._ward_boundary,
                seed=episode_seed,
            )
        self.scenario = scenario
        self.incident = scenario.incident
        self._degradation_fn = make_flood_degradation_fn(scenario.incident, self._raster)
        self.sim = Simulator(
            apply_initial_conditions(self._base_graph, scenario),
            seed=episode_seed,
            road_network=self._road_network,
            onset_hour_of_day=onset_hour_of_day(scenario),
        )
        spawn_ambulances(self.sim.graph, self._road_network, N_AMBULANCES_PER_HOSPITAL)
        self.goal = self.sim.rng.uniform(GOAL_LOW, GOAL_HIGH, size=GOAL_DIM).astype(np.float32)
        if options and "goal" in options:
            self.goal = np.asarray(options["goal"], dtype=np.float32)
        self._road_network.update_for_tick(self.incident, self.sim.tick, self.sim.dt_minutes)

        self.agents = list(self.possible_agents)
        self._prev_patient_deaths = 0
        self._prev_cascading_count = 0
        self._stable_streak = 0
        self._safety_violations_attempted = 0
        self.episode_trace = []
        self.last_action = None
        obs = self._observations()
        return obs, {a: {} for a in self.agents}

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
        transfer_requests, divert, surge = self._decode_health(actions[AGENT_HEALTH])
        shed_tier, repair_target = self._decode_power(actions[AGENT_POWER])
        assignment, destination, dispatch_log = self._decode_transport(actions[AGENT_TRANSPORT])
        action = AgentAction(
            repair_target=repair_target,
            ambulance_assignment=assignment or None,
            ambulance_destination=destination or None,
            transfer_requests=transfer_requests or None,
            divert=divert,
            surge=surge,
            shed_tier=shed_tier,
        )
        report = check(self.sim.graph, action, road_network=self._road_network)
        action = report.repaired_action
        self.last_action = action
        violations = len(report.violations)
        self._safety_violations_attempted += violations

        total_reward = 0.0
        for i in range(DECISION_INTERVAL_TICKS):
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
            first = i == 0
            snapshot = self.sim.step(
                degradation_fn=self._degradation_fn,
                repair_target=action.repair_target if first else None,
                ambulance_assignment=action.ambulance_assignment if first else None,
                ambulance_destination=action.ambulance_destination if first else None,
                transfer_requests=action.transfer_requests if first else None,
                divert=action.divert if first else None,
                surge=action.surge if first else None,
                shed_tier=action.shed_tier if first else None,
            )
            snapshot.safety_violations_attempted_cumulative = self._safety_violations_attempted
            if first:  # dispatches the simulator could not carry out (route closed this tick)
                for amb in action.ambulance_assignment or {}:
                    if self.sim.graph.nodes[amb]["asset"].attributes.get("status") == "idle":
                        dispatch_log["deferred"].append(
                            {"ambulance": amb, "reason": "route closed at dispatch"}
                        )
            total_reward += self._tick_reward(snapshot, violations if first else 0)
            self.episode_trace.append(snapshot)
            if self.sim.tick >= self.n_ticks:
                break

        terminated = self.terminate_on_stabilization and self._stable_streak >= STABILIZATION_TICKS
        truncated = self.sim.tick >= self.n_ticks and not terminated
        obs = self._observations()
        info = {
            "tick": self.sim.tick,
            "safety_violations_attempted": violations,
            "dispatch": dispatch_log,
        }
        agents = list(self.agents)
        if terminated or truncated:
            self.agents = []
        return (
            obs,
            {a: total_reward for a in agents},
            {a: terminated for a in agents},
            {a: truncated for a in agents},
            {a: dict(info) for a in agents},
        )

    # -------------------------------------------------------------- decoding
    def _decode_health(
        self, action: npt.NDArray[np.integer[Any]]
    ) -> tuple[list[tuple[str, int, str]], dict[str, bool], dict[str, bool]]:
        a = [int(x) for x in np.asarray(action).reshape(-1)]
        requests: list[tuple[str, int, str]] = []
        divert: dict[str, bool] = {}
        surge: dict[str, bool] = {}
        for i, h in enumerate(self._hospital_ids):
            option = TRANSFER_OPTIONS[a[3 * i]]
            if option is not None:
                requests.append((h, option[0], option[1]))
            divert[h] = bool(a[3 * i + 1])
            surge[h] = bool(a[3 * i + 2])
        return requests, divert, surge

    def _decode_power(
        self, action: npt.NDArray[np.integer[Any]]
    ) -> tuple[dict[str, int] | None, str | None]:
        a = [int(x) for x in np.asarray(action).reshape(-1)]
        shed: dict[str, int] | None = None
        if self.shedding_enabled:
            shed = dict(zip(self._substation_ids, a[: len(self._substation_ids)], strict=True))
        choice = a[-1]
        return shed, (self._facility_ids[choice - 1] if choice > 0 else None)

    def _legs(self, amb_id: str, job: dict[str, Any], dest_id: str) -> float | None:
        """Pickup travel time if all three legs are reachable now, else None."""
        assert self.sim is not None
        g, rn = self.sim.graph, self._road_network
        attrs = g.nodes[amb_id]["asset"].attributes
        home = attrs["home_node"]
        pickup = rn.nearest_node(*job["location"])
        dest_node = rn.nearest_node(*g.nodes[dest_id]["asset"].geometry["coordinates"][:2])
        to_pickup = rn.travel_time_minutes(home, pickup)
        to_dest = rn.travel_time_minutes(pickup, dest_node)
        back = 0.0 if dest_node == home else rn.travel_time_minutes(dest_node, home)
        if to_pickup is None or to_dest is None or back is None:
            return None
        return to_pickup

    def _decode_transport(
        self, action: npt.NDArray[np.integer[Any]]
    ) -> tuple[dict[str, str], dict[str, str], dict[str, list[dict[str, Any]]]]:
        assert self.sim is not None
        g = self.sim.graph
        a = [int(x) for x in np.asarray(action).reshape(-1)]
        jobs = {j["request_id"]: j for j in job_priority_order(g, self.sim.tick, self.sim.dt_hours)}
        idle = [
            n
            for n in g.nodes
            if g.nodes[n]["asset"].asset_type == AssetType.AMBULANCE
            and g.nodes[n]["asset"].attributes.get("status") == "idle"
        ]
        assignment: dict[str, str] = {}
        destination: dict[str, str] = {}
        log: dict[str, list[dict[str, Any]]] = {"served": [], "deferred": []}
        for slot, job_id in enumerate(self._slots):
            choice = a[slot] if slot < len(a) else 0
            if choice == 0:
                continue  # defer (the agent's choice)
            job = jobs.get(job_id)
            dest = self._hospital_ids[choice - 1]
            if job is None:
                continue  # already gone (picked up or expired)
            if job.get("kind") == "transfer" and job["from_hospital_id"] == dest:
                log["deferred"].append({"job": job_id, "reason": "transfer to its own source"})
                continue
            options = [(t, amb) for amb in idle if (t := self._legs(amb, job, dest)) is not None]
            if not options:
                reason = "no idle ambulance" if not idle else "destination or pickup unreachable"
                log["deferred"].append({"job": job_id, "reason": reason})
                continue
            _, amb = min(options)
            idle.remove(amb)
            assignment[amb] = job_id
            destination[amb] = dest
            log["served"].append({"job": job_id, "ambulance": amb, "destination": dest})
        return assignment, destination, log

    # ---------------------------------------------------------------- reward
    def _tick_reward(self, snapshot: TwinState, violations_attempted: int = 0) -> float:
        assert self.sim is not None
        raw = tick_terms(
            snapshot,
            self.sim.dt_hours,
            self._prev_patient_deaths,
            self._prev_cascading_count,
            ambulance_delay_mode=self._reward_options["ambulance_delay_mode"],
        )
        self._prev_patient_deaths = snapshot.patient_deaths_cumulative
        self._prev_cascading_count = snapshot.cascading_failure_count
        t = {k: v / self._reward_normalisers[k] for k, v in raw.items()}
        healthy = all(a.functional_level >= STABILIZATION_LEVEL for a in snapshot.assets)
        self._stable_streak = self._stable_streak + 1 if healthy else 0
        g_health, g_power, g_transport, _ = self.goal
        return -(
            float(g_health) * (t["unmet_patient_hours"] + 10.0 * t["patient_deaths"])
            + float(g_power) * t["unserved_energy_mwh"]
            + float(g_transport) * t["ambulance_response_delay_hours"]
            + float(self._reward_options["cascade_coefficient"]) * t["new_cascade_failures"]
            + 20.0 * violations_attempted
        )

    # ----------------------------------------------------------- observation
    def _observations(self) -> dict[str, npt.NDArray[np.float32]]:
        assert self.sim is not None
        ordered = job_priority_order(self.sim.graph, self.sim.tick, self.sim.dt_hours)
        self._slots = [j["request_id"] for j in ordered[:N_JOB_SLOTS]]
        common = self._common_obs()
        out = {
            AGENT_HEALTH: common + self._health_obs(),
            AGENT_POWER: common + self._power_obs(),
            AGENT_TRANSPORT: common + self._transport_obs(),
        }
        return {a: np.clip(np.array(v, dtype=np.float32), -1.0, 10.0) for a, v in out.items()}

    def _hospital_block(self, h: str) -> list[float]:
        """free beds, queue, divert, surge, surge left, power buffer, water buffer, access."""
        assert self.sim is not None
        asset = self.sim.graph.nodes[h]["asset"]
        attrs = asset.attributes
        buffers = {"power": 1.0, "water": 1.0}
        access: list[float] = []
        for e in dependency_edges_of(self.sim.graph, h):
            if e.kind in buffers:
                st = self.sim.edge_states.get(e.edge_id)
                if st is not None and st.capacity_hours > 0:
                    buffers[e.kind] = st.remaining_hours / st.capacity_hours
            elif e.kind == "access":
                road = self.sim.graph.nodes[e.supplier]["asset"].attributes
                access.append(1.0 - float(road.get("blockage", 0.0)))
        beds = max(1, int(attrs.get("beds_total", 1)))
        return [
            effective_free_beds(asset) / beds,
            min(2.0, int(attrs.get("patient_queue", 0)) / 20.0),
            float(bool(attrs.get("divert"))),
            float(bool(attrs.get("surge"))),
            max(0.0, 1.0 - float(attrs.get("surge_hours_used", 0.0)) / SURGE_MAX_HOURS),
            buffers["power"],
            buffers["water"],
            max(access) if access else 1.0,
        ]

    def _common_obs(self) -> list[float]:
        assert self.sim is not None and self.incident is not None
        g, tick = self.sim.graph, self.sim.tick
        parts: list[float] = []
        for f in self._facility_ids:
            asset = g.nodes[f]["asset"]
            depth = self._degradation_fn.depth_m(asset, tick) / MAX_DEPTH_AT_SEVERITY_1_M
            parts += [asset.functional_level, asset.intrinsic_level, depth]
        for h in self._hospital_ids:
            parts += self._hospital_block(h)
        hours = (tick - self.incident.onset_tick) * self.sim.dt_hours
        envelope, rising, peak, receding = flood_phase(self.incident, hours)
        parts += [self.incident.severity, min(1.0, hours / 24.0), envelope, rising, peak, receding]
        idle = sum(
            1
            for n in g.nodes
            if g.nodes[n]["asset"].asset_type == AssetType.AMBULANCE
            and g.nodes[n]["asset"].attributes.get("status") == "idle"
        )
        parts += [
            idle / self._n_ambulances,
            len(g.graph.get("pending_requests", [])) / 20.0,
            len(g.graph.get("transfer_requests", [])) / 20.0,
        ]
        crew = g.graph.get(crew_mod.CREW_KEY) or {"status": "idle", "target": None}
        parts += [float(crew["status"] == s) for s in CREW_STATUSES]
        parts += [float(crew.get("target") == f) for f in self._facility_ids]
        parts.append(min(2.0, float(crew.get("remaining_travel_min", 0.0)) / 60.0))
        hour = (self.sim.onset_hour_of_day + tick * self.sim.dt_hours) % 24.0
        parts += [float(np.sin(2 * np.pi * hour / 24.0)), float(np.cos(2 * np.pi * hour / 24.0))]
        parts += [float(x) for x in self.goal]
        return parts

    def _health_obs(self) -> list[float]:
        assert self.sim is not None
        g = self.sim.graph
        parts: list[float] = []
        for h in self._hospital_ids:
            pending = sum(
                1 for j in g.graph.get("transfer_requests", []) if j["from_hospital_id"] == h
            )
            inbound = 0
            for n in g.nodes:
                attrs = g.nodes[n]["asset"].attributes
                if (
                    g.nodes[n]["asset"].asset_type != AssetType.AMBULANCE
                    or attrs.get("status") == "idle"
                ):
                    continue
                if (attrs.get("delivery_hospital_id") or attrs.get("home_hospital_id")) == h:
                    inbound += 1
            parts += [min(2.0, pending / 10.0), inbound / self._n_ambulances]
        return parts

    def _power_obs(self) -> list[float]:
        assert self.sim is not None
        g, rn = self.sim.graph, self._road_network
        parts: list[float] = []
        for s in self._substation_ids:
            attrs = g.nodes[s]["asset"].attributes
            cap = max(1e-6, float(attrs.get("capacity_mw", 1.0)))
            parts += [
                min(2.0, float(attrs.get("load_mw", 0.0)) / cap),
                int(attrs.get("shed_tier", 0)) / 3.0,
            ]
        crew = crew_mod.crew_state(g, rn)
        for f in self._facility_ids:
            lon, lat = g.nodes[f]["asset"].geometry["coordinates"][:2]
            t = rn.travel_time_minutes(crew["node"], rn.nearest_node(lon, lat))
            parts += [
                float(t is not None),
                min(2.0, (t if t is not None else UNREACHABLE_MIN) / 60.0),
            ]
        return parts

    def _transport_obs(self) -> list[float]:
        assert self.sim is not None
        g, rn, tick, dt = self.sim.graph, self._road_network, self.sim.tick, self.sim.dt_hours
        jobs = {j["request_id"]: j for j in job_priority_order(g, tick, dt)}
        deadline = demand.PATIENT_WAIT_DEADLINE_HOURS
        hosp_nodes = [
            rn.nearest_node(*g.nodes[h]["asset"].geometry["coordinates"][:2])
            for h in self._hospital_ids
        ]
        parts: list[float] = []
        for slot in range(N_JOB_SLOTS):
            job = jobs.get(self._slots[slot]) if slot < len(self._slots) else None
            if job is None:
                parts += [0.0] * (5 + 3 * len(self._hospital_ids))
                continue
            transfer = job.get("kind") == "transfer"
            slack = job_slack_hours(g, job, tick, dt)
            parts += [
                1.0,
                float(transfer),
                float(not transfer or job.get("urgency") == "urgent"),
                (tick - int(job["requested_at_tick"])) * dt / deadline,
                (min(slack, 2 * deadline) / deadline),
            ]
            parts += [float(transfer and job["from_hospital_id"] == h) for h in self._hospital_ids]
            pickup = rn.nearest_node(*job["location"])
            for node in hosp_nodes:
                t = rn.travel_time_minutes(pickup, node)
                parts += [
                    float(t is not None),
                    min(2.0, (t if t is not None else UNREACHABLE_MIN) / 60.0),
                ]
        for n in g.nodes:
            if g.nodes[n]["asset"].asset_type == AssetType.AMBULANCE:
                status = g.nodes[n]["asset"].attributes.get("status", "idle")
                parts += [float(status == s) for s in AMBULANCE_STATUSES]
        for h in self._hospital_ids:
            asset = g.nodes[h]["asset"]
            block = self._hospital_block(h)
            parts += [asset.functional_level, block[0], block[1], block[2], block[7], block[3]]
        return parts
