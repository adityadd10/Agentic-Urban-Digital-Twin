"""RB-S: the strong rule-based baseline for twin-v3 (protocol
2026-09-29_twin_v3_success_criteria, addendum 6; dev doc §3.9).

RB-S emits the **v3 action arrays**, which `UDTMultiAgentEnvV3` decodes exactly
as it decodes MARL actions (same job slots, automatic ambulance assignment,
cadence and horizon). It reads the full simulator state (an operator with full
situational awareness), is reactive (no forecasting), and is built per sector
so Gate A can replace one sector with its fixed default. Only the four
thresholds in `RBSParams` are tuned (train split only); everything else is
fixed by addendum 6.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import numpy.typing as npt
import yaml

from udt.common.models import AssetType
from udt.envs.multi_env_v3 import (
    AGENT_HEALTH,
    AGENT_POWER,
    AGENT_TRANSPORT,
    N_JOB_SLOTS,
    TRANSFER_OPTIONS,
    UDTMultiAgentEnvV3,
    job_priority_order,
)
from udt.twin import crew as crew_mod
from udt.twin import demand
from udt.twin.demand import SURGE_MAX_HOURS, effective_free_beds
from udt.twin.graph import dependency_edges_of
from udt.twin.power import DESHED_THRESHOLD, MAX_SHED_TIER, OVERLOAD_THRESHOLD, post_shed_ratio

URGENT_SLACK_HOURS = 1.0
FROZEN_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "rbs_v3.yaml"
REPAIRABLE = (AssetType.HOSPITAL, AssetType.SUBSTATION, AssetType.WATER)
Actions = dict[str, npt.NDArray[np.int64]]


@dataclass(frozen=True)
class RBSParams:
    transfer_buffer_h: float = 2.0
    transfer_queue_ratio: float = 0.25
    divert_queue: int = 10
    surge_queue: int = 1

    @classmethod
    def load(cls, path: str | Path) -> RBSParams:
        return cls(**yaml.safe_load(Path(path).read_text())["params"])

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _option_index(count: int, urgency: str) -> int:
    for i, opt in enumerate(TRANSFER_OPTIONS):
        if opt == (count, urgency):
            return i
    raise ValueError(f"no transfer option ({count}, {urgency})")


class RuleBasedStrongV3:
    name = "rbs_v3"

    def __init__(self, params: RBSParams | None = None) -> None:
        self.params = params or RBSParams()

    @classmethod
    def frozen(cls) -> RuleBasedStrongV3:
        """The tuned, frozen RB-S (`configs/rbs_v3.yaml`, git tag `rbs-v3`). Use this
        for every gate and comparison; the defaults exist only for tuning and tests."""
        return cls(RBSParams.load(FROZEN_CONFIG))

    # ------------------------------------------------------------------ act
    def act(self, env: UDTMultiAgentEnvV3) -> Actions:
        return {
            AGENT_HEALTH: self.health_action(env),
            AGENT_POWER: self.power_action(env),
            AGENT_TRANSPORT: self.transport_action(env),
        }

    # --------------------------------------------------------------- health
    def health_action(self, env: UDTMultiAgentEnvV3) -> npt.NDArray[np.int64]:
        assert env.sim is not None
        g, tick, dt = env.sim.graph, env.sim.tick, env.sim.dt_hours
        p = self.params
        pending_from = {
            h: sum(1 for j in g.graph.get("transfer_requests", []) if j["from_hospital_id"] == h)
            for h in env._hospital_ids
        }
        free = {h: effective_free_beds(g.nodes[h]["asset"]) for h in env._hospital_ids}
        out: list[int] = []
        for h in env._hospital_ids:
            asset = g.nodes[h]["asset"]
            attrs = asset.attributes
            queue: list[int] = list(attrs.get("queue_arrivals", []))
            q = len(queue)
            effective_beds = round(int(attrs.get("beds_total", 0)) * asset.functional_level)

            # transfer request
            option = 0
            if pending_from[h] == 0:
                buffer_low = False
                for e in dependency_edges_of(g, h):
                    if e.kind not in ("power", "water"):
                        continue
                    st = env.sim.edge_states.get(e.edge_id)
                    if st is None or st.capacity_hours <= 0:
                        continue
                    drawing = st.remaining_hours < st.capacity_hours
                    if drawing and st.remaining_hours < p.transfer_buffer_h:
                        buffer_low = True
                queue_high = q > 0 and (
                    effective_beds <= 0 or q > p.transfer_queue_ratio * effective_beds
                )
                if buffer_low or queue_high:
                    slack = (
                        demand.PATIENT_WAIT_DEADLINE_HOURS - (tick - min(queue)) * dt
                        if queue
                        else float("inf")
                    )
                    if slack < URGENT_SLACK_HOURS:
                        count = 5 if q - free[h] >= 4 else 2
                        option = _option_index(count, "urgent")
                    else:
                        option = _option_index(2, "routine")

            # divert (with hysteresis)
            others_accepting = any(
                o != h and not g.nodes[o]["asset"].attributes.get("divert") and free[o] > 0
                for o in env._hospital_ids
            )
            if attrs.get("divert"):
                divert = q >= p.divert_queue / 2
            else:
                divert = q >= p.divert_queue and others_accepting

            # surge
            budget_left = float(attrs.get("surge_hours_used", 0.0)) < SURGE_MAX_HOURS
            surge = free[h] == 0 and q >= p.surge_queue and budget_left

            out += [option, int(divert), int(surge)]
        return np.array(out, dtype=np.int64)

    # ---------------------------------------------------------------- power
    def power_action(self, env: UDTMultiAgentEnvV3) -> npt.NDArray[np.int64]:
        assert env.sim is not None
        g, rn = env.sim.graph, env._road_network
        out: list[int] = []
        if env.shedding_enabled:
            for s in env._substation_ids:
                attrs = g.nodes[s]["asset"].attributes
                cap = float(attrs.get("capacity_mw", 0.0))
                tier = int(attrs.get("shed_tier", 0))
                if cap > 0:
                    ratio = post_shed_ratio(float(attrs.get("load_mw", 0.0)), cap, tier)
                    if ratio > OVERLOAD_THRESHOLD and tier < MAX_SHED_TIER:
                        tier += 1
                    elif ratio < DESHED_THRESHOLD and tier > 0:
                        tier -= 1
                out.append(tier)
        out.append(self._crew_choice(env, g, rn))
        return np.array(out, dtype=np.int64)

    def _crew_choice(self, env: UDTMultiAgentEnvV3, g: nx.DiGraph[str], rn: Any) -> int:
        crew = crew_mod.crew_state(g, rn)
        current = crew.get("target")

        def reachable(asset_id: str) -> bool:
            lon, lat = g.nodes[asset_id]["asset"].geometry["coordinates"][:2]
            return rn.travel_time_minutes(crew["node"], rn.nearest_node(lon, lat)) is not None

        if (
            current is not None
            and crew["status"] != "idle"
            and g.nodes[current]["asset"].intrinsic_level < 1.0
        ):
            return 0  # keep working on / travelling to the current (still damaged) target
        damaged = [f for f in env._facility_ids if g.nodes[f]["asset"].intrinsic_level < 1.0]

        def dependants(asset_id: str) -> int:
            return sum(
                1
                for d in nx.descendants(g, asset_id)
                if g.nodes[d]["asset"].asset_type in REPAIRABLE
            )

        for f in sorted(
            damaged, key=lambda f: (-dependants(f), g.nodes[f]["asset"].intrinsic_level)
        ):
            if reachable(f):
                return env._facility_ids.index(f) + 1
        return 0

    # ------------------------------------------------------------ transport
    def transport_action(self, env: UDTMultiAgentEnvV3) -> npt.NDArray[np.int64]:
        assert env.sim is not None
        g, rn = env.sim.graph, env._road_network
        jobs = {j["request_id"]: j for j in job_priority_order(g, env.sim.tick, env.sim.dt_hours)}
        hosp_nodes = {
            h: rn.nearest_node(*g.nodes[h]["asset"].geometry["coordinates"][:2])
            for h in env._hospital_ids
        }
        out = [0] * N_JOB_SLOTS
        for slot, job_id in enumerate(env._slots[:N_JOB_SLOTS]):
            job = jobs.get(job_id)
            if job is None:
                continue
            dest = self.destination(env, job, hosp_nodes)
            out[slot] = 0 if dest is None else env._hospital_ids.index(dest) + 1
        return np.array(out, dtype=np.int64)

    @staticmethod
    def destination(
        env: UDTMultiAgentEnvV3, job: dict[str, Any], hosp_nodes: dict[str, str]
    ) -> str | None:
        assert env.sim is not None
        g, rn = env.sim.graph, env._road_network
        pickup = rn.nearest_node(*job["location"])
        source = job.get("from_hospital_id") if job.get("kind") == "transfer" else None
        candidates: list[tuple[float, str]] = []
        for h, node in hosp_nodes.items():
            if h == source:
                continue
            t = rn.travel_time_minutes(pickup, node)
            if t is not None:
                candidates.append((t, h))
        candidates.sort()

        def accepting(h: str) -> bool:
            return not g.nodes[h]["asset"].attributes.get("divert")

        for test in (
            lambda h: accepting(h) and effective_free_beds(g.nodes[h]["asset"]) > 0,
            accepting,
            lambda h: True,
        ):
            for _, h in candidates:
                if test(h):
                    return h
        return None
