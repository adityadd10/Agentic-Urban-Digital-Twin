"""SOP baseline: "current practice" for twin-v3, coded from published guidance
(results/protocol/2026-10-02_sop_baseline.md). Untuned by design: every threshold is
the most natural reading of the source rule, never fitted to simulated floods.

- Health (NDMA Hospital Safety 2016): surge when full (§4.10); transfer to the nearest
  equipped hospital when full (§4.14); evacuate an affected hospital (critical patients
  to networked hospitals).
- Power (NDMA Urban Flooding 2010; utility practice): restore vital installations first:
  substations supplying hospitals, then pumps supplying hospitals, then the rest.
- Transport (108 EMS practice): serve every call with the nearest available ambulance
  (env-assigned) to the nearest suitable (functioning) facility.

Same v3 action interface as RB-S and MARL (`UDTMultiAgentEnvV3` decodes it).
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import numpy.typing as npt

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
from udt.twin.demand import SURGE_MAX_HOURS, effective_free_beds
from udt.twin.graph import dependency_edges_of

FUNCTIONING_LEVEL = 0.5


def _supply_failed(env: UDTMultiAgentEnvV3, h: str) -> bool:
    """A power or water buffer of the hospital is exhausted."""
    assert env.sim is not None
    for e in dependency_edges_of(env.sim.graph, h):
        if e.kind in ("power", "water"):
            st = env.sim.edge_states.get(e.edge_id)
            if st is not None and st.capacity_hours > 0 and st.remaining_hours <= 0:
                return True
    return False


class SOPBaselineV3:
    name = "sop_v3"

    def act(self, env: UDTMultiAgentEnvV3) -> dict[str, npt.NDArray[np.int64]]:
        return {
            AGENT_HEALTH: self.health_action(env),
            AGENT_POWER: self.power_action(env),
            AGENT_TRANSPORT: self.transport_action(env),
        }

    # ---------------------------------------------------------------- health
    def health_action(self, env: UDTMultiAgentEnvV3) -> npt.NDArray[np.int64]:
        assert env.sim is not None
        g = env.sim.graph
        out: list[int] = []
        for h in env._hospital_ids:
            asset = g.nodes[h]["asset"]
            a = asset.attributes
            queue = len(a.get("queue_arrivals", []))
            free = effective_free_beds(asset)
            pending = any(j["from_hospital_id"] == h for j in g.graph.get("transfer_requests", []))
            patients = int(a.get("beds_occupied", 0)) + queue
            affected = asset.intrinsic_level < 1.0 or _supply_failed(env, h)
            option, divert = 0, 0
            if affected and patients > 0:  # evacuation of an affected hospital
                option, divert = TRANSFER_OPTIONS.index((5, "urgent")), 1
            elif free == 0 and queue > 0 and not pending:  # §4.14 area networking
                option = TRANSFER_OPTIONS.index((2, "urgent"))
            budget = float(a.get("surge_hours_used", 0.0)) < SURGE_MAX_HOURS
            surge = int(free == 0 and queue > 0 and budget)  # §4.10 overflow
            out += [option, divert, surge]
        return np.array(out, dtype=np.int64)

    # ----------------------------------------------------------------- power
    def power_action(self, env: UDTMultiAgentEnvV3) -> npt.NDArray[np.int64]:
        assert env.sim is not None
        g, rn = env.sim.graph, env._road_network
        n_heads = len(env.action_space(AGENT_POWER).nvec)  # type: ignore[attr-defined]
        out = [0] * n_heads
        out[-1] = self._crew_choice(env, g, rn)
        return np.array(out, dtype=np.int64)

    @staticmethod
    def _priority_class(g: nx.DiGraph[str], f: str) -> int:
        """0 = substation supplying a hospital, 1 = pump supplying a hospital, 2 = other."""
        asset = g.nodes[f]["asset"]
        feeds_hospital = any(
            g.nodes[c]["asset"].asset_type == AssetType.HOSPITAL for c in g.successors(f)
        )
        if feeds_hospital and asset.asset_type == AssetType.SUBSTATION:
            return 0
        if feeds_hospital and asset.asset_type == AssetType.WATER:
            return 1
        return 2

    def _crew_choice(self, env: UDTMultiAgentEnvV3, g: nx.DiGraph[str], rn: Any) -> int:
        crew = crew_mod.crew_state(g, rn)
        current = crew.get("target")
        if (
            current is not None
            and crew["status"] != "idle"
            and g.nodes[current]["asset"].intrinsic_level < 1.0
        ):
            return 0
        damaged = [f for f in env._facility_ids if g.nodes[f]["asset"].intrinsic_level < 1.0]
        for f in sorted(
            damaged, key=lambda f: (self._priority_class(g, f), g.nodes[f]["asset"].intrinsic_level)
        ):
            lon, lat = g.nodes[f]["asset"].geometry["coordinates"][:2]
            if rn.travel_time_minutes(crew["node"], rn.nearest_node(lon, lat)) is not None:
                return env._facility_ids.index(f) + 1
        return 0

    # ------------------------------------------------------------- transport
    def transport_action(self, env: UDTMultiAgentEnvV3) -> npt.NDArray[np.int64]:
        assert env.sim is not None
        g, rn = env.sim.graph, env._road_network
        jobs = {j["request_id"]: j for j in job_priority_order(g, env.sim.tick, env.sim.dt_hours)}
        nodes = {
            h: rn.nearest_node(*g.nodes[h]["asset"].geometry["coordinates"][:2])
            for h in env._hospital_ids
        }
        out = [0] * N_JOB_SLOTS
        for slot, job_id in enumerate(env._slots[:N_JOB_SLOTS]):
            job = jobs.get(job_id)
            if job is None:
                continue
            pickup = rn.nearest_node(*job["location"])
            source = job.get("from_hospital_id") if job.get("kind") == "transfer" else None
            options = sorted(
                (t, h)
                for h, node in nodes.items()
                if h != source and (t := rn.travel_time_minutes(pickup, node)) is not None
            )
            dest = next(
                (
                    h
                    for _, h in options
                    if g.nodes[h]["asset"].functional_level >= FUNCTIONING_LEVEL
                ),
                options[0][1] if options else None,
            )
            out[slot] = 0 if dest is None else env._hospital_ids.index(dest) + 1
        return np.array(out, dtype=np.int64)
