"""Twin-v3 gate conditions (protocol 2026-09-29_twin_v3_success_criteria §6, addendum 7).

Every condition is the frozen RB-S (`RuleBasedStrongV3.frozen()`, tag `rbs-v3`) with only
the named sector replaced, so a gate isolates exactly one decision. `LookaheadCeiling` is
Gate F's upper-bound policy: it scores the Gate B–E candidates with `twin.counterfactual.
simulate` before acting.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import numpy.typing as npt

from udt.agents.rbs_v3 import REPAIRABLE, RBSParams, RuleBasedStrongV3
from udt.common.models import AgentAction, AssetType
from udt.constraints.engine import check
from udt.envs.multi_env_v3 import (
    AGENT_HEALTH,
    AGENT_POWER,
    AGENT_TRANSPORT,
    N_JOB_SLOTS,
    UDTMultiAgentEnvV3,
    job_priority_order,
)
from udt.twin import crew as crew_mod
from udt.twin.counterfactual import simulate
from udt.twin.demand import SURGE_MAX_HOURS, effective_free_beds
from udt.twin.graph import dependency_edges_of

Actions = dict[str, npt.NDArray[np.int64]]
FUNCTIONING_LEVEL = 0.5
CEILING_ROLLOUTS = 5
CEILING_HORIZON_TICKS = 24


def _frozen_params() -> RBSParams:
    return RuleBasedStrongV3.frozen().params


def _hospital_nodes(env: UDTMultiAgentEnvV3) -> dict[str, str]:
    assert env.sim is not None
    g, rn = env.sim.graph, env._road_network
    return {
        h: rn.nearest_node(*g.nodes[h]["asset"].geometry["coordinates"][:2])
        for h in env._hospital_ids
    }


def _reachable_by_time(
    env: UDTMultiAgentEnvV3, job: dict[str, Any], hosp_nodes: dict[str, str]
) -> list[tuple[float, str]]:
    """(travel time, hospital) from the pickup, reachable only, never a transfer's source."""
    assert env.sim is not None
    rn = env._road_network
    pickup = rn.nearest_node(*job["location"])
    source = job.get("from_hospital_id") if job.get("kind") == "transfer" else None
    out = []
    for h, node in hosp_nodes.items():
        if h == source:
            continue
        t = rn.travel_time_minutes(pickup, node)
        if t is not None:
            out.append((t, h))
    return sorted(out)


def _transport_with(env: UDTMultiAgentEnvV3, pick: Any) -> npt.NDArray[np.int64]:
    """Select every visible slot with destination `pick(env, job, hosp_nodes)`."""
    assert env.sim is not None
    jobs = {
        j["request_id"]: j
        for j in job_priority_order(env.sim.graph, env.sim.tick, env.sim.dt_hours)
    }
    nodes = _hospital_nodes(env)
    out = [0] * N_JOB_SLOTS
    for slot, job_id in enumerate(env._slots[:N_JOB_SLOTS]):
        job = jobs.get(job_id)
        if job is None:
            continue
        dest = pick(env, job, nodes)
        out[slot] = 0 if dest is None else env._hospital_ids.index(dest) + 1
    return np.array(out, dtype=np.int64)


def nearest_destination(
    env: UDTMultiAgentEnvV3, job: dict[str, Any], nodes: dict[str, str]
) -> str | None:
    """Gate A transport-fixed / E-uncoordinated rule: nearest reachable, beds/divert ignored."""
    options = _reachable_by_time(env, job, nodes)
    return options[0][1] if options else None


def nearest_functioning_destination(
    env: UDTMultiAgentEnvV3, job: dict[str, Any], nodes: dict[str, str]
) -> str | None:
    """Gate B alternative: nearest reachable with functional level >= 0.5, else nearest."""
    assert env.sim is not None
    options = _reachable_by_time(env, job, nodes)
    for _, h in options:
        if env.sim.graph.nodes[h]["asset"].functional_level >= FUNCTIONING_LEVEL:
            return h
    return options[0][1] if options else None


def _idle_count(env: UDTMultiAgentEnvV3) -> int:
    assert env.sim is not None
    g = env.sim.graph
    return sum(
        1
        for n in g.nodes
        if g.nodes[n]["asset"].asset_type == AssetType.AMBULANCE
        and g.nodes[n]["asset"].attributes.get("status") == "idle"
    )


# ---------------------------------------------------------------- Gate A
class HealthFixed(RuleBasedStrongV3):
    name = "a_health_fixed"

    def health_action(self, env: UDTMultiAgentEnvV3) -> npt.NDArray[np.int64]:
        return np.zeros(3 * len(env._hospital_ids), dtype=np.int64)


class PowerFixed(RuleBasedStrongV3):
    name = "a_power_fixed"

    def power_action(self, env: UDTMultiAgentEnvV3) -> npt.NDArray[np.int64]:
        return np.zeros(len(env.action_space(AGENT_POWER).nvec), dtype=np.int64)  # type: ignore[attr-defined]


class TransportFixed(RuleBasedStrongV3):
    """A-transport-fixed; identical to E-uncoordinated (addendum 7)."""

    name = "a_transport_fixed"

    def transport_action(self, env: UDTMultiAgentEnvV3) -> npt.NDArray[np.int64]:
        return _transport_with(env, nearest_destination)


# ---------------------------------------------------------------- Gate B
class NearestFunctioningDestination(RuleBasedStrongV3):
    name = "b_nearest_functioning"

    def transport_action(self, env: UDTMultiAgentEnvV3) -> npt.NDArray[np.int64]:
        return _transport_with(env, nearest_functioning_destination)


# ---------------------------------------------------------------- Gate C
class _Allocation(RuleBasedStrongV3):
    first_kind = "call"

    def transport_action(self, env: UDTMultiAgentEnvV3) -> npt.NDArray[np.int64]:
        assert env.sim is not None
        base = super().transport_action(env)  # RB-S destinations for every slot
        jobs = {
            j["request_id"]: j
            for j in job_priority_order(env.sim.graph, env.sim.tick, env.sim.dt_hours)
        }

        def kind(slot: int) -> str:
            job = jobs.get(env._slots[slot]) if slot < len(env._slots) else None
            return (
                "none" if job is None else ("transfer" if job.get("kind") == "transfer" else "call")
            )

        capacity = _idle_count(env)
        out = np.zeros_like(base)
        second = "transfer" if self.first_kind == "call" else "call"
        for wanted in (self.first_kind, second):
            for slot in range(N_JOB_SLOTS):
                if capacity > 0 and base[slot] > 0 and kind(slot) == wanted:
                    out[slot] = base[slot]
                    capacity -= 1
        return out


class CallsFirst(_Allocation):
    name = "c_calls_first"
    first_kind = "call"


class TransfersFirst(_Allocation):
    name = "c_transfers_first"
    first_kind = "transfer"


# ---------------------------------------------------------------- Gate D
class NearestReachableRepair(RuleBasedStrongV3):
    name = "d_nearest_reachable"

    def _crew_choice(self, env: UDTMultiAgentEnvV3, g: nx.DiGraph[str], rn: Any) -> int:
        crew = crew_mod.crew_state(g, rn)
        current = crew.get("target")
        if (
            current is not None
            and crew["status"] != "idle"
            and g.nodes[current]["asset"].intrinsic_level < 1.0
        ):
            return 0
        best: tuple[float, str] | None = None
        for f in env._facility_ids:
            if g.nodes[f]["asset"].intrinsic_level >= 1.0:
                continue
            lon, lat = g.nodes[f]["asset"].geometry["coordinates"][:2]
            t = rn.travel_time_minutes(crew["node"], rn.nearest_node(lon, lat))
            if t is not None and (best is None or t < best[0]):
                best = (t, f)
        return 0 if best is None else env._facility_ids.index(best[1]) + 1


# ---------------------------------------------------------------- Gate E
def receiving_hospital(env: UDTMultiAgentEnvV3) -> str | None:
    """Most inbound ambulances (carrying, or assigned with it as destination); None if tied/zero."""
    assert env.sim is not None
    g = env.sim.graph
    inbound = dict.fromkeys(env._hospital_ids, 0)
    for n in g.nodes:
        a = g.nodes[n]["asset"]
        if a.asset_type != AssetType.AMBULANCE or a.attributes.get("status") == "idle":
            continue
        dest = a.attributes.get("delivery_hospital_id") or a.attributes.get("home_hospital_id")
        if dest in inbound:
            inbound[dest] += 1
    top = max(inbound.values())
    leaders = [h for h, v in inbound.items() if v == top]
    return leaders[0] if top > 0 and len(leaders) == 1 else None


def supply_chain(g: nx.DiGraph[str], hospital: str) -> set[str]:
    chain: set[str] = set()
    for e in dependency_edges_of(g, hospital):
        if e.kind in ("power", "water"):
            chain.add(e.supplier)
            chain |= {
                e2.supplier for e2 in dependency_edges_of(g, e.supplier) if e2.kind == "power"
            }
    return chain


class Coordinated(RuleBasedStrongV3):
    name = "e_coordinated"

    def health_action(self, env: UDTMultiAgentEnvV3) -> npt.NDArray[np.int64]:
        assert env.sim is not None
        out = super().health_action(env)
        target = receiving_hospital(env)
        if target is not None:
            g = env.sim.graph
            i = env._hospital_ids.index(target)
            attrs = g.nodes[target]["asset"].attributes
            inbound = sum(
                1
                for n in g.nodes
                if g.nodes[n]["asset"].asset_type == AssetType.AMBULANCE
                and g.nodes[n]["asset"].attributes.get("status") != "idle"
                and (
                    g.nodes[n]["asset"].attributes.get("delivery_hospital_id")
                    or g.nodes[n]["asset"].attributes.get("home_hospital_id")
                )
                == target
            )
            budget = float(attrs.get("surge_hours_used", 0.0)) < SURGE_MAX_HOURS
            if budget and effective_free_beds(g.nodes[target]["asset"]) <= inbound:
                out[3 * i + 2] = 1
        return out

    def _crew_choice(self, env: UDTMultiAgentEnvV3, g: nx.DiGraph[str], rn: Any) -> int:
        target = receiving_hospital(env)
        crew = crew_mod.crew_state(g, rn)
        current = crew.get("target")
        if (
            current is not None
            and crew["status"] != "idle"
            and g.nodes[current]["asset"].intrinsic_level < 1.0
        ):
            return 0
        if target is not None:
            chain = supply_chain(g, target)

            def dependants(asset_id: str) -> int:
                return sum(
                    1
                    for d in nx.descendants(g, asset_id)
                    if g.nodes[d]["asset"].asset_type in REPAIRABLE
                )

            damaged = [
                f
                for f in env._facility_ids
                if f in chain and g.nodes[f]["asset"].intrinsic_level < 1.0
            ]
            for f in sorted(
                damaged, key=lambda f: (-dependants(f), g.nodes[f]["asset"].intrinsic_level)
            ):
                lon, lat = g.nodes[f]["asset"].geometry["coordinates"][:2]
                if rn.travel_time_minutes(crew["node"], rn.nearest_node(lon, lat)) is not None:
                    return env._facility_ids.index(f) + 1
        return super()._crew_choice(env, g, rn)


Uncoordinated = TransportFixed  # addendum 7: the same policy


# ---------------------------------------------------------------- Gate F
CEILING_CANDIDATES: tuple[type[RuleBasedStrongV3], ...] = (
    RuleBasedStrongV3,
    NearestFunctioningDestination,
    CallsFirst,
    TransfersFirst,
    NearestReachableRepair,
    Coordinated,
    Uncoordinated,
)


class LookaheadCeiling:
    """Gate F: at each decision, score every candidate's joint action with
    `simulate` (5 rollouts, 2 h, base_seed = tick; same seeds for all) by mean
    (new deaths + unmet patient-hours) and act with the lowest (ties: candidate order)."""

    name = "f_lookahead_ceiling"

    def __init__(self, params: RBSParams | None = None) -> None:
        p = params or _frozen_params()
        self.candidates = [cls(p) for cls in CEILING_CANDIDATES]
        self.last_scores: list[float] = []

    def act(self, env: UDTMultiAgentEnvV3) -> Actions:
        assert env.sim is not None
        best: tuple[float, int, Actions] | None = None
        self.last_scores = []
        for i, cand in enumerate(self.candidates):
            acts = cand.act(env)
            requests, divert, surge = env._decode_health(acts[AGENT_HEALTH])
            shed, repair = env._decode_power(acts[AGENT_POWER])
            assignment, destination, _ = env._decode_transport(acts[AGENT_TRANSPORT])
            action = AgentAction(
                repair_target=repair,
                ambulance_assignment=assignment or None,
                ambulance_destination=destination or None,
                transfer_requests=requests or None,
                divert=divert,
                surge=surge,
                shed_tier=shed,
            )
            action = check(env.sim.graph, action, road_network=env._road_network).repaired_action
            result = simulate(
                env.sim,
                action,
                degradation_fn=env._degradation_fn,
                incident=env.incident,
                horizon_ticks=CEILING_HORIZON_TICKS,
                n_rollouts=CEILING_ROLLOUTS,
                base_seed=env.sim.tick,
            )
            score = float(
                np.mean([r.new_patient_deaths + r.unmet_patient_hours for r in result.rollouts])
            )
            self.last_scores.append(score)
            if best is None or score < best[0]:
                best = (score, i, acts)
        assert best is not None
        return best[2]


GATE_POLICIES: dict[str, Any] = {
    "rbs": RuleBasedStrongV3,
    "a_health_fixed": HealthFixed,
    "a_power_fixed": PowerFixed,
    "a_transport_fixed": TransportFixed,  # = e_uncoordinated
    "b_nearest_functioning": NearestFunctioningDestination,
    "c_calls_first": CallsFirst,
    "c_transfers_first": TransfersFirst,
    "d_nearest_reachable": NearestReachableRepair,
    "e_coordinated": Coordinated,
    "f_lookahead_ceiling": LookaheadCeiling,
}


def make_policy(name: str) -> Any:
    return GATE_POLICIES[name](_frozen_params())
