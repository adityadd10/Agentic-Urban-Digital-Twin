"""Twin-v3 gate conditions (protocol §6, addendum 7): each replaces exactly one RB-S sector."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from udt.agents.gate_heuristics_v3 import (
    GATE_POLICIES,
    CallsFirst,
    Coordinated,
    HealthFixed,
    LookaheadCeiling,
    NearestFunctioningDestination,
    NearestReachableRepair,
    PowerFixed,
    TransportFixed,
    Uncoordinated,
    make_policy,
    nearest_destination,
    nearest_functioning_destination,
    receiving_hospital,
)
from udt.agents.rbs_v3 import RuleBasedStrongV3
from udt.envs.multi_env_v3 import AGENT_HEALTH, AGENT_POWER, AGENT_TRANSPORT, UDTMultiAgentEnvV3
from udt.twin import ambulances, crew

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def env() -> Iterator[UDTMultiAgentEnvV3]:
    saved = (ambulances.COUNT_UNCOLLECTED_CASUALTY_DEATHS, crew.REPAIR_CREW_TRAVEL)
    e = UDTMultiAgentEnvV3(
        processed_dir=REPO_ROOT / "data/processed",
        reward_config=REPO_ROOT / "configs/reward.yaml",
        suite_dir=REPO_ROOT / "data/scenarios/flood_v3",
        scenario_split="train",
    )
    yield e
    e.close()
    ambulances.COUNT_UNCOLLECTED_CASUALTY_DEATHS, crew.REPAIR_CREW_TRAVEL = saved


def _severe(env: UDTMultiAgentEnvV3, steps: int = 12) -> None:
    """A severe train scenario a few hours in, so every sector has something to do."""
    env.reset(seed=10, options={"scenario_index": 10, "goal": [1, 1, 1, 1]})
    rbs = RuleBasedStrongV3.frozen()
    for _ in range(steps):
        env.step(rbs.act(env))


@pytest.mark.phase6
def test_every_gate_policy_emits_valid_v3_actions(env: UDTMultiAgentEnvV3) -> None:
    _severe(env)
    for name in GATE_POLICIES:
        if name == "f_lookahead_ceiling":
            continue
        acts = make_policy(name).act(env)
        for agent in (AGENT_HEALTH, AGENT_POWER, AGENT_TRANSPORT):
            assert env.action_space(agent).contains(acts[agent]), (name, agent)


@pytest.mark.phase6
def test_gate_policies_use_the_frozen_rbs_thresholds() -> None:
    assert make_policy("c_calls_first").params == RuleBasedStrongV3.frozen().params


@pytest.mark.phase6
def test_each_condition_replaces_only_its_own_sector(env: UDTMultiAgentEnvV3) -> None:
    _severe(env)
    rbs = RuleBasedStrongV3.frozen().act(env)
    p = RuleBasedStrongV3.frozen().params
    for cls, changed in (
        (HealthFixed, AGENT_HEALTH),
        (PowerFixed, AGENT_POWER),
        (TransportFixed, AGENT_TRANSPORT),
        (NearestFunctioningDestination, AGENT_TRANSPORT),
        (CallsFirst, AGENT_TRANSPORT),
        (NearestReachableRepair, AGENT_POWER),
    ):
        acts = cls(p).act(env)
        for agent in (AGENT_HEALTH, AGENT_POWER, AGENT_TRANSPORT):
            if agent != changed:
                assert np.array_equal(acts[agent], rbs[agent]), (cls.__name__, agent)
    assert not HealthFixed(p).health_action(env).any()
    assert not PowerFixed(p).power_action(env).any()
    assert Uncoordinated is TransportFixed


@pytest.mark.phase6
def test_fixed_and_functioning_destination_rules(env: UDTMultiAgentEnvV3) -> None:
    _severe(env, steps=1)
    assert env.sim is not None
    g, rn = env.sim.graph, env._road_network
    nodes = {
        h: rn.nearest_node(*g.nodes[h]["asset"].geometry["coordinates"][:2])
        for h in env._hospital_ids
    }
    h1, h2, h3 = env._hospital_ids
    job = {
        "request_id": "T",
        "kind": "transfer",
        "from_hospital_id": h1,
        "location": tuple(g.nodes[h1]["asset"].geometry["coordinates"][:2]),
        "requested_at_tick": 0,
        "urgency": "urgent",
    }
    nearest = nearest_destination(env, job, nodes)
    assert nearest in (h2, h3)
    g.nodes[nearest]["asset"].attributes["divert"] = True
    assert nearest_destination(env, job, nodes) == nearest  # ignores divert status
    saved = g.nodes[nearest]["asset"].functional_level
    g.nodes[nearest]["asset"].functional_level = 0.2
    assert nearest_functioning_destination(env, job, nodes) != nearest  # skips F < 0.5
    g.nodes[nearest]["asset"].functional_level = saved
    g.nodes[nearest]["asset"].attributes["divert"] = False


@pytest.mark.phase6
def test_calls_first_keeps_calls_when_ambulances_are_short(env: UDTMultiAgentEnvV3) -> None:
    _severe(env, steps=1)
    assert env.sim is not None
    g, tick = env.sim.graph, env.sim.tick
    h1 = env._hospital_ids[0]
    loc = tuple(g.nodes[h1]["asset"].geometry["coordinates"][:2])
    g.graph["pending_requests"] = [
        {"request_id": f"C{i}", "location": (72.885, 19.075), "requested_at_tick": tick}
        for i in range(3)
    ]
    g.graph["transfer_requests"] = [
        {
            "request_id": f"T{i}",
            "kind": "transfer",
            "from_hospital_id": h1,
            "location": loc,
            "requested_at_tick": tick - 50,
            "urgency": "urgent",
        }
        for i in range(3)
    ]
    idle = [n for n in g.nodes if g.nodes[n]["asset"].attributes.get("status") == "idle"]
    for amb in idle[3:]:
        g.nodes[amb]["asset"].attributes["status"] = "enroute"  # leave 3 ambulances
    env._observations()
    acts = CallsFirst(RuleBasedStrongV3.frozen().params).transport_action(env)
    chosen = [env._slots[i] for i in range(len(env._slots)) if acts[i] > 0]
    assert len(chosen) <= 3 and all(j.startswith("C") for j in chosen)


@pytest.mark.phase6
def test_nearest_reachable_repair_picks_the_closest_damaged_site(env: UDTMultiAgentEnvV3) -> None:
    _severe(env, steps=1)
    assert env.sim is not None
    g, rn = env.sim.graph, env._road_network
    c = crew.crew_state(g, rn)
    c.update({"status": "idle", "target": None})
    for f in env._facility_ids:
        g.nodes[f]["asset"].intrinsic_level = 0.5
    choice = NearestReachableRepair(RuleBasedStrongV3.frozen().params).power_action(env)[-1]
    times = {}
    for f in env._facility_ids:
        lon, lat = g.nodes[f]["asset"].geometry["coordinates"][:2]
        t = rn.travel_time_minutes(c["node"], rn.nearest_node(lon, lat))
        if t is not None:
            times[f] = t
    assert env._facility_ids[choice - 1] == min(times, key=times.get)


@pytest.mark.phase6
def test_coordinated_surges_the_receiving_hospital(env: UDTMultiAgentEnvV3) -> None:
    _severe(env, steps=1)
    assert env.sim is not None
    g = env.sim.graph
    target = env._hospital_ids[1]
    ambs = [n for n in g.nodes if g.nodes[n]["asset"].asset_type.value == "ambulance"]
    for amb in ambs[:2]:
        g.nodes[amb]["asset"].attributes.update(
            {"status": "enroute", "delivery_hospital_id": target}
        )
    for amb in ambs[2:]:
        g.nodes[amb]["asset"].attributes["status"] = "idle"
    a = g.nodes[target]["asset"].attributes
    a.update({"beds_total": 10, "beds_occupied": 10, "surge_hours_used": 0.0, "queue_arrivals": []})
    assert receiving_hospital(env) == target
    acts = Coordinated(RuleBasedStrongV3.frozen().params).health_action(env)
    assert acts[3 * env._hospital_ids.index(target) + 2] == 1


@pytest.mark.phase6
def test_lookahead_ceiling_scores_all_candidates_and_returns_one_of_them(
    env: UDTMultiAgentEnvV3,
) -> None:
    _severe(env, steps=4)
    ceiling = LookaheadCeiling()
    acts = ceiling.act(env)
    assert len(ceiling.last_scores) == 7
    candidates = [c.act(env) for c in ceiling.candidates]
    best = int(np.argmin(ceiling.last_scores))
    for agent in acts:
        assert np.array_equal(acts[agent], candidates[best][agent])
    again = LookaheadCeiling()
    again.act(env)
    assert again.last_scores == ceiling.last_scores  # deterministic (same rollout seeds)
