"""RB-S, the twin-v3 strong rule baseline (protocol addendum 6)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from udt.agents.rbs_v3 import RBSParams, RuleBasedStrongV3
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


@pytest.mark.phase4
def test_rbs_emits_v3_action_arrays_the_env_accepts(env: UDTMultiAgentEnvV3) -> None:
    env.reset(seed=1, options={"scenario_index": 10})
    acts = RuleBasedStrongV3().act(env)
    for agent in (AGENT_HEALTH, AGENT_POWER, AGENT_TRANSPORT):
        assert env.action_space(agent).contains(acts[agent])
    env.step(acts)  # decoded by the env exactly like MARL actions
    assert env.last_action is not None


@pytest.mark.phase4
def test_destination_prefers_accepting_with_beds_and_never_the_transfer_source(
    env: UDTMultiAgentEnvV3,
) -> None:
    env.reset(seed=2, options={"scenario_index": 0})
    assert env.sim is not None
    g, rn = env.sim.graph, env._road_network
    h1, h2, h3 = env._hospital_ids
    nodes = {
        h: rn.nearest_node(*g.nodes[h]["asset"].geometry["coordinates"][:2])
        for h in env._hospital_ids
    }
    job = {
        "request_id": "T",
        "kind": "transfer",
        "from_hospital_id": h1,
        "location": tuple(g.nodes[h1]["asset"].geometry["coordinates"][:2]),
        "requested_at_tick": 0,
        "urgency": "urgent",
    }
    dest = RuleBasedStrongV3.destination(env, job, nodes)
    assert dest in (h2, h3)
    # make the chosen one divert: the rule moves to the other accepting hospital with beds
    g.nodes[dest]["asset"].attributes["divert"] = True
    other = h3 if dest == h2 else h2
    assert RuleBasedStrongV3.destination(env, job, nodes) == other
    # every alternative diverting: still delivers (fallback c), never to the source
    g.nodes[other]["asset"].attributes["divert"] = True
    assert RuleBasedStrongV3.destination(env, job, nodes) in (h2, h3)


@pytest.mark.phase4
def test_no_new_transfer_request_while_one_is_pending(env: UDTMultiAgentEnvV3) -> None:
    env.reset(seed=3, options={"scenario_index": 0})
    assert env.sim is not None
    g = env.sim.graph
    h1 = env._hospital_ids[0]
    a = g.nodes[h1]["asset"].attributes
    a.update({"beds_total": 10, "beds_occupied": 10, "queue_arrivals": [env.sim.tick] * 8})
    rbs = RuleBasedStrongV3(RBSParams(transfer_queue_ratio=0.1))
    assert rbs.health_action(env)[0] != 0  # queue 8 > 0.1 x 10 beds: request
    g.graph["transfer_requests"] = [{"request_id": "X", "from_hospital_id": h1}]
    assert rbs.health_action(env)[0] == 0  # already pending: no duplicate


@pytest.mark.phase4
def test_divert_uses_hysteresis(env: UDTMultiAgentEnvV3) -> None:
    env.reset(seed=4, options={"scenario_index": 0})
    assert env.sim is not None
    g = env.sim.graph
    h1 = env._hospital_ids[0]
    a = g.nodes[h1]["asset"].attributes
    rbs = RuleBasedStrongV3(RBSParams(divert_queue=10))
    a["queue_arrivals"] = [env.sim.tick] * 10
    assert rbs.health_action(env)[1] == 1  # reach 10 -> divert
    a["divert"] = True
    a["queue_arrivals"] = [env.sim.tick] * 6
    assert rbs.health_action(env)[1] == 1  # still >= 5: keep diverting
    a["queue_arrivals"] = [env.sim.tick] * 4
    assert rbs.health_action(env)[1] == 0  # below half: release


@pytest.mark.phase4
def test_crew_keeps_its_damaged_target(env: UDTMultiAgentEnvV3) -> None:
    env.reset(seed=5, options={"scenario_index": 0})
    assert env.sim is not None
    g = env.sim.graph
    target = env._facility_ids[1]
    g.nodes[target]["asset"].intrinsic_level = 0.3
    c = crew.crew_state(g, env._road_network)
    c.update({"status": "working", "target": target})
    assert RuleBasedStrongV3().power_action(env)[-1] == 0  # keep
    c.update({"status": "idle", "target": None})
    assert RuleBasedStrongV3().power_action(env)[-1] == env._facility_ids.index(target) + 1


@pytest.mark.phase4
def test_params_round_trip(tmp_path: Path) -> None:
    p = RBSParams(transfer_buffer_h=4.0, transfer_queue_ratio=0.1, divert_queue=5, surge_queue=5)
    f = tmp_path / "rbs.yaml"
    import yaml

    f.write_text(yaml.safe_dump({"params": p.as_dict()}))
    assert RBSParams.load(f) == p
    assert np.isclose(p.transfer_buffer_h, 4.0)


@pytest.mark.phase4
def test_frozen_rbs_loads_the_tuned_thresholds() -> None:
    p = RuleBasedStrongV3.frozen().params
    assert p == RBSParams(
        transfer_buffer_h=1.0, transfer_queue_ratio=0.1, divert_queue=5, surge_queue=1
    )
