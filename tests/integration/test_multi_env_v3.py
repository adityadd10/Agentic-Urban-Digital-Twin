"""Twin-v3 multi-agent env (dev doc §3.9, "Twin-v3 environment specification")."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pettingzoo.test import parallel_api_test

from udt.common.models import AgentAction
from udt.envs import multi_env_v3 as v3mod
from udt.envs.multi_env_v3 import (
    AGENT_HEALTH,
    AGENT_POWER,
    AGENT_TRANSPORT,
    N_JOB_SLOTS,
    UDTMultiAgentEnvV3,
    job_priority_order,
)
from udt.twin import ambulances, crew

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data/processed"
REWARD = REPO_ROOT / "configs/reward.yaml"  # placeholder until v3 normalisers exist


@pytest.fixture(scope="module")
def env() -> Iterator[UDTMultiAgentEnvV3]:
    saved = (ambulances.COUNT_UNCOLLECTED_CASUALTY_DEATHS, crew.REPAIR_CREW_TRAVEL)
    e = UDTMultiAgentEnvV3(processed_dir=PROCESSED, reward_config=REWARD)
    yield e
    e.close()
    # restore twin-v2 settings for the rest of the test session
    ambulances.COUNT_UNCOLLECTED_CASUALTY_DEATHS, crew.REPAIR_CREW_TRAVEL = saved


def _defer_all(env: UDTMultiAgentEnvV3) -> dict[str, Any]:
    return {a: np.zeros(len(env.action_space(a).nvec), dtype=np.int64) for a in env.agents}


@pytest.mark.phase6
def test_pettingzoo_api_compliance(env: UDTMultiAgentEnvV3) -> None:
    parallel_api_test(env, num_cycles=10)


@pytest.mark.phase6
def test_network_and_spaces(env: UDTMultiAgentEnvV3) -> None:
    env.reset(seed=1)
    assert len(env._hospital_ids) == 3 and len(env._substation_ids) == 3
    assert env._n_ambulances == 6
    assert list(env.action_space(AGENT_HEALTH).nvec) == [4, 2, 2] * 3
    assert list(env.action_space(AGENT_POWER).nvec) == [4, 4, 4, 10]
    assert list(env.action_space(AGENT_TRANSPORT).nvec) == [4] * N_JOB_SLOTS


@pytest.mark.phase6
def test_v3_settings_on_and_v2_env_refuses_to_start(env: UDTMultiAgentEnvV3) -> None:
    from udt.envs.multi_env import UDTMultiAgentEnv

    assert ambulances.COUNT_UNCOLLECTED_CASUALTY_DEATHS and crew.REPAIR_CREW_TRAVEL
    with pytest.raises(RuntimeError, match="twin-v3"):
        UDTMultiAgentEnv(processed_dir=PROCESSED, scenario_split=None)


@pytest.mark.phase6
def test_actions_decode_into_one_checked_agent_action(
    env: UDTMultiAgentEnvV3, monkeypatch: pytest.MonkeyPatch
) -> None:
    env.reset(seed=2)
    seen: list[AgentAction] = []
    real_check = v3mod.check

    def spy(graph: Any, action: AgentAction, **kw: Any) -> Any:
        seen.append(action)
        return real_check(graph, action, **kw)

    monkeypatch.setattr(v3mod, "check", spy)
    acts = _defer_all(env)
    acts[AGENT_HEALTH] = np.array([3, 1, 1, 0, 0, 0, 0, 0, 0])  # H1: 5 urgent, divert, surge
    acts[AGENT_POWER] = np.array([2, 0, 0, 2])  # S0 tier 2; crew -> 2nd facility
    env.step(acts)
    assert len(seen) == 1
    a = env.last_action
    assert a is not None
    h1 = env._hospital_ids[0]
    assert a.transfer_requests == [(h1, 5, "urgent")]
    assert a.divert is not None and a.divert[h1] is True
    assert a.surge is not None and a.surge[h1] is True
    assert a.shed_tier == {
        env._substation_ids[0]: 2,
        env._substation_ids[1]: 0,
        env._substation_ids[2]: 0,
    }
    assert a.repair_target == env._facility_ids[1]


@pytest.mark.phase6
def test_job_slots_follow_slack_then_urgency_then_age(env: UDTMultiAgentEnvV3) -> None:
    env.reset(seed=3)
    assert env.sim is not None
    g, tick = env.sim.graph, env.sim.tick
    g.graph["pending_requests"] = [
        {"request_id": "OLD", "location": (72.88, 19.07), "requested_at_tick": tick - 30},
        {"request_id": "NEW", "location": (72.88, 19.07), "requested_at_tick": tick},
    ]
    h1 = env._hospital_ids[0]
    g.nodes[h1]["asset"].attributes["queue_arrivals"] = []  # transfer with no queue: no deadline
    g.graph["transfer_requests"] = [
        {
            "request_id": "TR_R",
            "kind": "transfer",
            "from_hospital_id": h1,
            "location": (72.867, 19.054),
            "requested_at_tick": tick - 40,
            "urgency": "routine",
        },
        {
            "request_id": "TR_U",
            "kind": "transfer",
            "from_hospital_id": h1,
            "location": (72.867, 19.054),
            "requested_at_tick": tick,
            "urgency": "urgent",
        },
    ]
    order = [j["request_id"] for j in job_priority_order(g, tick, env.sim.dt_hours)]
    assert order == ["OLD", "NEW", "TR_U", "TR_R"]


@pytest.mark.phase6
def test_transport_never_uses_more_ambulances_than_idle_and_defers_the_rest(
    env: UDTMultiAgentEnvV3,
) -> None:
    env.reset(seed=4)
    assert env.sim is not None
    g, tick = env.sim.graph, env.sim.tick
    h = env._hospital_ids
    g.graph["pending_requests"] = [
        {"request_id": f"C{i}", "location": (72.885, 19.075), "requested_at_tick": tick}
        for i in range(8)
    ]
    env._observations()  # refresh slots
    assignment, destination, log = env._decode_transport(np.array([2] * N_JOB_SLOTS))
    assert len(assignment) == len(set(assignment.values())) == 6  # 6 ambulances, 6 slots
    assert set(destination.values()) == {h[1]}
    # all 6 busy now: a second decision with every slot selected defers everything
    for amb in assignment:
        g.nodes[amb]["asset"].attributes["status"] = "enroute"
    _, _, log2 = env._decode_transport(np.array([1] * N_JOB_SLOTS))
    assert log2["served"] == [] and all(
        d["reason"] == "no idle ambulance" for d in log2["deferred"]
    )


@pytest.mark.phase6
def test_transfer_back_to_its_own_source_is_deferred_not_rerouted(env: UDTMultiAgentEnvV3) -> None:
    env.reset(seed=5)
    assert env.sim is not None
    g, tick = env.sim.graph, env.sim.tick
    h1 = env._hospital_ids[0]
    g.graph["pending_requests"] = []
    g.graph["transfer_requests"] = [
        {
            "request_id": "TR",
            "kind": "transfer",
            "from_hospital_id": h1,
            "location": (72.867, 19.054),
            "requested_at_tick": tick,
            "urgency": "urgent",
        }
    ]
    env._observations()
    assignment, _, log = env._decode_transport(np.array([1] + [0] * (N_JOB_SLOTS - 1)))
    assert assignment == {}
    assert log["deferred"] == [{"job": "TR", "reason": "transfer to its own source"}]
    assert g.graph["transfer_requests"][0]["request_id"] == "TR"  # still waiting


@pytest.mark.phase6
def test_shedding_head_can_be_switched_off() -> None:
    saved = (ambulances.COUNT_UNCOLLECTED_CASUALTY_DEATHS, crew.REPAIR_CREW_TRAVEL)
    e = UDTMultiAgentEnvV3(processed_dir=PROCESSED, reward_config=REWARD, shedding_enabled=False)
    try:
        assert list(e.action_space(AGENT_POWER).nvec) == [10]
        e.reset(seed=6)
        acts = _defer_all(e)
        e.step(acts)
        assert e.last_action is not None and e.last_action.shed_tier is None
    finally:
        e.close()
        ambulances.COUNT_UNCOLLECTED_CASUALTY_DEATHS, crew.REPAIR_CREW_TRAVEL = saved


@pytest.mark.phase6
def test_resume_state_continues_bit_identically(env: UDTMultiAgentEnvV3) -> None:
    rng = np.random.default_rng(0)
    env.reset(seed=7)
    for _ in range(5):
        env.step({a: rng.integers(0, env.action_space(a).nvec) for a in env.agents})
    blob = env.resume_state()
    later = [
        {a: rng.integers(0, env.action_space(a).nvec) for a in env.possible_agents}
        for _ in range(5)
    ]
    straight = [env.step(acts)[0] for acts in later]
    env.load_resume_state(blob)
    resumed = [env.step(acts)[0] for acts in later]
    for s, r in zip(straight, resumed, strict=True):
        for a in s:
            assert np.array_equal(s[a], r[a])


@pytest.mark.phase6
def test_episodes_run_the_full_horizon_like_the_rule_harness(env: UDTMultiAgentEnvV3) -> None:
    """Seed 7 stabilised after 4 decisions under v2's early stop; v3 runs all 288 ticks."""
    env.reset(seed=7)
    done, n = False, 0
    while not done:
        _, _, term, trunc, _ = env.step(_defer_all(env) if env.agents else {})
        n += 1
        done = any(term.values()) or any(trunc.values())
    assert n == 96 and len(env.episode_trace) == 288
    assert not any(term.values()) and all(trunc.values())
