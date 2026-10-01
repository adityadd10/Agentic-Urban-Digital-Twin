"""SOP ("current practice") baseline for twin-v3 (protocol 2026-10-02_sop_baseline)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from udt.agents.sop_v3 import SOPBaselineV3
from udt.envs.multi_env_v3 import (
    AGENT_HEALTH,
    AGENT_POWER,
    AGENT_TRANSPORT,
    TRANSFER_OPTIONS,
    UDTMultiAgentEnvV3,
)
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
def test_sop_emits_valid_v3_actions(env: UDTMultiAgentEnvV3) -> None:
    env.reset(seed=1, options={"scenario_index": 10})
    for _ in range(6):
        acts = SOPBaselineV3().act(env)
        for agent in (AGENT_HEALTH, AGENT_POWER, AGENT_TRANSPORT):
            assert env.action_space(agent).contains(acts[agent])
        env.step(acts)


@pytest.mark.phase4
def test_affected_hospital_is_evacuated_and_full_hospital_networks(env: UDTMultiAgentEnvV3) -> None:
    env.reset(seed=2, options={"scenario_index": 0})
    assert env.sim is not None
    g = env.sim.graph
    h1, h2, _ = env._hospital_ids
    g.nodes[h1]["asset"].intrinsic_level = 0.3  # flooded: evacuate
    a2 = g.nodes[h2]["asset"].attributes
    a2.update({"beds_total": 10, "beds_occupied": 10, "queue_arrivals": [env.sim.tick] * 3})
    acts = SOPBaselineV3().health_action(env)
    assert acts[0] == TRANSFER_OPTIONS.index((5, "urgent")) and acts[1] == 1
    assert (
        acts[3] == TRANSFER_OPTIONS.index((2, "urgent")) and acts[4] == 0
    )  # networking, no divert
    assert acts[5] == 1  # surge when full


@pytest.mark.phase4
def test_restoration_order_puts_hospital_substations_first(env: UDTMultiAgentEnvV3) -> None:
    env.reset(seed=3, options={"scenario_index": 0})
    assert env.sim is not None
    g = env.sim.graph
    for f in env._facility_ids:
        g.nodes[f]["asset"].intrinsic_level = 0.5
    c = crew.crew_state(g, env._road_network)
    c.update({"status": "idle", "target": None})
    choice = SOPBaselineV3().power_action(env)[-1]
    picked = env._facility_ids[choice - 1]
    assert g.nodes[picked]["asset"].asset_type.value == "substation"
    assert SOPBaselineV3._priority_class(g, picked) == 0
