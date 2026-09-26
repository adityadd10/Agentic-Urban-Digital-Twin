"""Phase 6 acceptance tests (dev doc §13 "Unit — envs" row extended to
PettingZoo: "API compliance checkers pass; obs/action shapes match
[...]"), modules M6a + M6b (goal-conditioning).

Runs against the real Kurla `data/processed/` outputs — skipped if they
don't exist, same convention as `tests/integration/test_single_env.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pettingzoo.test import parallel_api_test

from udt.common.models import AssetType, ConstraintReport, ConstraintViolation
from udt.envs.multi_env import (
    AGENT_HEALTH,
    AGENT_POWER,
    AGENT_TRANSPORT,
    GOAL_DIM,
    GOAL_HIGH,
    GOAL_LOW,
    UDTMultiAgentEnv,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
REQUIRED_FILES = (
    "dependency_graph.json",
    "flood_susceptibility.tif",
    "ward_boundary.geojson",
    "roads_full.graphml",
)

pytestmark = pytest.mark.skipif(
    not all((PROCESSED_DIR / f).exists() for f in REQUIRED_FILES),
    reason="needs data/processed/{dependency_graph.json,flood_susceptibility.tif,"
    "ward_boundary.geojson,roads_full.graphml} (run the 01-06 pipeline scripts)",
)


@pytest.fixture
def env() -> UDTMultiAgentEnv:
    e = UDTMultiAgentEnv(processed_dir=PROCESSED_DIR, n_ticks=30, base_seed=0)
    yield e
    e.close()


def _random_actions(env: UDTMultiAgentEnv) -> dict[str, np.ndarray]:
    return {agent: env.action_space(agent).sample() for agent in env.agents}


@pytest.mark.phase6
def test_pettingzoo_api_compliance(env: UDTMultiAgentEnv) -> None:
    """dev doc §13's acceptance criterion, PettingZoo's own checker."""
    parallel_api_test(env, num_cycles=20)


@pytest.mark.phase6
def test_three_agents_exist(env: UDTMultiAgentEnv) -> None:
    assert set(env.possible_agents) == {AGENT_HEALTH, AGENT_POWER, AGENT_TRANSPORT}


@pytest.mark.phase6
def test_action_spaces_match_real_kurla_graph(env: UDTMultiAgentEnv) -> None:
    # 2 hospitals -> 2 ordered pairs x 3 tiers + "none" = 7
    assert env.action_space(AGENT_HEALTH).n == 7
    # 1 substation (shed 0-3) + repair-among-3 (1 substation + 2 water,
    # M2 revision — data/manual_facilities.yaml's flagged-synthetic water
    # facilities) (+"none") = MultiDiscrete([4, 4])
    assert list(env.action_space(AGENT_POWER).nvec) == [4, 4]
    # 4 ambulances, dispatch-or-not each
    assert list(env.action_space(AGENT_TRANSPORT).nvec) == [2, 2, 2, 2]


@pytest.mark.phase6
def test_reset_returns_observation_per_agent(env: UDTMultiAgentEnv) -> None:
    obs, infos = env.reset(seed=0)
    assert set(obs.keys()) == {AGENT_HEALTH, AGENT_POWER, AGENT_TRANSPORT}
    for agent in env.possible_agents:
        assert obs[agent].shape == env.observation_space(agent).shape
        assert env.observation_space(agent).contains(obs[agent])


@pytest.mark.phase6
def test_reward_is_identical_across_agents(env: UDTMultiAgentEnv) -> None:
    """dev doc §5.4: 'shared team reward; identical for single- and
    multi-agent'."""
    env.reset(seed=0)
    _, rewards, _, _, _ = env.step(_random_actions(env))
    values = set(rewards.values())
    assert len(values) == 1, f"expected one shared reward value, got {rewards}"


@pytest.mark.phase6
def test_step_advances_one_decision_interval(env: UDTMultiAgentEnv) -> None:
    env.reset(seed=0)
    _, _, _, _, infos = env.step(_random_actions(env))
    assert infos[AGENT_HEALTH]["tick"] == 3


@pytest.mark.phase6
def test_episode_runs_to_truncation_within_n_ticks(env: UDTMultiAgentEnv) -> None:
    env.reset(seed=0)
    for _ in range(20):  # n_ticks=30 / DECISION_INTERVAL_TICKS=3 -> 10 steps
        _, _, terminations, truncations, _ = env.step(_random_actions(env))
        if any(terminations.values()) or any(truncations.values()):
            assert any(truncations.values())  # shouldn't stabilize in 30 ticks
            assert env.agents == []  # PettingZoo convention on episode end
            return
    pytest.fail("episode never terminated or truncated within 20 steps")


@pytest.mark.phase6
def test_deterministic_with_same_seed(env: UDTMultiAgentEnv) -> None:
    """dev doc §3.6's determinism guarantee, extended to the multi-agent
    wrapper."""
    env.reset(seed=42)
    actions = [_random_actions(env) for _ in range(5)]

    def run() -> list[dict[str, float]]:
        env.reset(seed=42)
        rewards = []
        for a in actions:
            _, r, terminations, truncations, _ = env.step(a)
            rewards.append(r)
            if any(terminations.values()) or any(truncations.values()):
                break
        return rewards

    assert run() == run()


@pytest.mark.phase6
def test_goal_sampled_within_dev_doc_bounds(env: UDTMultiAgentEnv) -> None:
    """dev doc §5.4: g bounded [0.5, 2.0]."""
    for seed in (0, 1, 2):
        env.reset(seed=seed)
        assert env.goal.shape == (GOAL_DIM,)
        assert np.all(env.goal >= GOAL_LOW)
        assert np.all(env.goal <= GOAL_HIGH)


@pytest.mark.phase6
def test_goal_differs_across_episodes(env: UDTMultiAgentEnv) -> None:
    """Sampled fresh each episode (dev doc §5.5) - not the same vector
    twice in a row for different seeds."""
    env.reset(seed=0)
    goal_a = env.goal.copy()
    env.reset(seed=1)
    goal_b = env.goal.copy()
    assert not np.allclose(goal_a, goal_b)


@pytest.mark.phase6
def test_goal_appears_in_every_agents_observation(env: UDTMultiAgentEnv) -> None:
    """dev doc §5.2: g is part of the *common* block every agent
    observes — it sits at the end of that block, i.e. right before each
    agent's own agent-specific additions (not necessarily the very end
    of the full per-agent vector)."""
    obs, _ = env.reset(seed=0)
    common_dim = env._common_obs_dim()
    for agent in env.possible_agents:
        goal_slice = obs[agent][common_dim - GOAL_DIM : common_dim]
        assert np.allclose(goal_slice, env.goal, atol=1e-6)


@pytest.mark.phase6
def test_reward_scales_with_goal_health_weight(env: UDTMultiAgentEnv) -> None:
    """`_tick_reward` must scale the unmet-patient-hours term by
    `goal[0]` (g_health) exactly, per dev doc §5.4's formula - checked
    directly against a controlled snapshot, not inferred from a full
    stochastic rollout."""
    env.reset(seed=0)
    assert env.sim is not None
    snapshot = env.sim.step()  # one real tick, whatever it produces

    # `_tick_reward` mutates `_prev_cascading_count`/`_prev_patient_deaths`/
    # `_stable_streak` as a side effect - capture them beforehand so the
    # *same* snapshot can be re-scored under a different goal without the
    # second call's delta terms being silently zeroed by the first call's
    # bookkeeping update.
    prev_cascading = env._prev_cascading_count
    prev_deaths = env._prev_patient_deaths
    prev_streak = env._stable_streak

    env.goal = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    reward_at_1x = env._tick_reward(snapshot)

    env._prev_cascading_count = prev_cascading
    env._prev_patient_deaths = prev_deaths
    env._stable_streak = prev_streak
    env.goal = np.array([2.0, 1.0, 1.0, 1.0], dtype=np.float32)
    reward_at_2x = env._tick_reward(snapshot)

    # The g_health delta (1.0 -> 2.0) must account for exactly the health
    # penalty term's own magnitude, recomputed independently here.
    assert env.sim is not None
    queued_this_tick = sum(
        int(a.attributes.get("patient_queue", 0))
        for a in snapshot.assets
        if a.asset_type == AssetType.HOSPITAL
    )
    unmet_patient_hours = queued_this_tick * env.sim.dt_hours
    deaths_delta = snapshot.patient_deaths_cumulative - prev_deaths
    expected_diff = unmet_patient_hours + 10.0 * deaths_delta
    assert (reward_at_1x - reward_at_2x) == pytest.approx(expected_diff, abs=1e-6)


# ---------------------------------------------------------------------------
# M7 slice 2: `action_mask()` + `constrained_reward` (dev doc §5.5's
# Experiment G).
# ---------------------------------------------------------------------------
@pytest.mark.phase7
def test_action_mask_power_is_always_all_true(env: UDTMultiAgentEnv) -> None:
    """`power_balance` is a disclosed no-op (`constraints/engine.py`) —
    nothing masks `power`'s action space."""
    env.reset(seed=0)
    mask = env.action_mask(AGENT_POWER)
    nvec = [int(n) for n in env.action_space(AGENT_POWER).nvec]  # type: ignore[union-attr]
    assert [len(head) for head in mask] == nvec
    assert all(bool(head.all()) for head in mask)


@pytest.mark.phase7
def test_action_mask_health_no_transfer_choice_always_feasible(env: UDTMultiAgentEnv) -> None:
    env.reset(seed=0)
    mask = env.action_mask(AGENT_HEALTH)
    assert len(mask) == 1
    assert bool(mask[0][0])  # choice 0 = "no transfer"


@pytest.mark.phase7
def test_action_mask_health_marks_oversized_transfer_infeasible(env: UDTMultiAgentEnv) -> None:
    """Force one hospital completely full — any transfer choice decoding
    to a transfer *into* it must be masked `False`; choices decoding
    into the other (still has free beds) hospital must stay `True`."""
    env.reset(seed=0)
    assert env.sim is not None
    h_full, h_free = env._hospital_ids[0], env._hospital_ids[1]
    env.sim.graph.nodes[h_full]["asset"].attributes["beds_total"] = 10
    env.sim.graph.nodes[h_full]["asset"].attributes["beds_occupied"] = 10
    env.sim.graph.nodes[h_free]["asset"].attributes["beds_total"] = 200
    env.sim.graph.nodes[h_free]["asset"].attributes["beds_occupied"] = 0

    mask = env.action_mask(AGENT_HEALTH)[0]
    n = len(mask)
    checked_full, checked_free = False, False
    for choice in range(1, n):
        candidate = env._decode_health_action(np.array([choice]))
        if candidate is None:
            continue
        _from_id, to_id, _count = candidate
        if to_id == h_full:
            assert not mask[choice], f"choice {choice} (-> full hospital) should be infeasible"
            checked_full = True
        elif to_id == h_free:
            assert mask[choice], f"choice {choice} (-> hospital with free beds) should be feasible"
            checked_free = True
    assert checked_full and checked_free, "test's own premise needs both directions represented"


@pytest.mark.phase7
def test_action_mask_transport_marks_unsafe_dispatch_infeasible(
    env: UDTMultiAgentEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same `RoadNetwork.shortest_path_max_depth` unit already tests the
    depth math itself (`test_road_network.py`) — this test only checks
    that `action_mask` wires that result into a `False` for the specific
    ambulance head whose dispatch would use it, and leaves every other
    head (a non-idle ambulance, or one with no route problem) alone."""
    env.reset(seed=0)
    assert env.sim is not None
    env.sim.graph.graph["pending_requests"] = [
        {"request_id": "REQ_TEST", "location": (72.88, 19.07), "requested_at_tick": 0}
    ]
    monkeypatch.setattr(env._road_network, "shortest_path_max_depth", lambda *a, **k: 10.0)

    ambulance_ids = [
        asset_id
        for asset_id in env.sim.graph.nodes
        if env.sim.graph.nodes[asset_id]["asset"].asset_type == AssetType.AMBULANCE
    ]
    idle_index = next(
        i
        for i, aid in enumerate(ambulance_ids)
        if env.sim.graph.nodes[aid]["asset"].attributes.get("status") == "idle"
    )

    mask = env.action_mask(AGENT_TRANSPORT)
    assert mask[idle_index][0]  # "don't dispatch" always feasible
    assert not mask[idle_index][1]  # dispatch -> unsafe route (depth 10.0 m >= 0.4 m)


@pytest.mark.phase7
def test_action_mask_transport_all_true_without_pending_requests(env: UDTMultiAgentEnv) -> None:
    env.reset(seed=0)
    assert env.sim is not None
    env.sim.graph.graph["pending_requests"] = []
    mask = env.action_mask(AGENT_TRANSPORT)
    assert all(bool(head.all()) for head in mask)


def _fake_check_with_n_violations(n: int):  # type: ignore[no-untyped-def]
    def fake_check(graph, action, *, road_network=None):  # type: ignore[no-untyped-def]
        violations = [
            ConstraintViolation(
                rule_id="test_rule", scope="health", on_violation="clip", detail="x"
            )
            for _ in range(n)
        ]
        return ConstraintReport(passed=n == 0, violations=violations, repaired_action=action)

    return fake_check


@pytest.mark.phase7
def test_constrained_reward_flag_drops_violation_penalty(monkeypatch: pytest.MonkeyPatch) -> None:
    """dev doc §5.4's own footnote: the fixed `-20.0 x safety_violations`
    reward term is for the UNCONSTRAINED variant only — with
    `constrained_reward=True` it must be absent, even though the
    attempted-violations *metric* (`info["safety_violations_attempted"]`)
    still counts them either way."""
    monkeypatch.setattr("udt.envs.multi_env.check", _fake_check_with_n_violations(2))

    unconstrained = UDTMultiAgentEnv(processed_dir=PROCESSED_DIR, n_ticks=30, base_seed=0)
    unconstrained.reset(seed=0)
    actions = {a: unconstrained.action_space(a).sample() for a in unconstrained.agents}
    _, rewards_u, _, _, infos_u = unconstrained.step(actions)
    unconstrained.close()

    constrained = UDTMultiAgentEnv(
        processed_dir=PROCESSED_DIR, n_ticks=30, base_seed=0, constrained_reward=True
    )
    constrained.reset(seed=0)
    _, rewards_c, _, _, infos_c = constrained.step(actions)
    constrained.close()

    reward_u = next(iter(rewards_u.values()))
    reward_c = next(iter(rewards_c.values()))
    assert reward_c - reward_u == pytest.approx(20.0 * 2, abs=1e-6)
    assert next(iter(infos_u.values()))["safety_violations_attempted"] == 2
    assert next(iter(infos_c.values()))["safety_violations_attempted"] == 2
