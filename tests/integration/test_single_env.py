"""Phase 5 acceptance tests (dev doc §13 "Unit — envs" row: "Gymnasium
[...] API compliance checkers pass; obs/action shapes match [...]") and
§14 Phase 5's own env-check criterion.

Runs against the real Kurla `data/processed/` outputs — skipped if they
don't exist, same convention as `tests/integration/test_experiment_a.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from udt.envs.single_env import UDTSingleAgentEnv

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
def env() -> UDTSingleAgentEnv:
    e = UDTSingleAgentEnv(processed_dir=PROCESSED_DIR, n_ticks=30, base_seed=0)
    yield e
    e.close()


@pytest.mark.phase5
def test_gymnasium_api_compliance(env: UDTSingleAgentEnv) -> None:
    """dev doc §13's exact acceptance criterion for this layer."""
    check_env(env, skip_render_check=True)


@pytest.mark.phase5
def test_action_space_shape_matches_real_kurla_graph(env: UDTSingleAgentEnv) -> None:
    # Real Kurla data: 2 hospitals, 1 substation, 2 water (M2 revision —
    # data/manual_facilities.yaml's flagged-synthetic pumping stations,
    # closing the "0 water facilities" gap) -> 2*2=4 ambulances.
    # [transfer(7), shed_tier(4)x1_substation, repair(1+5), dispatch(2)x4_ambulances]
    assert list(env.action_space.nvec) == [7, 4, 6, 2, 2, 2, 2]


@pytest.mark.phase5
def test_observation_water_reserve_slot_reflects_real_buffer_state(
    env: UDTSingleAgentEnv,
) -> None:
    """M2 revision: the per-hospital water-reserve observation slot used
    to be a hardcoded 0.0 placeholder (0 water facilities in the real
    graph). Now that data/manual_facilities.yaml adds 2, this slot must
    actually track the hospital's real water-edge buffer state, not just
    stop crashing — verified against both the helper directly and the
    exact slot in the assembled observation vector."""
    from udt.twin.cascade import EdgeRuntimeState
    from udt.twin.graph import dependency_edges_of

    env.reset(seed=0)
    assert env.sim is not None
    hospital_id = env._hospital_ids[0]
    water_edges = [
        e for e in dependency_edges_of(env.sim.graph, hospital_id) if e.kind == "water"
    ]
    assert water_edges, "real Kurla graph should have a water edge per hospital post-M2"
    edge_id = water_edges[0].edge_id

    env.sim.edge_states[edge_id] = EdgeRuntimeState(capacity_hours=6.0, remaining_hours=6.0)
    assert env._edge_buffer_fraction(hospital_id, "water") == pytest.approx(1.0)

    env.sim.edge_states[edge_id] = EdgeRuntimeState(capacity_hours=6.0, remaining_hours=1.5)
    assert env._edge_buffer_fraction(hospital_id, "water") == pytest.approx(0.25)

    obs = env._build_observation()
    # Slot layout (see _obs_dim/_build_observation): 2 per critical asset,
    # then 1 power-buffer slot per hospital, then a 4-wide
    # (beds, icu, queue, water) block per hospital — water is slot 3 of 4.
    n_critical = len(env._critical_ids)
    n_hospitals = len(env._hospital_ids)
    hospital_index = env._hospital_ids.index(hospital_id)
    water_slot = n_critical * 2 + n_hospitals + hospital_index * 4 + 3
    assert obs[water_slot] == pytest.approx(0.25, abs=1e-5)


@pytest.mark.phase5
def test_reset_returns_observation_matching_declared_space(env: UDTSingleAgentEnv) -> None:
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)


@pytest.mark.phase5
def test_step_runs_and_returns_well_typed_tuple(env: UDTSingleAgentEnv) -> None:
    env.reset(seed=0)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "tick" in info
    assert info["tick"] == 3  # one decision interval advanced


@pytest.mark.phase5
def test_no_op_action_keeps_action_space_well_formed(env: UDTSingleAgentEnv) -> None:
    """A "do nothing" action (all zeros) must decode without error and
    must not crash the twin — the same credibility bar `DoNothingAgent`
    sets for the rule-based harness applies here too."""
    env.reset(seed=0)
    zero_action = np.zeros(env.action_space.shape, dtype=np.int64)
    obs, reward, terminated, truncated, info = env.step(zero_action)
    assert np.all(np.isfinite(obs))
    assert np.isfinite(reward)


@pytest.mark.phase5
def test_episode_runs_to_truncation_within_n_ticks(env: UDTSingleAgentEnv) -> None:
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(20):  # comfortably more than n_ticks=30 / DECISION_INTERVAL_TICKS=3 -> 10 steps
        action = rng.integers(0, env.action_space.nvec, dtype=np.int64)
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            assert truncated  # this scenario shouldn't stabilize in 30 ticks
            return
    pytest.fail("episode never terminated or truncated within 20 steps")


@pytest.mark.phase5
def test_deterministic_with_same_seed(env: UDTSingleAgentEnv) -> None:
    """Same seed, same fixed action sequence -> identical reward
    trajectory (dev doc §3.6's determinism guarantee, extended to the
    env wrapper)."""
    actions = [env.action_space.sample() for _ in range(5)]

    def run() -> list[float]:
        env.reset(seed=42)
        rewards = []
        for a in actions:
            _, r, terminated, truncated, _ = env.step(a)
            rewards.append(r)
            if terminated or truncated:
                break
        return rewards

    assert run() == run()
