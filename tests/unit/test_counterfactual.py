"""Phase 8 acceptance tests: `twin.counterfactual.simulate` (dev doc
§3.7), module M8.

Uses small hand-built graphs (same convention as `test_repair.py`/
`test_patient_transfer.py`), not the real Kurla data — `simulate()`'s
own logic is exercised directly, `risk/engine.py`'s integration tests
cover the real-data path.
"""

from __future__ import annotations

import pytest

from udt.common.models import (
    AgentAction,
    Asset,
    AssetType,
    DependencyEdge,
    DependencyGraph,
    Incident,
)
from udt.twin.counterfactual import simulate
from udt.twin.simulator import Simulator

POINT = {"type": "Point", "coordinates": [72.88, 19.07]}


def _hospital_graph(beds_total: int = 100, beds_occupied: int = 0) -> DependencyGraph:
    hospital = Asset(
        asset_id="H1",
        asset_type=AssetType.HOSPITAL,
        geometry=POINT,
        attributes={"beds_total": beds_total, "beds_occupied": beds_occupied},
    )
    return DependencyGraph(assets=[hospital], edges=[])


def _cascade_graph() -> DependencyGraph:
    """S1 -> H1 via a power edge with zero buffer (no cushioning) and
    floor=0.0 — degrading S1 immediately drags H1's functional_level
    down too, a real cascade the fixed-point resolution produces, not a
    hand-set value."""
    substation = Asset(
        asset_id="S1",
        asset_type=AssetType.SUBSTATION,
        geometry=POINT,
        attributes={"capacity_mw": 10.0, "load_mw": 5.0, "shed_tier": 0},
    )
    hospital = Asset(
        asset_id="H1",
        asset_type=AssetType.HOSPITAL,
        geometry=POINT,
        attributes={"beds_total": 100, "beds_occupied": 0},
    )
    edge = DependencyEdge(
        edge_id="E1",
        supplier="S1",
        consumer="H1",
        kind="power",
        demand=1.0,
        criticality=1.0,
        buffer_hours=0.0,
        floor=0.0,
    )
    return DependencyGraph(assets=[substation, hospital], edges=[edge])


INCIDENT = Incident(
    incident_id="cf_test",
    type="flood",
    location=POINT,
    onset_tick=0,
    severity=1.0,
    directly_affected_assets=[],
)


def _no_degradation(tick: int, graph: object) -> dict[str, float]:
    return {}


@pytest.mark.phase8
def test_simulate_returns_exactly_n_rollouts() -> None:
    sim = Simulator(_hospital_graph())
    result = simulate(
        sim, AgentAction(), degradation_fn=_no_degradation, n_rollouts=5, horizon_ticks=3
    )
    assert len(result.rollouts) == 5


@pytest.mark.phase8
def test_simulate_does_not_mutate_the_original_simulator() -> None:
    sim = Simulator(_hospital_graph())
    tick_before = sim.tick
    level_before = sim.asset("H1").functional_level
    simulate(sim, AgentAction(), degradation_fn=_no_degradation, n_rollouts=3, horizon_ticks=5)
    assert sim.tick == tick_before
    assert sim.asset("H1").functional_level == pytest.approx(level_before)


@pytest.mark.phase8
def test_simulate_is_deterministic_given_the_same_base_seed() -> None:
    """dev doc §3.6's determinism guarantee, extended to `simulate()`:
    same starting `Simulator` state + action + base_seed -> identical
    result, exercised via the hospital demand model's own rng use."""

    def run() -> object:
        sim = Simulator(_hospital_graph())
        return simulate(
            sim,
            AgentAction(),
            degradation_fn=_no_degradation,
            n_rollouts=4,
            horizon_ticks=10,
            base_seed=7,
        )

    assert run() == run()


@pytest.mark.phase8
def test_p_failure_is_zero_when_hospital_stays_healthy() -> None:
    sim = Simulator(_hospital_graph())
    result = simulate(
        sim, AgentAction(), degradation_fn=_no_degradation, n_rollouts=5, horizon_ticks=5
    )
    assert result.p_failure == 0.0
    assert all(r.min_hospital_functional_level >= 0.3 for r in result.rollouts)


@pytest.mark.phase8
def test_p_failure_is_one_when_hospital_already_below_threshold() -> None:
    hospital = Asset(
        asset_id="H1",
        asset_type=AssetType.HOSPITAL,
        geometry=POINT,
        intrinsic_level=0.1,
        functional_level=0.1,
        attributes={"beds_total": 100, "beds_occupied": 0},
    )
    sim = Simulator(DependencyGraph(assets=[hospital], edges=[]))
    result = simulate(
        sim, AgentAction(), degradation_fn=_no_degradation, n_rollouts=5, horizon_ticks=5
    )
    assert result.p_failure == 1.0


@pytest.mark.phase8
def test_repair_action_is_applied_once_not_every_tick_of_the_rollout() -> None:
    """The `action` passed to `simulate()` is enacted on the rollout's
    first tick only (dev doc §3.7: "applies the plan/actions"), same
    one-shot convention `experiments/runner.py`/`envs/*.py` already use
    — not re-applied every tick, which would over-repair. With no
    degradation active, the substation's level only ever *rises* (once,
    at tick 0) then holds flat — so the rollout's *minimum* level over
    the whole horizon is exactly "one tick of repair", not "one tick per
    horizon tick"."""
    from udt.twin.simulator import DEFAULT_REPAIR_RATE_PER_TICK

    substation = Asset(
        asset_id="S1",
        asset_type=AssetType.SUBSTATION,
        geometry=POINT,
        intrinsic_level=0.5,
        functional_level=0.5,
        attributes={"capacity_mw": 10.0, "load_mw": 1.0, "shed_tier": 0},
    )
    sim = Simulator(DependencyGraph(assets=[substation], edges=[]))
    result = simulate(
        sim,
        AgentAction(repair_target="S1"),
        degradation_fn=_no_degradation,
        n_rollouts=1,
        horizon_ticks=3,
    )
    expected_after_one_repair_tick = 0.5 + DEFAULT_REPAIR_RATE_PER_TICK
    assert result.rollouts[0].min_critical_functional_level == pytest.approx(
        expected_after_one_repair_tick
    )


@pytest.mark.phase8
def test_new_cascading_failures_only_counts_this_rollouts_delta() -> None:
    """If the original `sim` already had cascades *before* `simulate()`
    is called, a rollout that doesn't trigger any new ones must report
    `new_cascading_failures == 0`, not the pre-existing total."""
    sim = Simulator(_cascade_graph())
    # Cascade H1 once, for real, before ever calling simulate() - S1
    # degrades hard enough that H1's functional_level drops under 0.5
    # too (both count as "cascaded" here since this test doesn't pass
    # `directly_affected_assets` to `step()` - same convention every
    # other harness in this codebase already uses).
    sim.apply_degradation("S1", 0.95)
    sim.step()
    assert sim.asset("H1").functional_level < 0.5
    assert (
        len(sim._ever_cascaded) == 2
    )  # confirms the test's own premise: some pre-existing history

    result = simulate(
        sim, AgentAction(), degradation_fn=_no_degradation, n_rollouts=3, horizon_ticks=3
    )
    for rollout in result.rollouts:
        assert rollout.new_cascading_failures == 0


@pytest.mark.phase8
def test_new_cascading_failures_counts_a_cascade_that_happens_during_the_rollout() -> None:
    sim = Simulator(_cascade_graph())
    assert sim.asset("H1").functional_level >= 0.5  # healthy before the rollout

    def degrade_substation_once(tick: int, graph: object) -> dict[str, float]:
        return {"S1": 0.95} if tick == 0 else {}

    result = simulate(
        sim, AgentAction(), degradation_fn=degrade_substation_once, n_rollouts=1, horizon_ticks=3
    )
    # Both S1 (directly degraded) and H1 (drags down as a real cascade)
    # count here - this test doesn't pass `directly_affected_assets`,
    # same convention every other harness in this codebase already uses.
    assert result.rollouts[0].new_cascading_failures == 2


@pytest.mark.phase8
def test_simulate_respects_incident_driven_power_surge() -> None:
    """`update_substation_load` runs every rollout tick when `incident`
    is passed — a substation loaded to 90% of capacity isn't overloaded
    on its own (`OVERLOAD_THRESHOLD=0.95`), but a flood's load surge
    (`LOAD_SURGE_FACTOR=0.5` at full severity) pushes it over that line,
    so 5 ticks of `apply_overload_damage` should measurably lower the
    substation's functional level — but only in the rollout that was
    told about the incident."""

    def substation() -> Asset:
        return Asset(
            asset_id="S1",
            asset_type=AssetType.SUBSTATION,
            geometry=POINT,
            attributes={"capacity_mw": 10.0, "load_mw": 9.0, "shed_tier": 0},
        )

    sim_with_incident = Simulator(DependencyGraph(assets=[substation()], edges=[]))
    sim_without_incident = Simulator(DependencyGraph(assets=[substation()], edges=[]))

    with_incident = simulate(
        sim_with_incident,
        AgentAction(),
        degradation_fn=_no_degradation,
        incident=INCIDENT,
        n_rollouts=1,
        horizon_ticks=5,
    )
    without_incident = simulate(
        sim_without_incident,
        AgentAction(),
        degradation_fn=_no_degradation,
        incident=None,
        n_rollouts=1,
        horizon_ticks=5,
    )
    assert without_incident.rollouts[0].min_critical_functional_level == pytest.approx(1.0)
    assert with_incident.rollouts[0].min_critical_functional_level < 1.0
    # Both original simulators stay untouched either way.
    assert sim_with_incident.asset("S1").intrinsic_level == pytest.approx(1.0)
    assert sim_without_incident.asset("S1").intrinsic_level == pytest.approx(1.0)
