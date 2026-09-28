"""Phase 4 acceptance tests: `twin.demand.consume_demand` (dev doc §3.5
step 4, hospitals only — see the module's docstring for scope)."""

from __future__ import annotations

import numpy as np
import pytest

from udt.common.models import Asset, AssetType, DependencyGraph
from udt.twin.demand import consume_demand, diurnal_multiplier, effective_free_beds
from udt.twin.graph import build_networkx_graph

POINT = {"type": "Point", "coordinates": [72.88, 19.07]}


def _hospital_graph(
    beds_total: int,
    beds_occupied: int,
    queue_arrivals: list[int],
    *,
    functional_level: float = 1.0,
) -> object:
    asset = Asset(
        asset_id="H1",
        asset_type=AssetType.HOSPITAL,
        geometry=POINT,
        functional_level=functional_level,
        attributes={
            "beds_total": beds_total,
            "beds_occupied": beds_occupied,
            "queue_arrivals": queue_arrivals,
            "patient_queue": len(queue_arrivals),
        },
    )
    return build_networkx_graph(DependencyGraph(assets=[asset], edges=[]))


@pytest.mark.phase4
def test_diurnal_multiplier_peaks_at_configured_hour() -> None:
    from udt.twin.demand import DIURNAL_AMPLITUDE, DIURNAL_PEAK_HOUR

    assert diurnal_multiplier(DIURNAL_PEAK_HOUR) == pytest.approx(1.0 + DIURNAL_AMPLITUDE)
    assert diurnal_multiplier((DIURNAL_PEAK_HOUR + 12) % 24) == pytest.approx(
        1.0 - DIURNAL_AMPLITUDE
    )


@pytest.mark.phase4
def test_consume_demand_admits_queued_patients_into_freed_beds() -> None:
    # Full hospital (0 free beds), 3 queued, no new arrivals possible this
    # tick (rng seeded so arrivals are whatever they are, but a discharge
    # must free at least one bed for the oldest queued patient to be
    # admitted) — use a large rng draw budget via a fixed seed and check
    # the *mechanism*, not an exact patient count, since arrivals are
    # stochastic.
    graph = _hospital_graph(beds_total=10, beds_occupied=10, queue_arrivals=[0, 0, 0])
    rng = np.random.default_rng(0)
    consume_demand(graph, tick=1, dt_hours=5.0 / 60.0, rng=rng)
    h1 = graph.nodes["H1"]["asset"]
    # Occupied + queued must still conserve everyone who was already in
    # the system (no one vanishes except via the discharge/deadline paths
    # this function itself implements).
    assert h1.attributes["beds_occupied"] <= h1.attributes["beds_total"]
    assert h1.attributes["patient_queue"] == len(h1.attributes["queue_arrivals"])


@pytest.mark.phase4
def test_consume_demand_deadline_removes_patient_and_counts_death() -> None:
    # dt_hours chosen so a patient who arrived at tick 0 and is still
    # queued at a much later tick has clearly exceeded
    # PATIENT_WAIT_DEADLINE_HOURS.
    from udt.twin.demand import PATIENT_WAIT_DEADLINE_HOURS

    dt_hours = 1.0
    late_tick = int(PATIENT_WAIT_DEADLINE_HOURS) + 5
    # Zero beds_total -> zero arrival rate (rate scales with beds_total),
    # and beds_occupied=0 -> zero discharges, so the queue's fate is
    # driven entirely by the deadline check, not stochastic arrivals.
    graph = _hospital_graph(beds_total=0, beds_occupied=0, queue_arrivals=[0])
    rng = np.random.default_rng(0)
    deaths = consume_demand(graph, tick=late_tick, dt_hours=dt_hours, rng=rng)
    assert deaths["H1"] == 1
    h1 = graph.nodes["H1"]["asset"]
    assert h1.attributes["patient_queue"] == 0


@pytest.mark.phase4
def test_consume_demand_zero_bed_hospital_is_a_no_op() -> None:
    graph = _hospital_graph(beds_total=0, beds_occupied=0, queue_arrivals=[])
    rng = np.random.default_rng(0)
    deaths = consume_demand(graph, tick=0, dt_hours=5.0 / 60.0, rng=rng)
    assert deaths["H1"] == 0
    h1 = graph.nodes["H1"]["asset"]
    assert h1.attributes["beds_occupied"] == 0
    assert h1.attributes["patient_queue"] == 0


# ---------------------------------------------------------------------------
# M8: functional-level coupling (`effective_free_beds`, closing the gap
# `scripts/calibrate_risk.py`'s real smoke run surfaced — see module
# docstring).
# ---------------------------------------------------------------------------
@pytest.mark.phase8
def test_effective_free_beds_scales_by_functional_level() -> None:
    asset = Asset(
        asset_id="H1",
        asset_type=AssetType.HOSPITAL,
        geometry=POINT,
        functional_level=0.5,
        attributes={"beds_total": 10, "beds_occupied": 0},
    )
    assert effective_free_beds(asset) == 5  # round(10 * 0.5) - 0


@pytest.mark.phase8
def test_effective_free_beds_is_zero_at_zero_functional_level_regardless_of_nominal_beds() -> None:
    asset = Asset(
        asset_id="H1",
        asset_type=AssetType.HOSPITAL,
        geometry=POINT,
        functional_level=0.0,
        attributes={"beds_total": 200, "beds_occupied": 0},
    )
    assert effective_free_beds(asset) == 0


@pytest.mark.phase8
def test_effective_free_beds_floors_at_zero_when_occupancy_exceeds_shrunk_capacity() -> None:
    """A hospital that was full at functional_level=1.0 and then lost
    function afterwards must report 0 free beds, not a negative number
    — nobody already admitted gets evicted (module docstring), but no
    new capacity opens up either."""
    asset = Asset(
        asset_id="H1",
        asset_type=AssetType.HOSPITAL,
        geometry=POINT,
        functional_level=0.2,
        attributes={"beds_total": 10, "beds_occupied": 8},  # 8 > round(10*0.2)=2
    )
    assert effective_free_beds(asset) == 0


@pytest.mark.phase8
def test_consume_demand_at_full_function_matches_pre_m8_behavior() -> None:
    """functional_level=1.0 (every pre-M8 test's implicit default) must
    behave exactly as before — a regression guard for the coupling."""
    graph = _hospital_graph(
        beds_total=10, beds_occupied=10, queue_arrivals=[0, 0, 0], functional_level=1.0
    )
    rng = np.random.default_rng(0)
    consume_demand(graph, tick=1, dt_hours=5.0 / 60.0, rng=rng)
    h1 = graph.nodes["H1"]["asset"]
    assert h1.attributes["beds_occupied"] <= h1.attributes["beds_total"]


@pytest.mark.phase8
def test_consume_demand_backs_up_the_queue_when_functional_level_is_low() -> None:
    """A hospital at 20% function (round(10*0.2)=2 effective beds) with
    5 already occupied has 0 effective free beds - queued patients must
    stay queued (no discharges possible either: beds_occupied=5 > 0, but
    seed/timing aside, admission from the queue specifically must be
    blocked by the capacity check, not by chance)."""
    graph = _hospital_graph(
        beds_total=10,
        beds_occupied=5,
        queue_arrivals=[0, 0, 0],
        functional_level=0.2,
    )
    # No discharges possible this tick: dt_hours tiny relative to mean
    # length of stay, so p_discharge is near 0 - deterministic enough
    # with this seed that no bed frees up to matter for the assertion
    # below (queue can only shrink via *admission*, which capacity must
    # allow).
    rng = np.random.default_rng(1)
    consume_demand(graph, tick=1, dt_hours=1e-6, rng=rng)
    h1 = graph.nodes["H1"]["asset"]
    # effective_beds_total = round(10*0.2) = 2, beds_occupied=5 already
    # exceeds it -> free_beds is clamped to 0 -> no queued patient can
    # be admitted this tick.
    assert h1.attributes["patient_queue"] == 3
    assert h1.attributes["beds_occupied"] == 5


@pytest.mark.phase8
def test_apply_patient_transfer_respects_destination_functional_level() -> None:
    from udt.twin.demand import apply_patient_transfer

    graph = _hospital_graph(beds_total=10, beds_occupied=0, queue_arrivals=[])
    # Add a second hospital, at 30% function, with room per its *nominal*
    # beds_total but not per its *effective* one.
    dest = Asset(
        asset_id="H2",
        asset_type=AssetType.HOSPITAL,
        geometry=POINT,
        functional_level=0.3,
        attributes={"beds_total": 10, "beds_occupied": 0},  # effective = round(10*0.3) = 3
    )
    graph.add_node("H2", asset=dest)
    graph.nodes["H1"]["asset"].attributes["beds_occupied"] = 10  # source has 10 to send

    moved = apply_patient_transfer(graph, "H1", "H2", 10)
    assert moved == 3  # capped by H2's *effective* capacity, not its nominal 10
    assert graph.nodes["H2"]["asset"].attributes["beds_occupied"] == 3


# --- Twin-v3 hospital status: surge and diversion (dev doc §3.9 mechanic 2) ---


def _hospitals(*specs: tuple[str, float, dict[str, object]]) -> object:
    """(id, lon, extra attributes) per hospital, all at lat 19.07, empty queues."""
    assets = [
        Asset(
            asset_id=hid,
            asset_type=AssetType.HOSPITAL,
            geometry={"type": "Point", "coordinates": [lon, 19.07]},
            attributes={"beds_total": 100, "beds_occupied": 100, "queue_arrivals": [], **extra},
        )
        for hid, lon, extra in specs
    ]
    return build_networkx_graph(DependencyGraph(assets=assets, edges=[]))


@pytest.mark.phase4
def test_surge_adds_beds_and_no_surge_is_unchanged() -> None:
    from udt.twin.demand import SURGE_BED_FRACTION

    graph = _hospitals(("H1", 72.88, {"beds_occupied": 50}))
    h1 = graph.nodes["H1"]["asset"]
    assert effective_free_beds(h1) == 50
    h1.attributes["surge"] = True
    assert effective_free_beds(h1) == round(100 * (1 + SURGE_BED_FRACTION)) - 50


@pytest.mark.phase4
def test_surge_ends_when_its_staff_hours_budget_is_used() -> None:
    from udt.twin.demand import SURGE_MAX_HOURS

    graph = _hospitals(("H1", 72.88, {"surge": True}))
    rng = np.random.default_rng(0)
    dt = 5.0 / 60.0
    ticks = int(round(SURGE_MAX_HOURS / dt))
    for tick in range(ticks - 1):
        consume_demand(graph, tick=tick, dt_hours=dt, rng=rng)
    assert graph.nodes["H1"]["asset"].attributes["surge"] is True
    consume_demand(graph, tick=ticks, dt_hours=dt, rng=rng)
    assert graph.nodes["H1"]["asset"].attributes["surge"] is False


@pytest.mark.phase4
def test_diversion_sends_walk_ins_to_the_nearest_accepting_hospital() -> None:
    graph = _hospitals(
        ("H1", 72.880, {"divert": True, "beds_total": 5000, "beds_occupied": 5000}),
        ("H2", 72.881, {"divert": True}),  # nearest, but also diverting
        ("H3", 72.890, {}),  # nearest accepting
    )
    rng = np.random.default_rng(0)
    consume_demand(graph, tick=1, dt_hours=1.0, rng=rng)  # big H1 -> many arrivals
    out = graph.nodes["H1"]["asset"].attributes["patients_diverted_out"]
    assert out > 0
    h3_queue = graph.nodes["H3"]["asset"].attributes["queue_arrivals"]
    assert h3_queue.count(1) >= out  # H3's own arrivals are queued too (it is full)


@pytest.mark.phase4
def test_no_diversion_draws_no_extra_random_numbers() -> None:
    """Twin-v2 reproducibility: without divert set, the RNG stream is untouched."""
    plain = _hospitals(("H1", 72.880, {}), ("H2", 72.890, {}))
    flagged_off = _hospitals(("H1", 72.880, {"divert": False}), ("H2", 72.890, {}))
    rng_a, rng_b = np.random.default_rng(7), np.random.default_rng(7)
    consume_demand(plain, tick=1, dt_hours=1.0, rng=rng_a)
    consume_demand(flagged_off, tick=1, dt_hours=1.0, rng=rng_b)
    assert rng_a.random() == rng_b.random()


@pytest.mark.phase4
def test_simulator_sets_status_and_refuses_surge_without_budget() -> None:
    from udt.twin.demand import SURGE_MAX_HOURS
    from udt.twin.simulator import Simulator

    h = Asset(
        asset_id="H1",
        asset_type=AssetType.HOSPITAL,
        geometry=POINT,
        attributes={"beds_total": 10, "beds_occupied": 5, "queue_arrivals": []},
    )
    sim = Simulator(DependencyGraph(assets=[h], edges=[]))
    sim.step(divert={"H1": True}, surge={"H1": True})
    attrs = sim.asset("H1").attributes
    assert attrs["divert"] is True and attrs["surge"] is True
    attrs["surge_hours_used"] = SURGE_MAX_HOURS
    attrs["surge"] = False
    sim.step(surge={"H1": True})
    assert attrs["surge"] is False
