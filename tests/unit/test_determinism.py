"""Phase 1 acceptance test (dev doc §3.6): same seed + same scenario =>
bit-identical trajectories."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))
from sample_graph import build_sample_dependency_graph  # noqa: E402

from udt.common.models import Incident
from udt.twin.simulator import Simulator

FLOOD_INCIDENT = Incident(
    incident_id="det_test",
    type="flood",
    location={"type": "Point", "coordinates": [72.88, 19.07]},
    onset_tick=0,
    severity=0.7,
    directly_affected_assets=[],
)


def _run(n_ticks: int = 60) -> list[dict]:
    dep_graph = build_sample_dependency_graph()

    def degradation_fn(tick: int, graph: object) -> dict[str, float]:
        # Deterministic synthetic degradation (no raster I/O needed here —
        # this test is about the simulator's own determinism, not the
        # flood-specific degradation function, which has its own tests).
        import math

        return {"H1": 0.05 * math.sin(tick / 5.0) ** 2} if tick < 20 else {}

    sim = Simulator(dep_graph)
    snapshots = sim.run(n_ticks, degradation_fn=degradation_fn)
    return [s.model_dump(mode="json") for s in snapshots]


@pytest.mark.phase1
def test_same_scenario_gives_bit_identical_trajectory() -> None:
    trace_a = _run()
    trace_b = _run()
    assert trace_a == trace_b


@pytest.mark.phase1
def test_cascading_failure_count_is_monotonic_nondecreasing() -> None:
    trace = _run()
    counts = [snap["cascading_failure_count"] for snap in trace]
    assert counts == sorted(counts)
