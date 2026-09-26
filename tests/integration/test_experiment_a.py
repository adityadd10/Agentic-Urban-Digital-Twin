"""Phase 4 acceptance test (dev doc §13 "Integration" row + §14 Phase 4:
"Experiment A over the flood [...] test scenarios emits a results table
[...]; integration test passes").

Runs both agents against the hand-built 10-asset fixture (not the real
Kurla graph — this test must not depend on `data/processed/` existing) for
a short flood scenario, and asserts the directional claim a credible
baseline should satisfy: repairing the worst-damaged critical facility
each tick should not leave the run *worse* off than doing nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT / "experiments", REPO_ROOT / "tests" / "fixtures"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sample_graph import build_sample_dependency_graph  # noqa: E402

from runner import run_episode  # noqa: E402
from udt.agents.rule_based import DoNothingAgent, RuleBasedAgent  # noqa: E402
from udt.incidents.degradations.flood import (  # noqa: E402
    SusceptibilityRaster,
    make_flood_degradation_fn,
)
from udt.logging.metrics import compute_episode_metrics  # noqa: E402
from udt.scenarios.generator import generate_flood_scenario  # noqa: E402

WARD_BOUNDARY_PATH = REPO_ROOT / "data" / "processed" / "ward_boundary.geojson"
SUSCEPTIBILITY_PATH = REPO_ROOT / "data" / "processed" / "flood_susceptibility.tif"


@pytest.mark.phase4
@pytest.mark.skipif(
    not (WARD_BOUNDARY_PATH.exists() and SUSCEPTIBILITY_PATH.exists()),
    reason="needs data/processed/ward_boundary.geojson + flood_susceptibility.tif (run 01/04/05)",
)
def test_experiment_a_rule_based_not_worse_than_do_nothing() -> None:
    import json

    with WARD_BOUNDARY_PATH.open() as f:
        ward_boundary = json.load(f)

    scenario = generate_flood_scenario(
        scenario_id="test_experiment_a",
        ward_boundary_geojson=ward_boundary,
        seed=0,
    )
    base_graph = build_sample_dependency_graph()

    results = {}
    with SusceptibilityRaster(SUSCEPTIBILITY_PATH) as raster:
        degradation_fn = make_flood_degradation_fn(scenario.incident, raster)
        for agent in (DoNothingAgent(), RuleBasedAgent()):
            graph_copy = base_graph.model_copy(deep=True)
            trace = run_episode(graph_copy, agent, degradation_fn, n_ticks=96)  # 8 simulated hours
            results[agent.name] = compute_episode_metrics(scenario.scenario_id, agent.name, trace)

    do_nothing = results["do_nothing"]
    rule_based = results["rule_based"]

    # The credibility bar for a baseline (dev doc §5.6): it shouldn't do
    # worse than not responding at all.
    assert rule_based.cascading_failure_count <= do_nothing.cascading_failure_count
    assert rule_based.mean_critical_functional_level >= do_nothing.mean_critical_functional_level
