#!/usr/bin/env python3
"""Experiment A harness (dev doc §5.6, Phase 4, module M4 — "first
end-to-end numbers milestone").

Runs the same flood scenario, over the same real dependency graph, twice —
once with `DoNothingAgent` (the control: what happens if nobody responds)
and once with `RuleBasedAgent` (dev doc §5.6's baseline, prototype-1 scope:
repair-crew dispatch + ambulance dispatch — see `udt.agents.rule_based`'s
docstring for what's still deferred and why) — and prints/writes a
comparison table.

**Disclosed prototype-1 scope:**
- Only ONE scenario is run, not the dev doc's full 20/10/10 frozen suite
  (§4.3) — that suite doesn't exist yet (M3's own row defers it to
  whichever module first needs train/val/test splits). This table is
  "one scenario's outcome per agent", not "mean ± CI over N scenarios" —
  reported as such. Run with `--seed` to see a different single scenario;
  extend this runner to loop over a scenario list once the suite exists.
- No MLflow (not yet a project dependency, dev doc §12.2 mentions it for
  the full training pipeline) — results are printed and written to
  `runs/<run_id>/experiment_a_results.json`. Add MLflow when a real
  multi-scenario/multi-seed sweep makes a tracking UI worth the
  dependency, not before.
- Ambulance dispatch needs `roads_full.graphml` (the complete real road
  network, from `02_extract_roads.py`) — if it's missing, this runner
  still runs, just without a fleet (`road_network=None` throughout,
  matching how `RuleBasedAgent`/`DoNothingAgent` already handle that
  case).

**M7 addition:** every decision is now run through `constraints/
engine.py`'s `check(...)` (dev doc §8 usage (2), "post-hoc validation of
every action before execution, all experiments, all paths") before it's
applied to `sim.step` — whatever comes back repaired (clipped/dropped)
is what actually executes. `RuleBasedAgent`'s own rules rarely trip
these (they're already conservative — e.g. `_pick_patient_transfer`
already checks free-bed count itself), so this is mostly a consistency/
safety-net wiring exercise for this harness, not expected to change
Experiment A's numbers much; `EpisodeMetrics.safety_violations_
attempted` reports whatever it does catch (dev doc §8's H3 metric).

**Fixed after M6a landed — a real fairness bug, not just a style
nit:** this harness originally called `agent.act()` and applied a fresh
decision *every* tick (5 min). `envs/single_env.py`/`multi_env.py` (M5/
M6a) only let their policy decide once every `DECISION_INTERVAL_TICKS`
ticks (15 min, dev doc §5.1 exactly), holding that decision for the
other 2 ticks. That meant `rule_based`/`do_nothing` got 3x as many
decision opportunities as any RL policy evaluated against the Gym/
PettingZoo envs — a real advantage for the rule-based side in any
cross-harness comparison (`scripts/evaluate.py`), not a fair baseline.
`run_episode` now decides once per `DECISION_INTERVAL_TICKS`-tick
window too, applying that decision's commands only on the window's
first tick — the exact same pattern `single_env.step`/`multi_env.step`
already use internally.

Usage:
  uv run python experiments/runner.py --config configs/data.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "tests" / "fixtures"))

from sample_graph import build_sample_dependency_graph  # noqa: E402
from shapely.geometry import shape  # noqa: E402

from _pipeline_common import configure_logging, load_config, resolve_path  # noqa: E402
from udt.agents.base import Agent  # noqa: E402
from udt.agents.rule_based import DoNothingAgent, RuleBasedAgent  # noqa: E402
from udt.common.models import AgentAction, DependencyGraph, EpisodeMetrics, Incident  # noqa: E402
from udt.constraints.engine import check  # noqa: E402
from udt.incidents.degradations.flood import (  # noqa: E402
    SusceptibilityRaster,
    make_flood_degradation_fn,
)
from udt.logging.metrics import compute_episode_metrics  # noqa: E402
from udt.scenarios.generator import generate_flood_scenario  # noqa: E402
from udt.twin.ambulances import generate_requests, spawn_ambulances  # noqa: E402
from udt.twin.power import update_substation_load  # noqa: E402
from udt.twin.road_network import RoadNetwork  # noqa: E402
from udt.twin.simulator import (  # noqa: E402
    DegradationFn,
    Simulator,
    TwinState,
)

N_AMBULANCES_PER_HOSPITAL = 2  # disclosed placeholder, see twin/ambulances.py
# dev doc §5.1 exactly — matches envs/single_env.py's/multi_env.py's constant
# of the same name and value; duplicated rather than imported, since this
# harness predates those envs and has no other reason to depend on them.
DECISION_INTERVAL_TICKS = 3


def run_episode(
    dep_graph: DependencyGraph,
    agent: Agent,
    degradation_fn: DegradationFn,
    n_ticks: int,
    *,
    seed: int = 0,
    road_network: RoadNetwork | None = None,
    incident: Incident | None = None,
    ward_polygon: object | None = None,
) -> list[TwinState]:
    """One full episode with one agent. `seed` drives the twin's own
    patient-arrival/discharge/request-generation sampling — passed
    explicitly so both agents in one comparison see the same arrivals,
    not two independently-random sequences.

    `road_network`/`incident`/`ward_polygon`: if all three are given, a
    fleet is spawned and emergency requests are generated each tick
    (dev doc §5.3/§5.6 ambulance dispatch); `road_network.update_for_tick`
    and request generation happen *before* `agent.act`, so the agent's
    own routing queries and dispatch decision see this tick's state, not
    last tick's (see `twin/simulator.py`'s docstring on why this ordering
    matters and lives in the caller, not inside `Simulator.step`).
    `incident` alone (independent of `road_network`/`ward_polygon`) also
    drives `twin/power.py`'s `update_substation_load` each tick, for the
    same before-`agent.act` reason — the load-shedding rule needs this
    tick's current load, not last tick's.

    **Decision cadence matches the Gym/PettingZoo envs (dev doc §5.1):**
    `agent.act` is only actually consulted on the first tick of each
    `DECISION_INTERVAL_TICKS`-tick window; the other ticks in that window
    re-apply no *new* repair/ambulance/transfer/shed command (physics —
    degradation, buffers, demand, ambulances already en route — still
    advances every tick regardless). See this function's module-level
    docstring note for why this was fixed after M6a exposed the
    asymmetry."""
    sim = Simulator(dep_graph, seed=seed, road_network=road_network)
    if road_network is not None:
        spawn_ambulances(sim.graph, road_network, N_AMBULANCES_PER_HOSPITAL)

    trace: list[TwinState] = []
    action: AgentAction | None = None
    safety_violations_attempted_total = 0
    for _ in range(n_ticks):
        if incident is not None:
            update_substation_load(sim.graph, sim.tick, incident, sim.dt_minutes)
        if road_network is not None and incident is not None and ward_polygon is not None:
            road_network.update_for_tick(incident, sim.tick, sim.dt_minutes)
            generate_requests(
                sim.graph,
                sim.tick,
                sim.dt_hours,
                incident,
                ward_polygon,
                sim.rng,  # type: ignore[arg-type]
            )

        is_decision_tick = sim.tick % DECISION_INTERVAL_TICKS == 0
        if is_decision_tick:
            action = agent.act(
                sim.graph, sim.tick, road_network=road_network, edge_states=sim.edge_states
            )
            # M7: post-hoc validation (dev doc §8 usage (2)) — see this
            # function's module-level docstring note. The repaired
            # action is what's actually applied below.
            report = check(sim.graph, action, road_network=road_network)
            action = report.repaired_action
            safety_violations_attempted_total += len(report.violations)
        assert action is not None  # first tick (tick=0) is always a decision tick

        snapshot = sim.step(
            degradation_fn=degradation_fn,
            repair_target=action.repair_target if is_decision_tick else None,
            ambulance_assignment=action.ambulance_assignment if is_decision_tick else None,
            patient_transfer=action.patient_transfer if is_decision_tick else None,
            shed_tier=action.shed_tier if is_decision_tick else None,
        )
        snapshot.safety_violations_attempted_cumulative = safety_violations_attempted_total
        trace.append(snapshot)
    return trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--seed", type=int, default=None, help="overrides configs/data.yaml's seed")
    parser.add_argument("--n-ticks", type=int, default=288)
    args = parser.parse_args()

    log = configure_logging()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]

    processed_dir = resolve_path(cfg["paths"]["processed_dir"])
    dep_graph_path = processed_dir / "dependency_graph.json"
    if dep_graph_path.exists():
        with dep_graph_path.open() as f:
            base_graph = DependencyGraph.model_validate(json.load(f))
        log.info("using_real_dependency_graph", path=str(dep_graph_path))
    else:
        base_graph = build_sample_dependency_graph()
        log.warning("dependency_graph_not_found_using_fixture", expected_path=str(dep_graph_path))

    susceptibility_path = processed_dir / "flood_susceptibility.tif"
    boundary_path = processed_dir / "ward_boundary.geojson"
    roads_full_path = processed_dir / "roads_full.graphml"
    if not susceptibility_path.exists():
        raise SystemExit(f"{susceptibility_path} not found — run 04/05 first")
    if not boundary_path.exists():
        raise SystemExit(f"{boundary_path} not found — run 01_get_boundary.py first")
    with boundary_path.open() as f:
        ward_boundary = json.load(f)
    ward_polygon = shape(ward_boundary["features"][0]["geometry"])

    scenario = generate_flood_scenario(
        scenario_id="experiment_a_flood_001",
        ward_boundary_geojson=ward_boundary,
        seed=seed,
    )
    log.info("scenario_generated", severity=scenario.incident.severity, seed=seed)

    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
    run_dir = REPO_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    results: list[EpisodeMetrics] = []
    with SusceptibilityRaster(susceptibility_path) as raster:
        degradation_fn = make_flood_degradation_fn(scenario.incident, raster)

        road_network: RoadNetwork | None = None
        if roads_full_path.exists():
            road_network = RoadNetwork.load(roads_full_path, raster)
            log.info("road_network_loaded", path=str(roads_full_path))
        else:
            log.warning(
                "roads_full_not_found_no_ambulance_fleet", expected_path=str(roads_full_path)
            )

        for agent in (DoNothingAgent(), RuleBasedAgent()):
            # Fresh deep copy per agent: Simulator mutates Asset objects
            # in place (dev doc §3.5), so reusing base_graph across two
            # runs would silently start the second agent from the first
            # agent's already-damaged end state instead of the same
            # initial conditions.
            graph_copy = base_graph.model_copy(deep=True)
            trace = run_episode(
                graph_copy,
                agent,
                degradation_fn,
                args.n_ticks,
                seed=seed,
                road_network=road_network,
                incident=scenario.incident,
                ward_polygon=ward_polygon,
            )
            metrics = compute_episode_metrics(scenario.scenario_id, agent.name, trace)
            results.append(metrics)
            log.info("episode_complete", **metrics.model_dump())

    results_path = run_dir / "experiment_a_results.json"
    results_path.write_text(json.dumps([r.model_dump() for r in results], indent=2))

    print("\nExperiment A — rule-based baseline vs. do-nothing (n=1 scenario, not a CI)")
    print(
        f"{'agent':<14} {'cascading_failures':>18} {'mean_hospital_fl':>17} "
        f"{'mean_critical_fl':>17} {'unmet_patient_hrs':>17} {'patient_deaths':>14} "
        f"{'mean_amb_delay_hrs':>18} {'reqs_done':>9} {'reqs_pending':>12} {'transferred':>11} "
        f"{'unserved_mwh':>12} {'safety_viol':>11}"
    )
    for r in results:
        delay = (
            f"{r.mean_ambulance_response_delay_hours:.2f}"
            if r.mean_ambulance_response_delay_hours is not None
            else "n/a"
        )
        print(
            f"{r.agent_name:<14} {r.cascading_failure_count:>18} "
            f"{r.mean_hospital_functional_level:>17.3f} {r.mean_critical_functional_level:>17.3f} "
            f"{r.unmet_patient_hours:>17.1f} {r.patient_deaths:>14} "
            f"{delay:>18} {r.requests_completed:>9} {r.requests_pending_at_end:>12} "
            f"{r.patients_transferred:>11} {r.unserved_energy_mwh:>12.2f} "
            f"{r.safety_violations_attempted:>11}"
        )
    print(f"\nWritten to {results_path}")


if __name__ == "__main__":
    main()
