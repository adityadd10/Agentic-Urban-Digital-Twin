#!/usr/bin/env python3
"""Prototype-1 demo entrypoint.

Injects a flood scenario, ticks the twin, logs per-tick functional levels,
writes a full JSON trace under `runs/<run_id>/`. Headless/scripted — no
frontend yet (dev doc M10c, later).

Uses the real dependency graph (`data/processed/dependency_graph.json`,
from `scripts/06_build_dependency_graph.py`) if it exists; otherwise falls
back to the hand-built 10-asset test fixture
(`tests/fixtures/sample_graph.py`) with a clear warning — the real graph
needs `02_extract_roads.py`/`03_extract_facilities.py`, which need OSM
access this sandbox didn't have (see MTP_Module_Planner.md's session log).
Either way, drives the *real* DEM-derived susceptibility raster
(`data/processed/flood_susceptibility.tif`) and the real ward boundary.

Usage:
  uv run python scripts/run_flood_demo.py --config configs/data.yaml
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
sys.path.insert(0, str(REPO_ROOT / "tests" / "fixtures"))

from sample_graph import build_sample_dependency_graph  # noqa: E402

from _pipeline_common import configure_logging, load_config, resolve_path  # noqa: E402
from udt.common.models import DependencyGraph  # noqa: E402
from udt.incidents.degradations.flood import (  # noqa: E402
    SusceptibilityRaster,
    make_flood_degradation_fn,
)
from udt.scenarios.generator import generate_flood_scenario  # noqa: E402
from udt.twin.simulator import Simulator  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--seed", type=int, default=None, help="overrides configs/data.yaml's seed")
    parser.add_argument(
        "--n-ticks", type=int, default=288, help="288 ticks = 24h at dt=5min (dev doc §3.1)"
    )
    parser.add_argument(
        "--log-every", type=int, default=12, help="log a summary every N ticks (1h default)"
    )
    args = parser.parse_args()

    log = configure_logging()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]

    processed_dir = resolve_path(cfg["paths"]["processed_dir"])
    dep_graph_path = processed_dir / "dependency_graph.json"
    if dep_graph_path.exists():
        with dep_graph_path.open() as f:
            dep_graph = DependencyGraph.model_validate(json.load(f))
        log.info("using_real_dependency_graph", path=str(dep_graph_path))
    else:
        dep_graph = build_sample_dependency_graph()
        log.warning(
            "dependency_graph_not_found_using_fixture",
            expected_path=str(dep_graph_path),
            note="run 02/03/06 (needs OSM access) for the real ward graph",
        )

    susceptibility_path = processed_dir / "flood_susceptibility.tif"
    if not susceptibility_path.exists():
        raise SystemExit(
            f"{susceptibility_path} not found — run 04_get_dem.py "
            "and 05_flood_susceptibility.py first"
        )

    boundary_path = processed_dir / "ward_boundary.geojson"
    if not boundary_path.exists():
        raise SystemExit(f"{boundary_path} not found — run 01_get_boundary.py first")
    with boundary_path.open() as f:
        ward_boundary = json.load(f)

    scenario = generate_flood_scenario(
        scenario_id="demo_flood_001",
        ward_boundary_geojson=ward_boundary,
        seed=seed,
    )
    log.info("scenario_generated", severity=scenario.incident.severity, seed=seed)

    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
    run_dir = REPO_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with SusceptibilityRaster(susceptibility_path) as raster:
        degradation_fn = make_flood_degradation_fn(scenario.incident, raster)
        sim = Simulator(dep_graph, seed=seed)

        trace = []
        for _ in range(args.n_ticks):
            snapshot = sim.step(degradation_fn=degradation_fn)
            trace.append(snapshot.model_dump(mode="json"))
            if snapshot.tick % args.log_every == 0:
                levels = {a.asset_id: round(a.functional_level, 3) for a in snapshot.assets}
                log.info(
                    "tick_summary",
                    tick=snapshot.tick,
                    hours=round(snapshot.tick * 5 / 60, 2),
                    cascading_failures=snapshot.cascading_failure_count,
                    functional_levels=levels,
                )

    trace_path = run_dir / "trace.json"
    trace_path.write_text(json.dumps(trace, indent=2))
    log.info(
        "demo_complete", run_dir=str(run_dir), trace_path=str(trace_path), n_ticks=args.n_ticks
    )


if __name__ == "__main__":
    main()
