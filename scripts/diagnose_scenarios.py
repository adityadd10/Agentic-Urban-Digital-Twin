#!/usr/bin/env python3
"""Scenario-diversity diagnostic (dev doc §4.3, 2026-09-26: "check that outcomes
vary across scenarios before generating the frozen suite").

Runs N generated flood scenarios on the real graph with no agent acting
(do-nothing), for a full 24 h episode each, and prints one row per scenario:
severity, footprint size, how many roads are blocked at peak, which
facilities failed from flooding, cascade count (the revised §3.5 metric),
minimum hospital functional level, and the patient-death proxy. Then it
summarises the spread. If every row looks the same, the generator still
produces one flood in disguise.

Usage:
  uv run python scripts/diagnose_scenarios.py --n 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from _pipeline_common import load_config, resolve_path  # noqa: E402
from udt.common.models import AssetType, DependencyGraph  # noqa: E402
from udt.incidents.degradations.flood import (  # noqa: E402
    FAILED_RESIDUAL,
    SusceptibilityRaster,
    make_flood_degradation_fn,
)
from udt.scenarios.generator import (  # noqa: E402
    apply_initial_conditions,
    generate_flood_scenario,
    onset_hour_of_day,
)
from udt.twin.power import update_substation_load  # noqa: E402
from udt.twin.simulator import Simulator  # noqa: E402

N_TICKS = 288  # 24 h at dt = 5 min
FACILITY_TYPES = (AssetType.HOSPITAL, AssetType.SUBSTATION, AssetType.WATER)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--n", type=int, default=20, help="number of scenarios (seeds 0..n-1)")
    args = parser.parse_args()

    processed = resolve_path(load_config(args.config)["paths"]["processed_dir"])
    with (processed / "dependency_graph.json").open() as f:
        base_graph = DependencyGraph.model_validate(json.load(f))
    with (processed / "ward_boundary.geojson").open() as f:
        ward = json.load(f)

    rows = []
    with SusceptibilityRaster(processed / "flood_susceptibility.tif") as raster:
        for seed in range(args.n):
            scenario = generate_flood_scenario(
                scenario_id=f"diag{seed}", ward_boundary_geojson=ward, seed=seed
            )
            incident = scenario.incident
            fn = make_flood_degradation_fn(incident, raster)
            sim = Simulator(
                apply_initial_conditions(base_graph, scenario),
                seed=seed,
                onset_hour_of_day=onset_hour_of_day(scenario),
            )
            roads = [n for n in sim.graph.nodes if sim.asset(n).asset_type == AssetType.ROAD]
            facilities = [n for n in sim.graph.nodes if sim.asset(n).asset_type in FACILITY_TYPES]
            peak_blocked, min_hosp = 0, 1.0
            flooded: set[str] = set()
            snap = None
            for _ in range(N_TICKS):
                update_substation_load(sim.graph, sim.tick, incident, sim.dt_minutes)
                snap = sim.step(degradation_fn=fn)
                peak_blocked = max(
                    peak_blocked,
                    sum(1 for r in roads if sim.asset(r).attributes.get("blockage", 0) >= 0.95),
                )
                for fid in facilities:
                    a = sim.asset(fid)
                    if a.intrinsic_level <= FAILED_RESIDUAL[a.asset_type] + 1e-9:
                        flooded.add(fid)
                    if a.asset_type == AssetType.HOSPITAL:
                        min_hosp = min(min_hosp, a.functional_level)
            assert snap is not None
            rows.append(
                {
                    "seed": seed,
                    "severity": incident.severity,
                    "sigma_km": incident.profile["footprint"]["sigma_m"] / 1000,
                    "roads_blocked_peak": peak_blocked,
                    "n_roads": len(roads),
                    "flooded": sorted(flooded),
                    "cascades": snap.cascading_failure_count,
                    "min_hospital_fl": min_hosp,
                    "deaths_proxy": snap.patient_deaths_cumulative,
                }
            )

    print(
        f"{'seed':>4} {'sev':>5} {'σ km':>5} {'roads blocked':>13} {'cascades':>8} "
        f"{'min hosp fl':>11} {'deaths*':>7}  flooded facilities"
    )
    for r in rows:
        print(
            f"{r['seed']:>4} {r['severity']:>5.2f} {r['sigma_km']:>5.2f} "
            f"{r['roads_blocked_peak']:>6}/{r['n_roads']:<6} {r['cascades']:>8} "
            f"{r['min_hospital_fl']:>11.2f} {r['deaths_proxy']:>7}  "
            f"{', '.join(r['flooded']) or '-'}"
        )

    def spread(name: str, values: list[float], fmt: str) -> None:
        v = np.array(values, dtype=float)
        lo, mid, hi = v.min(), float(np.median(v)), v.max()
        print(f"  {name:<28} {lo:{fmt}} / {mid:{fmt}} / {hi:{fmt}}")

    print("\nspread across scenarios (min / median / max):")
    spread("roads fully blocked at peak", [r["roads_blocked_peak"] for r in rows], ".0f")
    spread("facilities flooded", [len(r["flooded"]) for r in rows], ".0f")
    spread("min hospital functional", [r["min_hospital_fl"] for r in rows], ".2f")
    spread("deaths (queue-wait proxy)", [r["deaths_proxy"] for r in rows], ".0f")
    print(f"  scenarios with >=1 cascade:  {sum(r['cascades'] > 0 for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
