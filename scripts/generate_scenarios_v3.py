#!/usr/bin/env python3
"""Generate the frozen twin-v3 scenario suite, once (protocol
2026-09-29_twin_v3_success_criteria §3 + addendum 3; dev doc §3.9).

Like `generate_scenarios.py` (v2), with twin-v3 strata and trade-off features:
- strata = flood-centre sector (nearest hospital) x severity band (3 x 3 cells);
  each cell gets floor(n / 9) scenarios, the n mod 9 extra slots go to the next
  scenarios in seed order;
- trade-off features T1' (route-based), T2 (multi-facility threat) and T3 (high
  call volume) come from the flood physics only (`udt.scenarios.tradeoffs`). On
  train each must reach `tradeoff_min_share_train`: scanning in seed order, a
  scenario lacking a still-needed feature is skipped only when accepting it would
  make that quota unreachable. No policy is ever run. T3's median is the train
  split's median of expected calls; T1 (as registered, unattainable) is reported;
- the same leakage checks as v2; every file is written read-only and its SHA-256
  recorded. The manifest records the generating commit (the relevant code must be
  committed and clean) and the SHA-256 of every data input.

Usage: uv run python scripts/generate_scenarios_v3.py --config configs/scenarios_v3.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from shapely.geometry import shape

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from _pipeline_common import configure_logging, resolve_path  # noqa: E402
from udt.common.models import DependencyGraph, Scenario  # noqa: E402
from udt.common.versions import ENV_VERSION_V3  # noqa: E402
from udt.incidents.degradations.flood import SusceptibilityRaster  # noqa: E402
from udt.scenarios.generator import DEFAULT_SEVERITY_RANGE, generate_flood_scenario  # noqa: E402
from udt.scenarios.suite import (  # noqa: E402
    MANIFEST_NAME,
    feature_ranges,
    min_distance_to,
    normalised_features,
    sha256_file,
)
from udt.scenarios.tradeoffs import (  # noqa: E402
    SEVERITY_BANDS,
    route_tradeoff,
    tradeoff_features,
)
from udt.twin.road_network import RoadNetwork  # noqa: E402

MAX_SEEDS_SCANNED = 200_000
CODE_PATHS = (
    "src/udt/twin",
    "src/udt/incidents",
    "src/udt/scenarios",
    "scripts/generate_scenarios_v3.py",
    "configs/scenarios_v3.yaml",
)
DATA_FILES = (
    "data/processed/dependency_graph_v3.json",
    "data/processed/flood_susceptibility.tif",
    "data/processed/ward_boundary.geojson",
    "data/processed/roads_full.graphml",
)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def _features(sc: Scenario, graph: DependencyGraph, raster: Any, rn: RoadNetwork) -> dict[str, Any]:
    f = tradeoff_features(sc, graph, raster)
    f["T1p_route_tradeoff"] = route_tradeoff(sc, graph, rn, f["p_fail"])
    return f


def sample_split(
    split: str,
    n: int,
    seed_start: int,
    ward_geojson: dict[str, Any],
    graph: DependencyGraph,
    raster: Any,
    rn: RoadNetwork,
    hospitals: list[str],
    quotas: dict[str, int],
    suite_version: str,
) -> tuple[list[Scenario], list[dict[str, Any]]]:
    cells = [(h, b) for h in hospitals for b in SEVERITY_BANDS]
    per_cell = n // len(cells)
    counts = dict.fromkeys(cells, 0)
    free_slots = n - per_cell * len(cells)
    have = dict.fromkeys(quotas, 0)
    chosen: list[Scenario] = []
    feats: list[dict[str, Any]] = []
    for seed in range(seed_start, seed_start + MAX_SEEDS_SCANNED):
        sc = generate_flood_scenario(
            scenario_id=f"flood_v3_{split}_{len(chosen):02d}",
            ward_boundary_geojson=ward_geojson,
            seed=seed,
        )
        f = _features(sc, graph, raster, rn)
        cell = (f["sector"], f["severity_band"])
        uses_free = counts[cell] >= per_cell
        if uses_free and free_slots == 0:
            continue
        slots_left_after = n - len(chosen) - 1
        feasible = all(
            max(0, quotas[k] - have[k] - int(bool(f[k]))) <= slots_left_after for k in quotas
        )
        if not feasible:
            continue
        counts[cell] += 1
        free_slots -= int(uses_free)
        for k in quotas:
            have[k] += int(bool(f[k]))
        sc.suite_version = suite_version
        chosen.append(sc)
        feats.append({"scenario_id": sc.scenario_id, "seed": seed, **f})
        if len(chosen) == n:
            return chosen, feats
    raise SystemExit(f"{split}: not filled within {MAX_SEEDS_SCANNED} seeds")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/scenarios_v3.yaml")
    args = parser.parse_args()
    log = configure_logging()
    cfg = yaml.safe_load(resolve_path(args.config).read_text())
    out_dir = resolve_path(cfg["out_dir"])
    if (out_dir / MANIFEST_NAME).exists():
        raise SystemExit(f"{out_dir / MANIFEST_NAME} exists: the suite is frozen")
    dirty = _git("status", "--porcelain", "--", *CODE_PATHS, *DATA_FILES)
    if dirty:
        raise SystemExit(f"commit the suite's code and data first:\n{dirty}")
    commit = _git("rev-parse", "HEAD")

    processed = REPO_ROOT / "data" / "processed"
    ward_geojson = json.loads((processed / "ward_boundary.geojson").read_text())
    ward = shape(ward_geojson["features"][0]["geometry"])
    graph = DependencyGraph.model_validate_json(
        (processed / "dependency_graph_v3.json").read_text()
    )
    hospitals = [a.asset_id for a in graph.assets if a.asset_type.value == "hospital"]

    splits: dict[str, list[Scenario]] = {}
    features: dict[str, list[dict[str, Any]]] = {}
    with SusceptibilityRaster(processed / "flood_susceptibility.tif") as raster:
        rn = RoadNetwork.load(processed / "roads_full.graphml", raster)
        for name, spec in cfg["splits"].items():
            k = math.ceil(cfg["tradeoff_min_share_train"] * spec["n"]) if name == "train" else 0
            quotas = {"T1p_route_tradeoff": k, "T2_multi_facility_threat": k} if k else {}
            splits[name], features[name] = sample_split(
                name,
                spec["n"],
                spec["seed_start"],
                ward_geojson,
                graph,
                raster,
                rn,
                hospitals,
                quotas,
                cfg["suite_version"],
            )

    median_calls = float(np.median([f["expected_calls"] for f in features["train"]]))
    for fl in features.values():
        for f in fl:
            f["T3_high_call_volume"] = f["expected_calls"] > median_calls
    share = {
        name: {
            k: round(sum(bool(f[k]) for f in fl) / len(fl), 3)
            for k in (
                "T1_alt_access_closure",
                "T1p_route_tradeoff",
                "T2_multi_facility_threat",
                "T3_high_call_volume",
            )
        }
        for name, fl in features.items()
    }

    checks: dict[str, Any] = {}
    seeds = {n: [s.seed for s in sc] for n, sc in splits.items()}
    all_seeds = [x for v in seeds.values() for x in v]
    checks["seeds_disjoint_across_splits"] = len(all_seeds) == len(set(all_seeds))
    frag = [s.incident.profile["fragility_seed"] for sc in splits.values() for s in sc]
    beds = [s.initial_conditions["bed_occupancy_seed"] for sc in splits.values() for s in sc]
    checks["fragility_seeds_unique"] = len(frag) == len(set(frag))
    checks["bed_occupancy_seeds_unique"] = len(beds) == len(set(beds))
    ranges = feature_ranges(ward.bounds, DEFAULT_SEVERITY_RANGE)
    vectors = [
        tuple(np.round(normalised_features(s, ranges), 9)) for sc in splits.values() for s in sc
    ]
    checks["no_duplicate_parameter_vectors"] = len(vectors) == len(set(vectors))
    min_d = {
        name: min_distance_to(splits[name], splits["train"], ranges) for name in ("val", "test")
    }
    min_d["test_to_val"] = min_distance_to(splits["test"], splits["val"], ranges)
    checks["min_normalised_distance"] = {k: round(v, 4) for k, v in min_d.items()}
    checks["no_near_duplicates"] = all(v >= cfg["leakage_min_distance"] for v in min_d.values())
    coverage = {
        name: sorted({f"{f['sector']}|{f['severity_band']}" for f in fl})
        for name, fl in features.items()
    }
    checks["all_cells_covered"] = all(len(v) == 3 * len(SEVERITY_BANDS) for v in coverage.values())
    need = cfg["tradeoff_min_share_train"]
    checks["train_tradeoff_shares_met"] = all(
        share["train"][k] >= need
        for k in ("T1p_route_tradeoff", "T2_multi_facility_threat", "T3_high_call_volume")
    )
    must_pass = [
        "seeds_disjoint_across_splits",
        "fragility_seeds_unique",
        "bed_occupancy_seeds_unique",
        "no_duplicate_parameter_vectors",
        "no_near_duplicates",
        "all_cells_covered",
        "train_tradeoff_shares_met",
    ]
    failed = [k for k in must_pass if not checks[k]]
    if failed:
        raise SystemExit(
            f"suite checks failed, nothing written: {failed}\n"
            f"{json.dumps(checks, indent=2)}\n{share}"
        )

    tmp = Path(
        tempfile.mkdtemp(
            prefix="suite_v3_", dir=out_dir.parent if out_dir.parent.exists() else None
        )
    )
    files = []
    for name, sc in splits.items():
        (tmp / name).mkdir(parents=True)
        for s in sc:
            path = tmp / name / f"{s.scenario_id}.json"
            path.write_text(json.dumps(s.model_dump(mode="json"), indent=2, sort_keys=True) + "\n")
            files.append(
                {"path": f"{name}/{path.name}", "sha256": sha256_file(path), "seed": s.seed}
            )
    manifest = {
        "suite_version": cfg["suite_version"],
        "incident_type": cfg["incident_type"],
        "env_version": ENV_VERSION_V3,
        "generated_at_commit": commit,
        "data_sha256": {p: sha256_file(REPO_ROOT / p) for p in DATA_FILES},
        "created_at": datetime.now(UTC).isoformat(),
        "config": cfg,
        "splits": {name: {"n": len(sc), "seeds": seeds[name]} for name, sc in splits.items()},
        "cells_covered": coverage,
        "tradeoff_shares": share,
        "t3_median_expected_calls_train": median_calls,
        "scenario_features": {
            name: [{k: v for k, v in f.items() if k not in ("p_fail", "access_closed")} for f in fl]
            for name, fl in features.items()
        },
        "checks": checks,
        "files": files,
    }
    (tmp / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    for p in tmp.rglob("*.json"):
        os.chmod(p, 0o444)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp.rename(out_dir)
    log.info("suite_v3_written", out_dir=str(out_dir), **{k: len(v) for k, v in splits.items()})
    print(
        json.dumps(
            {"checks": checks, "tradeoff_shares": share, "cells_covered": coverage}, indent=2
        )
    )


if __name__ == "__main__":
    main()
