#!/usr/bin/env python3
"""Generate the frozen scenario suite (dev doc §4.3), once.

Writes `data/scenarios/flood/{train,val,test}/*.json` plus `manifest.json`.
Each split scans its own disjoint seed range and keeps a scenario only if its
stratum (severity band x ward half, `configs/scenarios.yaml`) still has room,
so every split covers every stratum. Before anything is written, the script
checks:
- stratum coverage: every cell filled in every split;
- no leakage: disjoint seeds, unique fragility/bed-occupancy seeds, no
  duplicate parameter vectors, and every val/test scenario at least
  `leakage_min_distance` from its nearest train scenario;
- the twin code and data the suite depends on are unchanged since the
  `TWIN_GIT_TAG` it claims to be generated on.
Any failure aborts with nothing written. An existing manifest is never
overwritten: a changed generator or twin needs a new `suite_version`.

Files are written read-only (chmod 444). `udt.scenarios.suite.load_suite`
verifies every file's SHA-256 against the manifest before use.

Usage:
  uv run python scripts/generate_scenarios.py --config configs/scenarios.yaml
"""

from __future__ import annotations

import argparse
import json
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
from scipy.stats import ks_2samp
from shapely.geometry import shape

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from _pipeline_common import configure_logging, resolve_path  # noqa: E402
from udt.common.models import Scenario  # noqa: E402
from udt.common.versions import ENV_VERSION, TWIN_GIT_TAG  # noqa: E402
from udt.scenarios.generator import DEFAULT_SEVERITY_RANGE, generate_flood_scenario  # noqa: E402
from udt.scenarios.suite import (  # noqa: E402
    FEATURE_NAMES,
    MANIFEST_NAME,
    SPLITS,
    feature_ranges,
    min_distance_to,
    normalised_features,
    scenario_features,
    sha256_file,
    stratum,
)

MAX_SEEDS_SCANNED = 100_000
# The twin code/data a flood suite depends on. Must match TWIN_GIT_TAG exactly.
TWIN_PATHS = (
    "src/udt/twin",
    "src/udt/incidents",
    "src/udt/scenarios/generator.py",
    "data/processed/flood_susceptibility.tif",
    "data/processed/dependency_graph.json",
    "data/processed/ward_boundary.geojson",
)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def sample_split(
    split: str,
    n: int,
    seed_start: int,
    ward_geojson: dict[str, Any],
    bands: list[tuple[float, float]],
    mid_lat: float,
    suite_version: str,
) -> list[Scenario]:
    cells = [(i, h) for i in range(len(bands)) for h in ("north", "south")]
    if n % len(cells) != 0:
        raise SystemExit(f"{split}: n={n} is not a multiple of {len(cells)} strata")
    per_cell = n // len(cells)
    counts = dict.fromkeys(cells, 0)
    chosen: list[Scenario] = []
    for seed in range(seed_start, seed_start + MAX_SEEDS_SCANNED):
        scenario = generate_flood_scenario(
            scenario_id=f"flood_{split}_{len(chosen):02d}",
            ward_boundary_geojson=ward_geojson,
            seed=seed,
        )
        cell = stratum(scenario, bands, mid_lat)
        if cell is None or counts[cell] >= per_cell:
            continue
        counts[cell] += 1
        scenario.suite_version = suite_version
        chosen.append(scenario)
        if len(chosen) == n:
            return chosen
    raise SystemExit(f"{split}: strata not filled within {MAX_SEEDS_SCANNED} seeds")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/scenarios.yaml")
    args = parser.parse_args()
    log = configure_logging()

    cfg = yaml.safe_load(resolve_path(args.config).read_text())
    out_dir = resolve_path(cfg["out_dir"])
    if (out_dir / MANIFEST_NAME).exists():
        raise SystemExit(
            f"{out_dir / MANIFEST_NAME} exists — the suite is frozen. "
            "Bump suite_version and out_dir for a new suite; never regenerate in place."
        )

    # --- the suite must be generated on exactly the tagged twin ---
    tag_commit = _git("rev-list", "-n", "1", TWIN_GIT_TAG)
    drift = _git("diff", "--name-only", TWIN_GIT_TAG, "--", *TWIN_PATHS)
    uncommitted = _git("status", "--porcelain", "--", *TWIN_PATHS)
    if drift or uncommitted:
        raise SystemExit(f"twin code/data differ from {TWIN_GIT_TAG}:\n{drift}\n{uncommitted}")

    processed = REPO_ROOT / "data" / "processed"
    ward_geojson = json.loads((processed / "ward_boundary.geojson").read_text())
    ward = shape(ward_geojson["features"][0]["geometry"])
    mid_lat = (ward.bounds[1] + ward.bounds[3]) / 2
    bands = [tuple(b) for b in cfg["stratify"]["severity_bands"]]

    splits: dict[str, list[Scenario]] = {
        name: sample_split(
            name,
            spec["n"],
            spec["seed_start"],
            ward_geojson,
            bands,  # type: ignore[arg-type]
            mid_lat,
            cfg["suite_version"],
        )
        for name, spec in cfg["splits"].items()
    }

    # --- checks (all must pass before anything is written) ---
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
        name: sorted({f"band{c[0]}_{c[1]}" for s in sc if (c := stratum(s, bands, mid_lat))})
        for name, sc in splits.items()
    }
    checks["all_strata_covered"] = all(len(v) == len(bands) * 2 for v in coverage.values())
    ks = {}
    for feat in FEATURE_NAMES:
        tr = [scenario_features(s)[feat] for s in splits["train"]]
        for name in ("val", "test"):
            other = [scenario_features(s)[feat] for s in splits[name]]
            ks[f"{feat}_train_vs_{name}"] = round(float(ks_2samp(tr, other).pvalue), 3)
    checks["ks_pvalues_train_vs_holdout"] = ks  # informational: small n, same generator
    per_split_ranges = {
        name: {
            feat: [
                round(min(scenario_features(s)[feat] for s in sc), 4),
                round(max(scenario_features(s)[feat] for s in sc), 4),
            ]
            for feat in FEATURE_NAMES
        }
        for name, sc in splits.items()
    }
    must_pass = [
        "seeds_disjoint_across_splits",
        "fragility_seeds_unique",
        "bed_occupancy_seeds_unique",
        "no_duplicate_parameter_vectors",
        "no_near_duplicates",
        "all_strata_covered",
    ]
    failed = [k for k in must_pass if not checks[k]]
    if failed:
        raise SystemExit(f"suite checks failed, nothing written: {failed}\n{checks}")

    # --- write to a temp dir, then move into place; files read-only ---
    tmp = Path(tempfile.mkdtemp(prefix="suite_", dir=out_dir.parent))
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
        "env_version": ENV_VERSION,
        "twin_git_tag": TWIN_GIT_TAG,
        "twin_git_commit": tag_commit,
        "created_at": datetime.now(UTC).isoformat(),
        "config": cfg,
        "splits": {name: {"n": len(sc), "seeds": seeds[name]} for name, sc in splits.items()},
        "strata_covered": coverage,
        "parameter_ranges_per_split": per_split_ranges,
        "checks": checks,
        "files": files,
    }
    (tmp / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    for p in tmp.rglob("*.json"):
        os.chmod(p, 0o444)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    if out_dir.exists():
        shutil.rmtree(out_dir)  # only reachable if a partial dir exists without a manifest
    tmp.rename(out_dir)
    log.info("suite_written", out_dir=str(out_dir), **{k: len(v) for k, v in splits.items()})
    print(json.dumps({"checks": checks, "strata_covered": coverage}, indent=2))


if __name__ == "__main__":
    assert set(SPLITS) == {"train", "val", "test"}
    main()
