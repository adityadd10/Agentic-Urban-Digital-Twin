"""Frozen scenario suite (dev doc §4.3): loading, integrity checks, strata, leakage.

The suite lives in `data/scenarios/<type>/{train,val,test}/*.json` with a
`manifest.json` that records each file's SHA-256, the seeds, the suite and env
versions, and the git commit/tag of the twin it was generated on.
`load_suite` refuses to return scenarios whose files don't match the manifest,
or whose `env_version` differs from the running code's `ENV_VERSION`: a suite
generated on other physics is not "the same test set".

`scripts/generate_scenarios.py` writes the suite; this module holds the parts
that both the generator and the harnesses need.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from udt.common.models import Scenario
from udt.common.versions import ENV_VERSION
from udt.scenarios.generator import (
    FOOTPRINT_SIGMA_RANGE_M,
    GROWTH_HOURS_RANGE,
    HOLD_HOURS_RANGE,
    RECEDE_HOURS_RANGE,
)

SPLITS = ("train", "val", "test")
MANIFEST_NAME = "manifest.json"
FEATURE_NAMES = ("severity", "sigma_m", "lon", "lat", "growth_h", "hold_h", "recede_h")


class SuiteIntegrityError(RuntimeError):
    """The suite on disk doesn't match its manifest, or targets another env version."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(suite_dir: str | Path) -> dict[str, Any]:
    path = Path(suite_dir) / MANIFEST_NAME
    if not path.exists():
        raise SuiteIntegrityError(f"no {MANIFEST_NAME} in {suite_dir} — suite not generated")
    manifest: dict[str, Any] = json.loads(path.read_text())
    return manifest


def verify_suite(suite_dir: str | Path, *, env_version: str = ENV_VERSION) -> list[str]:
    """Every problem found (empty list = suite intact): hash mismatches, missing
    or extra files, env-version mismatch."""
    suite_dir = Path(suite_dir)
    manifest = load_manifest(suite_dir)
    problems: list[str] = []
    if manifest.get("env_version") != env_version:
        problems.append(
            f"suite env_version {manifest.get('env_version')!r} != running {env_version!r}"
        )
    recorded = {f["path"]: f["sha256"] for f in manifest["files"]}
    on_disk = {
        str(p.relative_to(suite_dir))
        for split in SPLITS
        for p in (suite_dir / split).glob("*.json")
    }
    for rel in sorted(set(recorded) - on_disk):
        problems.append(f"missing file {rel}")
    for rel in sorted(on_disk - set(recorded)):
        problems.append(f"file not in manifest {rel}")
    for rel in sorted(set(recorded) & on_disk):
        if sha256_file(suite_dir / rel) != recorded[rel]:
            problems.append(f"hash mismatch {rel}")
    return problems


def load_suite(
    suite_dir: str | Path, split: str, *, env_version: str = ENV_VERSION
) -> list[Scenario]:
    """The scenarios of one split, after verifying the whole suite."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    problems = verify_suite(suite_dir, env_version=env_version)
    if problems:
        raise SuiteIntegrityError("; ".join(problems))
    files = sorted((Path(suite_dir) / split).glob("*.json"))
    return [Scenario.model_validate_json(f.read_text()) for f in files]


def scenario_features(scenario: Scenario) -> dict[str, float]:
    p = scenario.incident.profile
    return {
        "severity": scenario.incident.severity,
        "sigma_m": float(p["footprint"]["sigma_m"]),
        "lon": float(p["footprint"]["lon"]),
        "lat": float(p["footprint"]["lat"]),
        "growth_h": float(p["growth_hours"]),
        "hold_h": float(p["hold_hours"]),
        "recede_h": float(p["recede_hours"]),
    }


def feature_ranges(
    ward_bounds: tuple[float, float, float, float], severity_range: tuple[float, float]
) -> dict[str, tuple[float, float]]:
    minx, miny, maxx, maxy = ward_bounds
    return {
        "severity": severity_range,
        "sigma_m": FOOTPRINT_SIGMA_RANGE_M,
        "lon": (minx, maxx),
        "lat": (miny, maxy),
        "growth_h": GROWTH_HOURS_RANGE,
        "hold_h": HOLD_HOURS_RANGE,
        "recede_h": RECEDE_HOURS_RANGE,
    }


def normalised_features(
    scenario: Scenario, ranges: dict[str, tuple[float, float]]
) -> np.ndarray[Any, np.dtype[np.float64]]:
    f = scenario_features(scenario)
    return np.array([(f[k] - ranges[k][0]) / (ranges[k][1] - ranges[k][0]) for k in FEATURE_NAMES])


def stratum(
    scenario: Scenario, severity_bands: list[tuple[float, float]], mid_lat: float
) -> tuple[int, str] | None:
    """(severity band index, "north"/"south") or None if outside every band."""
    sev = scenario.incident.severity
    half = "north" if float(scenario.incident.profile["footprint"]["lat"]) >= mid_lat else "south"
    for i, (lo, hi) in enumerate(severity_bands):
        last = i == len(severity_bands) - 1
        if lo <= sev < hi or (last and sev == hi):
            return i, half
    return None


def min_distance_to(
    query: list[Scenario], reference: list[Scenario], ranges: dict[str, tuple[float, float]]
) -> float:
    """Smallest normalised Euclidean distance from any `query` scenario to any
    `reference` scenario (leakage check: near-duplicates across splits)."""
    ref = np.stack([normalised_features(s, ranges) for s in reference])
    return float(
        min(np.min(np.linalg.norm(ref - normalised_features(s, ranges), axis=1)) for s in query)
    )
