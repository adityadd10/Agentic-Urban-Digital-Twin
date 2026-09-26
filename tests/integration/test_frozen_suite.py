"""The frozen scenario suite (dev doc §4.3): integrity, versioning, split hygiene.

Runs against the committed `data/scenarios/flood` suite, so any edit to a
scenario file, or a twin/env version bump without a new suite, fails CI.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from udt.common.versions import ENV_VERSION, SUITE_VERSION
from udt.scenarios.suite import (
    SPLITS,
    SuiteIntegrityError,
    load_manifest,
    load_suite,
    verify_suite,
)

SUITE_DIR = Path(__file__).resolve().parents[2] / "data" / "scenarios" / "flood"


@pytest.mark.phase3
def test_committed_suite_matches_its_manifest() -> None:
    assert verify_suite(SUITE_DIR) == []


@pytest.mark.phase3
def test_suite_versions_match_running_code() -> None:
    manifest = load_manifest(SUITE_DIR)
    assert manifest["env_version"] == ENV_VERSION
    assert manifest["suite_version"] == SUITE_VERSION
    for split in SPLITS:
        assert all(s.suite_version == SUITE_VERSION for s in load_suite(SUITE_DIR, split))


@pytest.mark.phase3
def test_split_sizes_and_disjoint_seeds() -> None:
    splits = {split: load_suite(SUITE_DIR, split) for split in SPLITS}
    assert {k: len(v) for k, v in splits.items()} == {"train": 20, "val": 10, "test": 10}
    seeds = [s.seed for v in splits.values() for s in v]
    assert len(seeds) == len(set(seeds))
    ids = [s.scenario_id for v in splits.values() for s in v]
    assert len(ids) == len(set(ids))


@pytest.mark.phase3
def test_tampering_is_detected(tmp_path: Path) -> None:
    copy = tmp_path / "flood"
    shutil.copytree(SUITE_DIR, copy)
    target = next((copy / "test").glob("*.json"))
    os.chmod(target, 0o644)
    target.write_text(target.read_text().replace('"severity": 0.', '"severity": 0.9', 1))
    problems = verify_suite(copy)
    assert any("hash mismatch" in p for p in problems)
    with pytest.raises(SuiteIntegrityError):
        load_suite(copy, "test")


@pytest.mark.phase3
def test_suite_for_another_env_version_is_refused() -> None:
    with pytest.raises(SuiteIntegrityError, match="env_version"):
        load_suite(SUITE_DIR, "train", env_version="udt_multi_env_v0")
