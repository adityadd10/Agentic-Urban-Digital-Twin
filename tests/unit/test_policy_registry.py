"""Phase 6 acceptance test: `agents.marl.registry` (dev doc §5.5's
`registry/policies.yaml` read/write), module M6b."""

from __future__ import annotations

from pathlib import Path

import pytest

from udt.agents.marl.registry import append_entry, load_registry
from udt.common.models import PolicyRegistryEntry


def _entry(policy_path: str) -> PolicyRegistryEntry:
    return PolicyRegistryEntry(
        incident_type="flood",
        policy_path=policy_path,
        env_version="udt_multi_env_v0",
        suite_version="none-n1-scenario",
        val_score=-123.4,
        val_score_is_true_holdout=False,
        trained_date="2026-09-11T00:00:00+00:00",
    )


@pytest.mark.phase6
def test_load_registry_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_registry(tmp_path / "does_not_exist.yaml") == []


@pytest.mark.phase6
def test_append_entry_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "policies.yaml"
    append_entry(path, _entry("run_a/model.pt"))
    entries = load_registry(path)
    assert len(entries) == 1
    assert entries[0].policy_path == "run_a/model.pt"
    assert entries[0].val_score_is_true_holdout is False


@pytest.mark.phase6
def test_append_entry_preserves_earlier_entries(tmp_path: Path) -> None:
    path = tmp_path / "policies.yaml"
    append_entry(path, _entry("run_a/model.pt"))
    append_entry(path, _entry("run_b/model.pt"))
    entries = load_registry(path)
    assert [e.policy_path for e in entries] == ["run_a/model.pt", "run_b/model.pt"]
