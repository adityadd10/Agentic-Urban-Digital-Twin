"""Phase 10 acceptance tests: `logging.decision_log` (dev doc §10's
decision-log schema + JSONL writer), module M10a."""

from __future__ import annotations

from pathlib import Path

import pytest

from udt.common.models import (
    AgentAction,
    Asset,
    AssetType,
    AutonomyTier,
    ConstraintReport,
    DependencyGraph,
    RoutingDecision,
)
from udt.logging.decision_log import (
    ApprovalSummary,
    DecisionLogWriter,
    DecisionRecord,
    LLMCallSummary,
    MarlActionSummary,
    RiskSummary,
    build_decision_id,
    load_decision_log,
    twin_state_digest,
)
from udt.twin.graph import build_networkx_graph
from udt.twin.simulator import Simulator

POINT = {"type": "Point", "coordinates": [72.88, 19.07]}


def _sim(functional_level: float = 1.0) -> Simulator:
    hospital = Asset(
        asset_id="H1",
        asset_type=AssetType.HOSPITAL,
        geometry=POINT,
        intrinsic_level=functional_level,
        functional_level=functional_level,
        attributes={"beds_total": 10, "beds_occupied": 0},
    )
    return Simulator(DependencyGraph(assets=[hospital], edges=[]))


def _record(decision_id: str = "d1") -> DecisionRecord:
    return DecisionRecord(
        decision_id=decision_id,
        episode_id="ep1",
        tick=0,
        twin_state_digest="abc123",
        incident_id="inc1",
        routing=RoutingDecision(path="ID", registry_hit=True, support_score=0.9, reasons=["ok"]),
        llm=LLMCallSummary(prompt_version="v1", model="claude-sonnet-5", tokens=100),
        plans=[],
        selected_plan_id=None,
        marl=MarlActionSummary(policy_id="rule_based", action={}, masked_actions_count=0),
        constraint_report=ConstraintReport(
            passed=True, violations=[], repaired_action=AgentAction()
        ),
        risk=RiskSummary(
            p_failure=0.1, consequence=1.0, risk=0.05, confidence=0.9, tier=AutonomyTier.AUTONOMOUS
        ),
        approval=ApprovalSummary(required=False),
        executed_action=AgentAction(),
    )


@pytest.mark.phase10
def test_twin_state_digest_is_deterministic() -> None:
    sim = _sim()
    assert twin_state_digest(sim) == twin_state_digest(sim)


@pytest.mark.phase10
def test_twin_state_digest_changes_when_functional_level_changes() -> None:
    a = twin_state_digest(_sim(1.0))
    b = twin_state_digest(_sim(0.5))
    assert a != b


@pytest.mark.phase10
def test_twin_state_digest_is_independent_of_node_iteration_order() -> None:
    """The digest sorts by asset id internally - two graphs built with
    the same assets in a different order must hash identically."""
    h1 = Asset(asset_id="H1", asset_type=AssetType.HOSPITAL, geometry=POINT, attributes={})
    h2 = Asset(asset_id="H2", asset_type=AssetType.HOSPITAL, geometry=POINT, attributes={})
    forward = build_networkx_graph(DependencyGraph(assets=[h1, h2], edges=[]))
    backward = build_networkx_graph(DependencyGraph(assets=[h2, h1], edges=[]))
    sim_forward = Simulator.__new__(Simulator)
    sim_forward.graph = forward
    sim_backward = Simulator.__new__(Simulator)
    sim_backward.graph = backward
    assert twin_state_digest(sim_forward) == twin_state_digest(sim_backward)


@pytest.mark.phase10
def test_build_decision_id_is_stable_and_readable() -> None:
    assert build_decision_id("ep1", 12) == "ep1_tick12"


@pytest.mark.phase10
def test_decision_record_round_trips_through_json() -> None:
    record = _record()
    restored = DecisionRecord.model_validate_json(record.model_dump_json())
    assert restored == record


@pytest.mark.phase10
def test_decision_log_writer_appends_one_line_per_record(tmp_path: Path) -> None:
    log_path = tmp_path / "decisions.jsonl"
    writer = DecisionLogWriter(log_path)
    writer.append(_record("d1"))
    writer.append(_record("d2"))
    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 2


@pytest.mark.phase10
def test_decision_log_writer_creates_parent_directories(tmp_path: Path) -> None:
    log_path = tmp_path / "nested" / "dir" / "decisions.jsonl"
    DecisionLogWriter(log_path).append(_record())
    assert log_path.exists()


@pytest.mark.phase10
def test_load_decision_log_round_trips(tmp_path: Path) -> None:
    log_path = tmp_path / "decisions.jsonl"
    writer = DecisionLogWriter(log_path)
    writer.append(_record("d1"))
    writer.append(_record("d2"))
    loaded = load_decision_log(log_path)
    assert [r.decision_id for r in loaded] == ["d1", "d2"]


@pytest.mark.phase10
def test_load_decision_log_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_decision_log(tmp_path / "does_not_exist.jsonl") == []


@pytest.mark.phase10
def test_decision_log_is_append_only_never_rewritten(tmp_path: Path) -> None:
    """dev doc §10: "Append-only JSONL per episode" - appending a third
    record must never disturb the first two already-written lines."""
    log_path = tmp_path / "decisions.jsonl"
    writer = DecisionLogWriter(log_path)
    writer.append(_record("d1"))
    first_write = log_path.read_text()
    writer.append(_record("d2"))
    assert log_path.read_text().startswith(first_write)
