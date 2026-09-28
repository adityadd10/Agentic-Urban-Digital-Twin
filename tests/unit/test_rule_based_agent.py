"""Phase 4 acceptance tests: `RuleBasedAgent`/`DoNothingAgent` (dev doc
§5.6, repair-crew-only scope — see `udt.agents.rule_based`'s docstring)."""

from __future__ import annotations

import pytest

from udt.agents.rule_based import DoNothingAgent, RuleBasedAgent, RuleBasedAgentV2
from udt.common.models import Asset, AssetType, DependencyEdge, DependencyGraph
from udt.twin.graph import build_networkx_graph

POINT = {"type": "Point", "coordinates": [72.88, 19.07]}


def _asset(asset_id: str, asset_type: AssetType, functional_level: float) -> Asset:
    return Asset(
        asset_id=asset_id,
        asset_type=asset_type,
        geometry=POINT,
        intrinsic_level=functional_level,
        functional_level=functional_level,
        attributes={},
    )


@pytest.mark.phase4
def test_rule_based_agent_targets_worst_repairable_asset() -> None:
    graph = DependencyGraph(
        assets=[
            _asset("H1", AssetType.HOSPITAL, 0.8),
            _asset("S1", AssetType.SUBSTATION, 0.3),  # worst
            _asset("H2", AssetType.HOSPITAL, 0.6),
            _asset("R1", AssetType.ROAD, 0.1),  # worse, but not repairable
        ],
        edges=[],
    )
    g = build_networkx_graph(graph)
    action = RuleBasedAgent().act(g, tick=0)
    assert action.repair_target == "S1"


@pytest.mark.phase4
def test_rule_based_agent_does_nothing_when_all_healthy() -> None:
    graph = DependencyGraph(
        assets=[
            _asset("H1", AssetType.HOSPITAL, 1.0),
            _asset("S1", AssetType.SUBSTATION, 1.0),
        ],
        edges=[],
    )
    g = build_networkx_graph(graph)
    action = RuleBasedAgent().act(g, tick=0)
    assert action.repair_target is None


@pytest.mark.phase4
def test_do_nothing_agent_never_acts() -> None:
    graph = DependencyGraph(assets=[_asset("S1", AssetType.SUBSTATION, 0.1)], edges=[])
    g = build_networkx_graph(graph)
    action = DoNothingAgent().act(g, tick=0)
    assert action.repair_target is None


# --- Rule-based v2 (dev doc §3.9) ---------------------------------------------


def _edge(supplier: str, consumer: str, kind: str) -> DependencyEdge:
    return DependencyEdge(
        edge_id=f"{supplier}_{consumer}",
        supplier=supplier,
        consumer=consumer,
        kind=kind,
        demand=0.1,
        criticality=0.9,
        buffer_hours=0.0,
        floor=0.3,
    )


def _starved_pump_graph() -> DependencyGraph:
    """The robustness R12 situation: the pump is physically intact but dark
    because its substation failed; the substation is the real damage."""
    pump = _asset("W1", AssetType.WATER, 0.0).model_copy(update={"intrinsic_level": 1.0})
    return DependencyGraph(
        assets=[
            _asset("S1", AssetType.SUBSTATION, 0.05),
            pump,
            _asset("H1", AssetType.HOSPITAL, 0.3),
        ],
        edges=[_edge("S1", "W1", "power"), _edge("W1", "H1", "water"), _edge("S1", "H1", "power")],
    )


@pytest.mark.phase4
def test_v1_sends_the_crew_to_an_intact_starved_facility() -> None:
    """Documents the v1 flaw that v2 fixes (robustness OUTCOME §2a)."""
    g = build_networkx_graph(_starved_pump_graph())
    assert RuleBasedAgent().act(g, tick=0).repair_target == "W1"


@pytest.mark.phase4
def test_v2_repairs_the_damaged_asset_most_others_depend_on() -> None:
    g = build_networkx_graph(_starved_pump_graph())
    assert RuleBasedAgentV2().act(g, tick=0).repair_target == "S1"


@pytest.mark.phase4
def test_v2_skips_intact_facilities_and_does_nothing_when_all_intact() -> None:
    graph = DependencyGraph(
        assets=[
            _asset("H1", AssetType.HOSPITAL, 0.2).model_copy(update={"intrinsic_level": 1.0}),
            _asset("S1", AssetType.SUBSTATION, 1.0),
        ],
        edges=[],
    )
    g = build_networkx_graph(graph)
    assert RuleBasedAgentV2().act(g, tick=0).repair_target is None
