"""Phase 4 acceptance tests: `RuleBasedAgent`/`DoNothingAgent` (dev doc
§5.6, repair-crew-only scope — see `udt.agents.rule_based`'s docstring)."""

from __future__ import annotations

import pytest

from udt.agents.rule_based import DoNothingAgent, RuleBasedAgent
from udt.common.models import Asset, AssetType, DependencyGraph
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
