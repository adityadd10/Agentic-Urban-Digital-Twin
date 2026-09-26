"""`agents/plan_interpreter.py` (dev doc §7.3.1), added 2026-09-22 to
close the gap where `Plan.goal_weights`/`priority_assets`/`directives`
had no executable effect anywhere in the codebase (`MTP_Module_Planner.
md`'s 2026-09-22 M9a entry) — every candidate plan simulated identically,
and live execution ignored a selected plan's priorities outright.

Same fixture conventions as `test_rule_based_agent.py`/
`test_patient_transfer.py`/`test_power.py`/`test_ambulances.py`, since
this module wraps exactly those four rules — a small local `RoadNetwork`
chain (`test_ambulances.py`'s own pattern, no file loading needed) rather
than a fake/mock."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from udt.agents.plan_interpreter import SHED_BIAS_DELTA, interpret_plan
from udt.agents.rule_based import RuleBasedAgent
from udt.common.models import Asset, AssetType, DependencyEdge, DependencyGraph
from udt.llm.schemas import Directive, GoalWeights, Plan
from udt.twin.cascade import EdgeRuntimeState
from udt.twin.graph import build_networkx_graph
from udt.twin.power import DESHED_THRESHOLD, OVERLOAD_THRESHOLD
from udt.twin.road_network import RoadNetwork

POINT_A = {"type": "Point", "coordinates": [72.88, 19.07]}
POINT_B = {"type": "Point", "coordinates": [72.90, 19.07]}

DEFAULT_WEIGHTS = GoalWeights(g_health=1.0, g_power=1.0, g_transport=1.0, g_cost=1.0)


def _empty_plan(plan_id: str = "p") -> Plan:
    return Plan(
        plan_id=plan_id,
        objective="x",
        goal_weights=DEFAULT_WEIGHTS,
        priority_assets=[],
        directives=[],
        rationale="x",
        expected_outcome="x",
    )


def _hospital(asset_id: str, geometry: dict, level: float = 1.0, beds_occupied: int = 10) -> Asset:
    return Asset(
        asset_id=asset_id,
        asset_type=AssetType.HOSPITAL,
        geometry=geometry,
        intrinsic_level=level,
        functional_level=level,
        attributes={"beds_total": 100, "beds_occupied": beds_occupied},
    )


def _substation(
    asset_id: str, geometry: dict, capacity_mw: float, load_mw: float, shed_tier: int = 0
) -> Asset:
    return Asset(
        asset_id=asset_id,
        asset_type=AssetType.SUBSTATION,
        geometry=geometry,
        attributes={"capacity_mw": capacity_mw, "load_mw": load_mw, "shed_tier": shed_tier},
    )


def _road_network_chain(n_edges: int, length_m: float = 300.0) -> RoadNetwork:
    """Same helper as `test_ambulances.py`'s own — a tiny synthetic chain,
    no file loading. Duplicated rather than imported (a private,
    test-local helper in another test file, same "duplicate a small
    thing" convention this codebase already uses elsewhere)."""
    g: nx.DiGraph[str] = nx.DiGraph()
    for i in range(n_edges + 1):
        g.add_node(str(i), x=72.88 + i * 0.001, y=19.07)
    for i in range(n_edges):
        g.add_edge(str(i), str(i + 1), length=length_m, susceptibility=0.0)
        g.add_edge(str(i + 1), str(i), length=length_m, susceptibility=0.0)
    node_ids = list(g.nodes)
    node_xy = np.array([[float(g.nodes[n]["x"]), float(g.nodes[n]["y"])] for n in node_ids])
    rn = RoadNetwork(g, node_ids, node_xy)
    return rn


# ---------------------------------------------------------------------------
# Degenerate case: empty plan == the rule-based baseline
# ---------------------------------------------------------------------------
@pytest.mark.phase9
def test_empty_plan_matches_rule_based_baseline() -> None:
    """The free correctness check this module's own docstring promises:
    no priority assets, no directives, default weights -> identical
    decision to `RuleBasedAgent.act`."""
    graph_model = DependencyGraph(
        assets=[
            _hospital("H1", POINT_A, level=0.4),
            _substation("S1", POINT_B, capacity_mw=50.0, load_mw=49.0),  # 98% > 95%
        ],
        edges=[],
    )
    g = build_networkx_graph(graph_model)
    baseline_action = RuleBasedAgent().act(g, tick=0, road_network=None, edge_states=None)
    interpreted = interpret_plan(_empty_plan(), g, road_network=None, edge_states=None)
    assert interpreted == baseline_action
    assert interpreted.repair_target == "H1"
    assert interpreted.shed_tier == {"S1": 1}


# ---------------------------------------------------------------------------
# protect_repair / priority_assets
# ---------------------------------------------------------------------------
@pytest.mark.phase9
def test_protect_repair_directive_redirects_from_worse_asset() -> None:
    graph_model = DependencyGraph(
        assets=[_hospital("H1", POINT_A, level=0.2), _hospital("H2", POINT_B, level=0.6)],
        edges=[],
    )
    g = build_networkx_graph(graph_model)
    # Baseline would pick H1 (worse).
    assert RuleBasedAgent().act(g, tick=0).repair_target == "H1"

    plan = Plan(
        plan_id="p",
        objective="x",
        goal_weights=DEFAULT_WEIGHTS,
        priority_assets=[],
        directives=[Directive(kind="protect_repair", details={"asset_id": "H2"})],
        rationale="x",
        expected_outcome="x",
    )
    assert interpret_plan(plan, g, road_network=None, edge_states=None).repair_target == "H2"


@pytest.mark.phase9
def test_protect_repair_directive_falls_back_if_asset_is_healthy() -> None:
    graph_model = DependencyGraph(
        assets=[_hospital("H1", POINT_A, level=0.2), _hospital("H2", POINT_B, level=1.0)],
        edges=[],
    )
    g = build_networkx_graph(graph_model)
    plan = Plan(
        plan_id="p",
        objective="x",
        goal_weights=DEFAULT_WEIGHTS,
        priority_assets=[],
        directives=[Directive(kind="protect_repair", details={"asset_id": "H2"})],  # healthy
        rationale="x",
        expected_outcome="x",
    )
    # H2 isn't damaged, so the directive doesn't fire - baseline (H1) stands.
    assert interpret_plan(plan, g, road_network=None, edge_states=None).repair_target == "H1"


@pytest.mark.phase9
def test_priority_assets_biases_repair_without_a_directive() -> None:
    graph_model = DependencyGraph(
        assets=[_hospital("H1", POINT_A, level=0.2), _hospital("H2", POINT_B, level=0.6)],
        edges=[],
    )
    g = build_networkx_graph(graph_model)
    plan = Plan(
        plan_id="p",
        objective="x",
        goal_weights=DEFAULT_WEIGHTS,
        priority_assets=["H2"],
        directives=[],
        rationale="x",
        expected_outcome="x",
    )
    assert interpret_plan(plan, g, road_network=None, edge_states=None).repair_target == "H2"


@pytest.mark.phase9
def test_protect_repair_directive_takes_precedence_over_priority_assets() -> None:
    graph_model = DependencyGraph(
        assets=[
            _hospital("H1", POINT_A, level=0.2),
            _hospital("H2", POINT_B, level=0.5),
            _hospital("H3", POINT_A, level=0.6),
        ],
        edges=[],
    )
    g = build_networkx_graph(graph_model)
    plan = Plan(
        plan_id="p",
        objective="x",
        goal_weights=DEFAULT_WEIGHTS,
        priority_assets=["H2"],
        directives=[Directive(kind="protect_repair", details={"asset_id": "H3"})],
        rationale="x",
        expected_outcome="x",
    )
    assert interpret_plan(plan, g, road_network=None, edge_states=None).repair_target == "H3"


# ---------------------------------------------------------------------------
# protect_transfer
# ---------------------------------------------------------------------------
@pytest.mark.phase9
def test_protect_transfer_directive_fires_preemptively() -> None:
    """Unlike the baseline rule (buffer-threshold triggered), a
    `protect_transfer` directive moves patients out as long as it's
    physically feasible, even with a healthy power buffer."""
    substation = Asset(
        asset_id="S1",
        asset_type=AssetType.SUBSTATION,
        geometry=POINT_A,
        attributes={"capacity_mw": 30.0},
    )
    h1 = _hospital("H1", POINT_A, beds_occupied=20)
    h2 = _hospital("H2", POINT_B, beds_occupied=0)
    edge = DependencyEdge(
        edge_id="E1_power",
        supplier="S1",
        consumer="H1",
        kind="power",
        demand=0.1,
        criticality=0.9,
        buffer_hours=8.0,
        floor=0.3,
    )
    g = build_networkx_graph(DependencyGraph(assets=[substation, h1, h2], edges=[edge]))
    edge_states = {"E1_power": EdgeRuntimeState(capacity_hours=8.0, remaining_hours=8.0)}  # healthy

    # Baseline: buffer is healthy, nothing to do.
    assert RuleBasedAgent().act(g, tick=0, edge_states=edge_states).patient_transfer is None

    plan = Plan(
        plan_id="p",
        objective="x",
        goal_weights=DEFAULT_WEIGHTS,
        priority_assets=[],
        directives=[Directive(kind="protect_transfer", details={"hospital_id": "H1"})],
        rationale="x",
        expected_outcome="x",
    )
    result = interpret_plan(plan, g, road_network=None, edge_states=edge_states)
    assert result.patient_transfer is not None
    from_id, to_id, _count = result.patient_transfer
    assert (from_id, to_id) == ("H1", "H2")


@pytest.mark.phase9
def test_protect_transfer_directive_falls_back_when_infeasible() -> None:
    h1 = _hospital("H1", POINT_A, beds_occupied=20)
    h2 = _hospital("H2", POINT_B, beds_occupied=100)  # full - no destination
    g = build_networkx_graph(DependencyGraph(assets=[h1, h2], edges=[]))
    plan = Plan(
        plan_id="p",
        objective="x",
        goal_weights=DEFAULT_WEIGHTS,
        priority_assets=[],
        directives=[Directive(kind="protect_transfer", details={"hospital_id": "H1"})],
        rationale="x",
        expected_outcome="x",
    )
    # No edge_states -> baseline can't trigger either -> both None.
    assert interpret_plan(plan, g, road_network=None, edge_states=None).patient_transfer is None


# ---------------------------------------------------------------------------
# shed_bias / goal_weights
# ---------------------------------------------------------------------------
@pytest.mark.phase9
def test_shed_bias_early_sheds_before_the_baseline_would() -> None:
    # Ratio strictly between (OVERLOAD_THRESHOLD - SHED_BIAS_DELTA) and
    # OVERLOAD_THRESHOLD: baseline does nothing, "early" bias sheds.
    ratio_target = OVERLOAD_THRESHOLD - SHED_BIAS_DELTA / 2
    load_mw = ratio_target * 50.0
    g = build_networkx_graph(
        DependencyGraph(assets=[_substation("S1", POINT_A, 50.0, load_mw)], edges=[])
    )
    assert RuleBasedAgent().act(g, tick=0).shed_tier is None

    plan = Plan(
        plan_id="p",
        objective="x",
        goal_weights=DEFAULT_WEIGHTS,
        priority_assets=[],
        directives=[Directive(kind="shed_bias", details={"direction": "early"})],
        rationale="x",
        expected_outcome="x",
    )
    assert interpret_plan(plan, g, road_network=None, edge_states=None).shed_tier == {"S1": 1}


@pytest.mark.phase9
def test_shed_bias_late_delays_shedding_the_baseline_would_do() -> None:
    # Ratio strictly between OVERLOAD_THRESHOLD and (OVERLOAD_THRESHOLD +
    # SHED_BIAS_DELTA): baseline sheds, "late" bias doesn't.
    ratio_target = OVERLOAD_THRESHOLD + SHED_BIAS_DELTA / 2
    load_mw = ratio_target * 50.0
    g = build_networkx_graph(
        DependencyGraph(assets=[_substation("S1", POINT_A, 50.0, load_mw)], edges=[])
    )
    assert RuleBasedAgent().act(g, tick=0).shed_tier == {"S1": 1}

    plan = Plan(
        plan_id="p",
        objective="x",
        goal_weights=DEFAULT_WEIGHTS,
        priority_assets=[],
        directives=[Directive(kind="shed_bias", details={"direction": "late"})],
        rationale="x",
        expected_outcome="x",
    )
    assert interpret_plan(plan, g, road_network=None, edge_states=None).shed_tier is None


@pytest.mark.phase9
def test_high_g_power_nudges_shed_earlier_without_a_directive() -> None:
    ratio_target = OVERLOAD_THRESHOLD - SHED_BIAS_DELTA / 2
    load_mw = ratio_target * 50.0
    g = build_networkx_graph(
        DependencyGraph(assets=[_substation("S1", POINT_A, 50.0, load_mw)], edges=[])
    )
    high_power_weights = GoalWeights(g_health=1.0, g_power=2.0, g_transport=1.0, g_cost=1.0)
    plan = Plan(
        plan_id="p",
        objective="x",
        goal_weights=high_power_weights,
        priority_assets=[],
        directives=[],
        rationale="x",
        expected_outcome="x",
    )
    assert interpret_plan(plan, g, road_network=None, edge_states=None).shed_tier == {"S1": 1}


@pytest.mark.phase9
def test_shed_bias_directive_overrides_goal_weight_nudge() -> None:
    # g_power=2.0 alone would nudge toward "early" (shed at this ratio);
    # an explicit "late" directive must win instead.
    ratio_target = OVERLOAD_THRESHOLD - SHED_BIAS_DELTA / 2
    load_mw = ratio_target * 50.0
    g = build_networkx_graph(
        DependencyGraph(assets=[_substation("S1", POINT_A, 50.0, load_mw)], edges=[])
    )
    high_power_weights = GoalWeights(g_health=1.0, g_power=2.0, g_transport=1.0, g_cost=1.0)
    plan = Plan(
        plan_id="p",
        objective="x",
        goal_weights=high_power_weights,
        priority_assets=[],
        directives=[Directive(kind="shed_bias", details={"direction": "late"})],
        rationale="x",
        expected_outcome="x",
    )
    assert interpret_plan(plan, g, road_network=None, edge_states=None).shed_tier is None


@pytest.mark.phase9
def test_shed_bias_never_inverts_the_hysteresis_band() -> None:
    """`OVERLOAD_THRESHOLD` (0.95) and `DESHED_THRESHOLD` (0.80) stay 0.15
    apart; the disclosed bias (max ~0.03 combined) can't cross them."""
    assert OVERLOAD_THRESHOLD - DESHED_THRESHOLD > 2 * SHED_BIAS_DELTA


# ---------------------------------------------------------------------------
# protect_dispatch
# ---------------------------------------------------------------------------
@pytest.mark.phase9
def test_protect_dispatch_directive_prefers_request_nearest_named_hospital() -> None:
    rn = _road_network_chain(n_edges=4)  # nodes "0".."4" along a line
    hospital_near = Asset(
        asset_id="H_near",
        asset_type=AssetType.HOSPITAL,
        geometry={"type": "Point", "coordinates": [72.884, 19.07]},  # near node "4"
        attributes={"beds_total": 10, "beds_occupied": 0},
    )
    ambulance = Asset(
        asset_id="AMB1",
        asset_type=AssetType.AMBULANCE,
        geometry={"type": "Point", "coordinates": [72.88, 19.07]},
        attributes={"status": "idle", "home_node": "0"},
    )
    g = build_networkx_graph(DependencyGraph(assets=[hospital_near, ambulance], edges=[]))
    old_far_request = {
        "request_id": "OLD_FAR",
        "location": (72.88, 19.07),
        "requested_at_tick": 0,
    }
    new_near_request = {
        "request_id": "NEW_NEAR",
        "location": (72.884, 19.07),
        "requested_at_tick": 5,
    }
    g.graph["pending_requests"] = [old_far_request, new_near_request]

    # Baseline: oldest-first -> OLD_FAR, regardless of hospital proximity.
    baseline = RuleBasedAgent().act(g, tick=10, road_network=rn)
    assert baseline.ambulance_assignment == {"AMB1": "OLD_FAR"}

    plan = Plan(
        plan_id="p",
        objective="x",
        goal_weights=DEFAULT_WEIGHTS,
        priority_assets=[],
        directives=[Directive(kind="protect_dispatch", details={"hospital_id": "H_near"})],
        rationale="x",
        expected_outcome="x",
    )
    result = interpret_plan(plan, g, road_network=rn, edge_states=None)
    assert result.ambulance_assignment == {"AMB1": "NEW_NEAR"}


@pytest.mark.phase9
def test_protect_dispatch_falls_back_without_road_network() -> None:
    ambulance = Asset(
        asset_id="AMB1",
        asset_type=AssetType.AMBULANCE,
        geometry=POINT_A,
        attributes={"status": "idle", "home_node": "0"},
    )
    g = build_networkx_graph(DependencyGraph(assets=[ambulance], edges=[]))
    g.graph["pending_requests"] = [
        {"request_id": "R1", "location": (72.88, 19.07), "requested_at_tick": 0}
    ]
    plan = Plan(
        plan_id="p",
        objective="x",
        goal_weights=DEFAULT_WEIGHTS,
        priority_assets=[],
        directives=[Directive(kind="protect_dispatch", details={"hospital_id": "H1"})],
        rationale="x",
        expected_outcome="x",
    )
    # No road_network -> both the directive and the baseline are inert.
    assert interpret_plan(plan, g, road_network=None, edge_states=None).ambulance_assignment is None


# ---------------------------------------------------------------------------
# Unknown directive kinds
# ---------------------------------------------------------------------------
@pytest.mark.phase9
def test_unrecognized_directive_kind_is_silently_not_acted_on() -> None:
    g = build_networkx_graph(
        DependencyGraph(assets=[_hospital("H1", POINT_A, level=0.4)], edges=[])
    )
    plan = Plan(
        plan_id="p",
        objective="x",
        goal_weights=DEFAULT_WEIGHTS,
        priority_assets=[],
        directives=[Directive(kind="some_future_kind", details={"asset_id": "H1"})],
        rationale="x",
        expected_outcome="x",
    )
    # Degenerates to the baseline - the unknown kind is neither an error
    # nor silently repaired into something it isn't.
    baseline = RuleBasedAgent().act(g, tick=0)
    assert interpret_plan(plan, g, road_network=None, edge_states=None) == baseline
