"""Phase 4 acceptance tests: `twin.power` (dev doc §5.6's load-shedding
rule, enactment half) and `agents.rule_based._pick_load_shed` (decision
half)."""

from __future__ import annotations

import networkx as nx
import pytest

from udt.agents.rule_based import _pick_load_shed
from udt.common.models import Asset, AssetType, Incident
from udt.twin.power import (
    apply_overload_damage,
    post_shed_ratio,
    update_substation_load,
)

POINT = {"type": "Point", "coordinates": [72.88, 19.07]}

FLOOD_INCIDENT = Incident(
    incident_id="power_test",
    type="flood",
    location=POINT,
    onset_tick=0,
    severity=1.0,
    directly_affected_assets=[],
)


def _substation_graph(capacity_mw: float, load_mw: float, shed_tier: int = 0) -> nx.DiGraph:
    substation = Asset(
        asset_id="S1",
        asset_type=AssetType.SUBSTATION,
        geometry=POINT,
        attributes={"capacity_mw": capacity_mw, "load_mw": load_mw, "shed_tier": shed_tier},
    )
    g = nx.DiGraph()
    g.add_node("S1", asset=substation)
    return g


@pytest.mark.phase4
def test_post_shed_ratio_reduces_with_higher_tier() -> None:
    r0 = post_shed_ratio(load_mw=100.0, capacity_mw=100.0, shed_tier=0)
    r3 = post_shed_ratio(load_mw=100.0, capacity_mw=100.0, shed_tier=3)
    assert r0 == pytest.approx(1.0)
    assert r3 == pytest.approx(0.4)  # tier 3 sheds 60% (dev doc §3.2 exact)


@pytest.mark.phase4
def test_update_substation_load_surges_with_severity_at_peak() -> None:
    g = _substation_graph(capacity_mw=50.0, load_mw=30.0)
    update_substation_load(g, tick=60, incident=FLOOD_INCIDENT, dt_minutes=5.0)  # hold window
    attrs = g.nodes["S1"]["asset"].attributes
    assert attrs["load_mw_base"] == pytest.approx(30.0)
    assert attrs["load_mw"] > 30.0  # severity=1.0, envelope=1.0 -> full surge applied


@pytest.mark.phase4
def test_update_substation_load_no_surge_before_onset() -> None:
    g = _substation_graph(capacity_mw=50.0, load_mw=30.0)
    update_substation_load(g, tick=0, incident=FLOOD_INCIDENT, dt_minutes=5.0)
    assert g.nodes["S1"]["asset"].attributes["load_mw"] == pytest.approx(30.0)


@pytest.mark.phase4
def test_apply_overload_damage_only_when_over_threshold() -> None:
    overloaded = _substation_graph(capacity_mw=50.0, load_mw=49.0)  # 98% > 95%
    healthy = _substation_graph(capacity_mw=50.0, load_mw=30.0)  # 60% < 95%
    assert apply_overload_damage(overloaded) == {"S1": pytest.approx(0.05)}
    assert apply_overload_damage(healthy) == {}


@pytest.mark.phase4
def test_apply_overload_damage_respects_current_shed_tier() -> None:
    # Same raw load/capacity as the overloaded case above, but tier-3
    # shedding (60% off) brings it back under threshold.
    g = _substation_graph(capacity_mw=50.0, load_mw=49.0, shed_tier=3)
    assert apply_overload_damage(g) == {}


@pytest.mark.phase4
def test_pick_load_shed_escalates_when_overloaded() -> None:
    g = _substation_graph(capacity_mw=50.0, load_mw=49.0, shed_tier=0)  # 98% > 95%
    assert _pick_load_shed(g) == {"S1": 1}


@pytest.mark.phase4
def test_pick_load_shed_deescalates_when_comfortable() -> None:
    g = _substation_graph(
        capacity_mw=50.0, load_mw=20.0, shed_tier=2
    )  # well under DESHED_THRESHOLD
    assert _pick_load_shed(g) == {"S1": 1}


@pytest.mark.phase4
def test_pick_load_shed_no_change_in_the_middle_band() -> None:
    # Between DESHED_THRESHOLD (0.80) and OVERLOAD_THRESHOLD (0.95) -
    # deliberately stable, no flapping. load=54, tier=1 (20% shed) ->
    # post-shed ratio = 54 x 0.8 / 50 = 0.864.
    g = _substation_graph(capacity_mw=50.0, load_mw=54.0, shed_tier=1)
    assert _pick_load_shed(g) is None


@pytest.mark.phase4
def test_pick_load_shed_never_exceeds_max_tier() -> None:
    # Raw load so high that even tier 3's 60% shed leaves it over
    # threshold (130 x 0.4 / 50 = 1.04 > 0.95) — already at max tier, so
    # there's nothing left to escalate to; returns None, not tier 4.
    g = _substation_graph(capacity_mw=50.0, load_mw=130.0, shed_tier=3)
    assert _pick_load_shed(g) is None
