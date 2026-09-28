"""Tick loop (dev doc §3.1, §3.5, module M1).

Orchestrates one simulated tick (default dt = 5 minutes, dev doc §3.1):
apply exogenous degradation, drain/recover buffers, resolve functional
levels to a fixed point, emit a `TwinState` snapshot.

Prototype-1 scope: step 4 of §3.5 ("consume demand: patients arrive,
treatments complete, ambulances advance, queues update") was originally
skipped entirely — no agents existed yet to act on patient/ambulance
state. M4 additions 2-4 (below) implement it in full for this
prototype's scope. `twin/counterfactual.py`'s `simulate()` (§3.7) is
still deferred to when an agent needs counterfactual rollouts.

M4 addition 1: `apply_repair`/`step`'s `repair_target` param. Nothing
before M4 ever *increased* `intrinsic_level` — only incidents existed, and
they only damage. Now that dev doc §5.6's rule-based agent needs to "send
a repair crew to the highest-criticality failed [facility]", `step`
accepts an optional single repair target per tick, applied before the
buffer/fixed-point resolution (same slot as exogenous degradation).

M4 addition 2: step 4 of §3.5 ("consume demand") is no longer skipped for
hospitals — `twin/demand.py`'s `consume_demand` now runs every tick,
after the fixed-point resolution (per §3.5's own step ordering). This
makes the twin stochastic for the first time (patient arrivals/discharges
are randomly sampled), so `Simulator` now owns its own seeded
`np.random.Generator` (dev doc §3.6: "any function that uses randomness
takes an explicit rng... seeded from config").

M4 addition 3: ambulance movement (the other half of step 4) is no
longer skipped either — `twin/ambulances.py`'s `dispatch_ambulance`/
`advance_ambulances`, backed by real Dijkstra routing over the full road
network (`twin/road_network.py`, resolving the "recomputed by Dijkstra"
gap dev doc §3.3 always promised but nothing ever implemented). `step`
only *enacts* dispatch (`ambulance_assignment`, decided by the agent
before `step` is even called) and advances ambulances already en route —
the deciding agent queries `road_network` directly for its own decision
(dev doc §5.3: the dispatch decision itself routes via Dijkstra, not
just its enactment), so `road_network.update_for_tick` must be called by
the caller (`experiments/runner.py`) *before* both the agent's `act` and
this `step` call for the same tick — `Simulator` doesn't call it itself,
to avoid updating it twice or in the wrong order relative to the agent's
own query.

M4 addition 4: `patient_transfer` param, enacted via `twin/demand.py`'s
`apply_patient_transfer` — dev doc §5.6's transfer rule, applied first in
step 4 (before this tick's own arrivals/discharge) so the moved patients
are reflected in the same tick's occupancy.

M4 addition 5 (the last of §5.6's four rules): `shed_tier` param, plus
`twin/power.py`'s `apply_overload_damage` — a substation whose post-shed
load still exceeds 95% capacity takes intrinsic-level damage, applied in
the *same slot as exogenous degradation* (step 1), since it's the same
kind of thing (a source of damage, just twin-caused rather than
incident-caused). `twin/power.py`'s docstring records the two design
decisions this needed (what makes load change at all; what "overloaded"
does) that nothing in the dev doc pinned precisely enough to just code.

M8 addition: `clone_for_counterfactual` — dev doc §3.7's `simulate()`
(`twin/counterfactual.py`, module M8) needs to run several independent
seeded rollouts from the *same* starting state without them stepping on
each other's mutable graph/buffer state; this method is that clone,
deliberately cheaper than a full `copy.deepcopy(simulator)` would be
(see its own docstring for what it does and doesn't copy).

2026-09-26 revision (dev doc §3.8 items 1 and 7):
- **Cascade metric.** An asset counts as a cascading failure only if it's a
  facility (hospital/substation/water) whose functional level fell below 0.5
  while its own intrinsic level was still >= 0.5, i.e. it failed because a
  supplier did. Before this, a spatial flood (which never lists
  `directly_affected_assets`) had every asset counted, roads included: 174 of
  174 on the real Kurla graph at every severity, so the metric couldn't
  distinguish agents.
- **Defensive copy.** `__init__` deep-copies the `DependencyGraph` it's given.
  It used to keep the caller's `Asset` objects, so two simulators built from
  one graph object shared damage. Every existing caller already copied first;
  this closes the trap for new ones.
"""

from __future__ import annotations

import copy
from collections.abc import Callable

import networkx as nx
import numpy as np

from udt.common.models import Asset, AssetType, DependencyGraph, TwinState
from udt.twin import ambulances as _ambulances
from udt.twin.ambulances import (
    add_transfer_requests,
    advance_ambulances,
    dispatch_ambulance,
    expire_uncollected_requests,
    find_job,
)
from udt.twin.cascade import EdgeRuntimeState, resolve_functional_levels, update_buffers
from udt.twin.demand import (
    CASUALTY_OUTCOMES_KEY,
    SURGE_MAX_HOURS,
    apply_patient_transfer,
    consume_demand,
)
from udt.twin.graph import build_networkx_graph, get_asset
from udt.twin.power import apply_overload_damage
from udt.twin.road_network import RoadNetwork

DegradationFn = Callable[[int, "nx.DiGraph[str]"], dict[str, float]]
# ^ nx.DiGraph isn't subscriptable at runtime (only under type-checking, via
# the `types-networkx` stub package) — the string keeps this a deferred
# ForwardRef instead of eagerly evaluating `nx.DiGraph[str]`.
"""(tick, graph) -> {asset_id: intrinsic_level reduction this tick}, e.g. a
flood degradation function (dev doc §4.2)."""

# Disclosed prototype-1 default, not sourced from anything (same status as
# the twin's other placeholder constants, e.g. flood.py's depth scales):
# a fully-destroyed asset (intrinsic_level=0) returns to full health
# (=1) after 20 repair applications. Agents and the harness apply a repair
# only on decision ticks (every 3rd tick, 15 min), so a full repair takes
# 20 x 15 min = 5 simulated hours, not 100 min (corrected 2026-09-29).
DEFAULT_REPAIR_RATE_PER_TICK = 0.05

# Dev doc §3.5 (revised 2026-09-26): only facilities can "cascade", and only
# when their functional drop isn't explained by their own physical damage.
CASCADE_THRESHOLD = 0.5
CASCADE_ELIGIBLE_TYPES = frozenset({AssetType.HOSPITAL, AssetType.SUBSTATION, AssetType.WATER})


def _snapshot_asset(asset: Asset) -> Asset:
    """Per-tick snapshot copy of an asset, equal in value to
    `asset.model_copy(deep=True)`.

    2026-09-28 speed-up: that deep copy was ~44% of MAPPO training time,
    mostly re-copying every road's coordinate list each tick. Geometry is
    never modified anywhere (only read), so the snapshot shares it;
    `attributes` (which does change: queues, blockage, ambulance state) is
    still deep-copied, and the scalars are immutable. Nothing uses pydantic's
    fields-set tracking (no `exclude_unset`), so `model_construct` is
    equivalent for every consumer. Bit-identical training confirmed by
    `scripts/check_reproduction.py`."""
    return Asset.model_construct(
        asset_id=asset.asset_id,
        asset_type=asset.asset_type,
        geometry=asset.geometry,
        intrinsic_level=asset.intrinsic_level,
        functional_level=asset.functional_level,
        attributes=copy.deepcopy(asset.attributes),
    )


class Simulator:
    """Owns one twin run's mutable state: the asset graph, each edge's
    buffer, and the current tick. `step()` advances exactly one tick
    (dev doc §3.5's five steps, step 4 skipped per this module's docstring).
    """

    def __init__(
        self,
        dep_graph: DependencyGraph,
        *,
        dt_minutes: float = 5.0,
        seed: int = 0,
        onset_hour_of_day: float = 0.0,
        road_network: RoadNetwork | None = None,
    ) -> None:
        dep_graph = dep_graph.model_copy(deep=True)  # see module docstring, 2026-09-26
        self.graph: nx.DiGraph[str] = build_networkx_graph(dep_graph)
        self.edge_states: dict[str, EdgeRuntimeState] = {
            edge.edge_id: EdgeRuntimeState.initial(edge) for edge in dep_graph.edges
        }
        self.dt_minutes = dt_minutes
        self.dt_hours = dt_minutes / 60.0
        self.tick = 0
        self.onset_hour_of_day = onset_hour_of_day
        # M4 ambulance-dispatch addition: optional, since not every run
        # needs a fleet (e.g. Phase 1-3 tests never pass one) — `None`
        # means ambulance enactment is skipped entirely in `step`.
        self.road_network = road_network
        # M4: the twin's own rng for patient-arrival/discharge sampling
        # (dev doc §3.6 — explicit, seeded, no bare np.random calls).
        self.rng = np.random.default_rng(seed)
        # Cascading failure count (dev doc §3.5, revised 2026-09-26): facilities
        # whose functional level dropped below 0.5 while still physically
        # intact (intrinsic >= 0.5) — tracked across the whole run, not per tick.
        self._ever_cascaded: set[str] = set()
        # Patient deaths (dev doc §5.4's patient_deaths' proxy, M4
        # addition) — cumulative across the run, same convention.
        self._patient_deaths_total = 0
        # Of which: casualties never collected within the wait deadline (dev
        # doc §3.9 item 2); stays 0 unless COUNT_UNCOLLECTED_CASUALTY_DEATHS.
        self._uncollected_deaths_total = 0
        # Patients moved by the transfer rule (dev doc §5.6, M4
        # addition) — cumulative across the run, same convention.
        self._patients_transferred_total = 0

    def clone_for_counterfactual(self, seed: int) -> Simulator:
        """M8 addition (dev doc §3.7's `simulate()`, `twin/
        counterfactual.py`): a cheap clone whose rollout can't affect
        this `Simulator`'s own state.

        Deep-copies `graph`/`edge_states` (the mutable state a rollout
        actually writes to) but **reuses `road_network` as-is, not
        deep-copied** — it's read-mostly (`~12k`-node real OSM graph;
        only its per-tick `_current_depth_scale` scalar changes, and
        that gets overwritten by `update_for_tick` before every query
        regardless of which clone is calling it), so deep-copying it for
        every one of a counterfactual's `n_rollouts` clones would be
        pure waste, not extra safety. Bypasses `__init__`'s
        `DependencyGraph` parsing via `object.__new__` — this object
        already *is* a live graph, not a fresh one to build from a
        pydantic model."""
        clone = object.__new__(Simulator)
        clone.graph = copy.deepcopy(self.graph)
        clone.edge_states = copy.deepcopy(self.edge_states)
        clone.dt_minutes = self.dt_minutes
        clone.dt_hours = self.dt_hours
        clone.tick = self.tick
        clone.onset_hour_of_day = self.onset_hour_of_day
        clone.road_network = self.road_network
        clone.rng = np.random.default_rng(seed)
        clone._ever_cascaded = set(self._ever_cascaded)
        clone._patient_deaths_total = self._patient_deaths_total
        clone._uncollected_deaths_total = self._uncollected_deaths_total
        clone._patients_transferred_total = self._patients_transferred_total
        return clone

    def asset(self, asset_id: str) -> Asset:
        return get_asset(self.graph, asset_id)

    def apply_degradation(self, asset_id: str, intrinsic_level_reduction: float) -> None:
        a = self.asset(asset_id)
        a.intrinsic_level = max(0.0, min(1.0, a.intrinsic_level - intrinsic_level_reduction))

    def apply_repair(self, asset_id: str, rate: float = DEFAULT_REPAIR_RATE_PER_TICK) -> None:
        """M4 addition (dev doc §5.6): agent-driven repair, symmetric to
        `apply_degradation` but restoring rather than damaging."""
        a = self.asset(asset_id)
        a.intrinsic_level = max(0.0, min(1.0, a.intrinsic_level + rate))

    def step(
        self,
        *,
        degradation_fn: DegradationFn | None = None,
        directly_affected_assets: frozenset[str] = frozenset(),
        repair_target: str | None = None,
        repair_rate: float = DEFAULT_REPAIR_RATE_PER_TICK,
        ambulance_assignment: dict[str, str] | None = None,
        ambulance_destination: dict[str, str] | None = None,
        patient_transfer: tuple[str, str, int] | None = None,
        shed_tier: dict[str, int] | None = None,
        divert: dict[str, bool] | None = None,
        surge: dict[str, bool] | None = None,
        transfer_requests: list[tuple[str, int, str]] | None = None,
    ) -> TwinState:
        """Advance one tick, dev doc §3.5:
        1. apply exogenous degradation (`degradation_fn`, if any is active),
           agent-driven repair (`repair_target`, M4 addition — not in the
           original dev doc numbering, applied in the same slot), enact
           `shed_tier` (writes the new tier, doesn't itself damage
           anything), then overload damage for any substation still over
           95% capacity after that shed (`twin/power.py`'s
           `apply_overload_damage` — same slot, it's a source of damage
           too, just twin-caused rather than incident-caused)
        2. drain/recover buffers
        3. fixed-point resolve functional levels
        4. consume demand — patient transfer (`patient_transfer`, M4
           addition, applied first so this tick's arrivals/discharges act
           on the post-transfer occupancy), then hospital arrivals/
           discharge (`twin/demand.py`), then, if `self.road_network` is
           set, ambulances (`ambulance_assignment` enacted, then every
           ambulance already en route/returning advanced one tick)
        5. emit `TwinState` snapshot + cascading-failure/patient-death/
           ambulance metrics
        """
        if degradation_fn is not None:
            for asset_id, reduction in degradation_fn(self.tick, self.graph).items():
                self.apply_degradation(asset_id, reduction)

        if repair_target is not None:
            self.apply_repair(repair_target, repair_rate)

        if shed_tier:
            for substation_id, tier in shed_tier.items():
                self.asset(substation_id).attributes["shed_tier"] = tier
        for hospital_id, on in (divert or {}).items():  # twin-v3 hospital status
            self.asset(hospital_id).attributes["divert"] = bool(on)
        for hospital_id, on in (surge or {}).items():
            attrs = self.asset(hospital_id).attributes
            used = float(attrs.get("surge_hours_used", 0.0))
            attrs["surge"] = bool(on) and used < SURGE_MAX_HOURS  # no budget left: stays off
        for substation_id, reduction in apply_overload_damage(self.graph).items():
            self.apply_degradation(substation_id, reduction)

        update_buffers(self.graph, self.edge_states, self.dt_hours)
        levels = resolve_functional_levels(self.graph, self.edge_states)
        for asset_id, level in levels.items():
            self.asset(asset_id).functional_level = level

        for asset_id, level in levels.items():
            asset = self.asset(asset_id)
            if (
                asset.asset_type in CASCADE_ELIGIBLE_TYPES
                and level < CASCADE_THRESHOLD
                and asset.intrinsic_level >= CASCADE_THRESHOLD
                and asset_id not in directly_affected_assets
            ):
                self._ever_cascaded.add(asset_id)

        if patient_transfer is not None:
            self._patients_transferred_total += apply_patient_transfer(
                self.graph, *patient_transfer
            )

        # Step 4 (dev doc §3.5), hospitals only — see module docstring.
        deaths_by_hospital = consume_demand(
            self.graph,
            self.tick,
            self.dt_hours,
            self.rng,
            onset_hour_of_day=self.onset_hour_of_day,
        )
        self._patient_deaths_total += sum(deaths_by_hospital.values())

        response_times_this_tick: list[float] = []
        pending_requests_count = 0
        fleet_contention = False
        if self.road_network is not None:
            if transfer_requests:  # twin-v3: health's requests become transport jobs
                add_transfer_requests(self.graph, self.tick, transfer_requests)
            if ambulance_assignment:
                for ambulance_id, request_id in ambulance_assignment.items():
                    ambulance = self.graph.nodes[ambulance_id]["asset"]
                    if ambulance.attributes.get("status") != "idle":
                        continue  # stale/invalid decision — already busy, ignore
                    request = find_job(self.graph, request_id)
                    if request is not None:
                        dispatch_ambulance(
                            self.graph,
                            self.road_network,
                            ambulance_id,
                            request,
                            (ambulance_destination or {}).get(ambulance_id),
                        )
            response_times_this_tick = advance_ambulances(self.graph, self.tick, self.dt_minutes)
            if _ambulances.COUNT_UNCOLLECTED_CASUALTY_DEATHS:
                expired = expire_uncollected_requests(self.graph, self.tick, self.dt_hours)
                self._uncollected_deaths_total += expired
                self._patient_deaths_total += expired
            pending_requests_count = len(self.graph.graph.get("pending_requests", []))
            # Twin-v3 metric: calls and transfers both waiting, fewer idle ambulances than jobs.
            n_transfers = len(self.graph.graph.get("transfer_requests", []))
            n_idle = sum(
                1
                for a in self.graph.nodes
                if self.graph.nodes[a]["asset"].asset_type == AssetType.AMBULANCE
                and self.graph.nodes[a]["asset"].attributes.get("status") == "idle"
            )
            fleet_contention = (
                pending_requests_count > 0
                and n_transfers > 0
                and n_idle < pending_requests_count + n_transfers
            )

        snapshot = TwinState(
            tick=self.tick,
            assets=[_snapshot_asset(self.asset(a)) for a in self.graph.nodes],
            cascading_failure_count=len(self._ever_cascaded),
            patient_deaths_cumulative=self._patient_deaths_total,
            uncollected_casualty_deaths_cumulative=self._uncollected_deaths_total,
            transfers_completed_cumulative=int(self.graph.graph.get("transfers_completed", 0)),
            pending_transfers_count=len(self.graph.graph.get("transfer_requests", [])),
            fleet_contention=fleet_contention,
            casualty_outcome_hours_this_tick=self.graph.graph.pop(CASUALTY_OUTCOMES_KEY, []),
            ambulance_response_times_this_tick=response_times_this_tick,
            pending_requests_count=pending_requests_count,
            patients_transferred_cumulative=self._patients_transferred_total,
        )
        self.tick += 1
        return snapshot

    def run(
        self,
        n_ticks: int,
        *,
        degradation_fn: DegradationFn | None = None,
        directly_affected_assets: frozenset[str] = frozenset(),
    ) -> list[TwinState]:
        """Convenience: step `n_ticks` times, returning every snapshot."""
        return [
            self.step(
                degradation_fn=degradation_fn, directly_affected_assets=directly_affected_assets
            )
            for _ in range(n_ticks)
        ]
