"""Patient demand consumption (dev doc §3.5 step 4, §2.3, module M4).

Implements, for hospitals only, the piece of §3.5 that every module before
M4 explicitly skipped ("no agents exist yet to act on patient state"):
patients arrive, treatments complete (discharge), queues update. Ambulance
movement (the other half of step 4) landed separately in `twin/
ambulances.py` + `twin/road_network.py` (real Dijkstra routing).

`apply_patient_transfer` (added alongside dev doc §5.6's patient-transfer
rule, `agents/rule_based.py`'s `_pick_patient_transfer`) moves already-
admitted patients between two hospitals — the decision of which pair and
how many lives in the agent, this function only enacts it.

**M8 addition — functional-level coupling, closing a gap this module's
own docstring used to flag ("arrival/discharge/death logic doesn't read
a hospital's functional_level at all"):** `effective_free_beds` scales a
hospital's *admission capacity* (not its discharge rate — see below) by
its current `functional_level`: `effective_beds_total = round(beds_total
x functional_level)`. A hospital running at 50% function can only
actually staff/safely operate about half its nominal beds; one at 0% can
admit no one. This is what `consume_demand`'s new-arrival/queue-admission
step and `apply_patient_transfer`'s destination-capacity check both use
instead of raw `beds_total`, so a power-starved hospital now genuinely
backs up its queue (and, past `PATIENT_WAIT_DEADLINE_HOURS`, produces
real deaths) and stops being a valid transfer destination — the risk
engine's Consequence signal (`risk/engine.py`'s `compute_risk`) depends
on exactly this, which is what surfaced the gap in the first place (M8's
own `scripts/calibrate_risk.py` smoke run against real Kurla data kept
reporting a near-zero Consequence even at P_failure=1.0, traced back
here).

**Still NOT modeled, disclosed rather than silently implied as
complete:** already-*admitted* patients aren't discharged early or given
excess mortality when their hospital's functional_level drops after
admission — real hospitals don't evict inpatients when the generator
fails, but care quality for patients already inside plausibly still
suffers, which this prototype doesn't represent. `MEAN_LENGTH_OF_STAY_
HOURS`-driven discharge is unaffected by functional_level for the same
reason discharge was already disclosed as a coarse exponential-service-
time approximation, not a clinical model.

**Disclosed prototype-1 shape — not fitted to any real hospital admission
dataset:**
- Arrivals: Poisson per hospital per tick, rate = `beds_total` x a fixed
  per-bed-per-hour rate x a diurnal multiplier. The per-bed rate and the
  diurnal curve's amplitude/peak hour are placeholder shapes matching the
  *spirit* of dev doc §2.3 ("Poisson, mean 2-4/hour scaled by bed count"),
  not sourced from any real Mumbai hospital dataset.
- Discharge: each occupied bed is emptied with probability
  `dt_hours / MEAN_LENGTH_OF_STAY_HOURS` this tick (a standard exponential-
  service-time approximation, not a real length-of-stay statistic).
- `patient_deaths'` (dev doc §5.4's simplified mortality proxy, its own
  wording: "say so in the thesis"): a queued patient who has waited longer
  than `PATIENT_WAIT_DEADLINE_HOURS` without getting a bed is removed and
  counted as a death this tick.

State beyond the dev doc's `beds_occupied`/`patient_queue` fields is kept
in `asset.attributes["queue_arrivals"]` — a list of the tick each still-
queued patient arrived, needed to check individual wait times against the
deadline; `patient_queue` itself stays just this list's length, matching
the dev doc's schema exactly.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np

from udt.common.models import Asset, AssetType

# All constants below are disclosed placeholders — see module docstring.
BASELINE_ARRIVAL_RATE_PER_BED_PER_HOUR = 0.02
DIURNAL_AMPLITUDE = 0.3
DIURNAL_PEAK_HOUR = 10.0
MEAN_LENGTH_OF_STAY_HOURS = 48.0
PATIENT_WAIT_DEADLINE_HOURS = 4.0

# Twin-v3 hospital status (dev doc §3.9 mechanic 2). Engineering assumptions
# (provenance category D), not sourced; both levers default off, so twin-v2
# behaviour and its random-number stream are unchanged.
SURGE_BED_FRACTION = 0.2  # surge adds 20% of nominal beds (still scaled by functional level)
SURGE_MAX_HOURS = 12.0  # staff can sustain surge this long per episode, then it ends
DIVERT_SHARE = 0.5  # share of new walk-in arrivals a diversion notice redirects


def diurnal_multiplier(hour_of_day: float) -> float:
    """1 +/- `DIURNAL_AMPLITUDE`, peaking at `DIURNAL_PEAK_HOUR` — shape
    only, disclosed in the module docstring as not fitted to real data."""
    phase = 2.0 * np.pi * (hour_of_day - DIURNAL_PEAK_HOUR) / 24.0
    return float(1.0 + DIURNAL_AMPLITUDE * np.cos(phase))


def _nominal_bed_capacity(attrs: dict[str, Any]) -> float:
    """`beds_total`, plus `SURGE_BED_FRACTION` of it while surge is on. Without
    surge this is the plain int, so twin-v2's arithmetic is unchanged."""
    beds_total = int(attrs.get("beds_total", 0))
    if attrs.get("surge"):
        return beds_total * (1.0 + SURGE_BED_FRACTION)
    return beds_total


def _nearest_accepting_hospital(graph: nx.DiGraph[str], hospital_id: str) -> str | None:
    """Closest other hospital (straight line) that is not diverting."""
    here = graph.nodes[hospital_id]["asset"].geometry["coordinates"]
    best_id, best_d = None, float("inf")
    for asset_id in graph.nodes:
        a = graph.nodes[asset_id]["asset"]
        if (
            asset_id == hospital_id
            or a.asset_type != AssetType.HOSPITAL
            or a.attributes.get("divert")
        ):
            continue
        x, y = a.geometry["coordinates"][:2]
        d = (x - here[0]) ** 2 + (y - here[1]) ** 2
        if d < best_d:
            best_id, best_d = asset_id, d
    return best_id


def effective_free_beds(asset: Asset) -> int:
    """M8 addition (see module docstring's "functional-level coupling"
    section): a hospital's actual admission capacity right now —
    `beds_total` scaled by its current `functional_level`, minus
    whatever's already occupied, floored at 0 (a hospital that's lost
    function *after* admitting patients past its new effective capacity
    isn't asked to retroactively discharge anyone — it just can't take
    more until it recovers or some occupied beds free up).

    Used by `apply_patient_transfer` (a destination hospital's real
    capacity) and by `constraints/engine.py`'s `bed_capacity` rule (so
    the constraint check and the actual enactment agree on what "free"
    means) — NOT by `consume_demand`, which inlines the same formula
    against its own in-progress local `beds_occupied` value rather than
    the asset's not-yet-written-back attribute."""
    beds_occupied = int(asset.attributes.get("beds_occupied", 0))
    effective_beds_total = round(_nominal_bed_capacity(asset.attributes) * asset.functional_level)
    return max(0, effective_beds_total - beds_occupied)


def consume_demand(
    graph: nx.DiGraph[str],
    tick: int,
    dt_hours: float,
    rng: np.random.Generator,
    *,
    onset_hour_of_day: float = 0.0,
) -> dict[str, int]:
    """dev doc §3.5 step 4, hospitals only. Mutates each hospital's
    `beds_occupied`/`patient_queue`/`queue_arrivals` attributes in place
    (same convention as `cascade.py` mutating `functional_level`).
    Returns `{hospital_asset_id: deaths_this_tick}` for the caller
    (`Simulator.step`) to accumulate into `TwinState.patient_deaths_cumulative`.

    Order per tick: discharge frees beds first, then queued patients (FIFO)
    fill freed beds, then new arrivals compete for whatever's left, then
    the deadline check removes anyone who's waited too long — matching
    dev doc §3.5's own step-4 wording order ("patients arrive [...],
    treatments complete [...], queues update").
    """
    hour_of_day = (onset_hour_of_day + tick * dt_hours) % 24.0
    deaths: dict[str, int] = {}
    redirected: dict[str, int] = {}  # twin-v3 diversion: patients arriving elsewhere this tick

    for asset_id in graph.nodes:
        asset = graph.nodes[asset_id]["asset"]
        if asset.asset_type != AssetType.HOSPITAL:
            continue
        attrs = asset.attributes
        beds_total = int(attrs.get("beds_total", 0))
        beds_occupied = int(attrs.get("beds_occupied", 0))
        queue: list[int] = list(attrs.get("queue_arrivals", []))

        # M8: a power/water/access-starved hospital can't safely staff
        # every nominal bed — see module docstring's "functional-level
        # coupling" section. Computed once here (not via the shared
        # `effective_free_beds` helper, which reads the asset's
        # attributes directly — this function's `beds_occupied` is a
        # local, still-being-updated value, not yet written back).
        effective_beds_total = round(_nominal_bed_capacity(attrs) * asset.functional_level)
        if attrs.get("surge"):  # twin-v3: surge uses up its staff-hours budget
            attrs["surge_hours_used"] = float(attrs.get("surge_hours_used", 0.0)) + dt_hours
            if attrs["surge_hours_used"] >= SURGE_MAX_HOURS:
                attrs["surge"] = False  # takes effect from the next tick

        # 1. discharges
        p_discharge = min(1.0, dt_hours / MEAN_LENGTH_OF_STAY_HOURS)
        n_discharge = int(rng.binomial(beds_occupied, p_discharge)) if beds_occupied > 0 else 0
        beds_occupied -= n_discharge

        # 2. admit queued patients (oldest first) into freed *effective* beds
        free_beds = effective_beds_total - beds_occupied
        n_admit_from_queue = min(len(queue), max(0, free_beds))
        if n_admit_from_queue > 0:
            queue = queue[n_admit_from_queue:]
            beds_occupied += n_admit_from_queue
            free_beds -= n_admit_from_queue

        # 3. new arrivals this tick - rate scales with the hospital's
        # *nominal* beds_total, not effective_beds_total: arrivals model
        # real-world demand showing up at the door regardless of the
        # hospital's current capacity to treat it; `free_beds` (effective,
        # capacity-limited) below is what actually gates how many of them
        # get admitted versus queued.
        rate_per_hour = (
            BASELINE_ARRIVAL_RATE_PER_BED_PER_HOUR * beds_total * diurnal_multiplier(hour_of_day)
        )
        n_arrivals = int(rng.poisson(max(0.0, rate_per_hour * dt_hours)))
        if attrs.get("divert") and n_arrivals > 0:  # twin-v3; never drawn in twin-v2
            target = _nearest_accepting_hospital(graph, asset_id)
            if target is not None:
                n_redirect = int(rng.binomial(n_arrivals, DIVERT_SHARE))
                n_arrivals -= n_redirect
                redirected[target] = redirected.get(target, 0) + n_redirect
                attrs["patients_diverted_out"] = (
                    int(attrs.get("patients_diverted_out", 0)) + n_redirect
                )
        n_admit_now = min(n_arrivals, max(0, free_beds))
        beds_occupied += n_admit_now
        queue.extend([tick] * (n_arrivals - n_admit_now))

        # 4. deadline check — dev doc §5.4 patient_deaths' proxy
        deadline_ticks = PATIENT_WAIT_DEADLINE_HOURS / dt_hours
        still_waiting = [t for t in queue if (tick - t) <= deadline_ticks]
        n_deaths = len(queue) - len(still_waiting)

        attrs["beds_occupied"] = beds_occupied
        attrs["patient_queue"] = len(still_waiting)
        attrs["queue_arrivals"] = still_waiting
        deaths[asset_id] = n_deaths

    # Redirected walk-ins join the receiving hospital's queue now (travel time
    # between hospitals is ignored for walk-ins, a disclosed simplification)
    # and are admitted from the queue from the next tick.
    for target, n in redirected.items():
        if n > 0:
            t_attrs = graph.nodes[target]["asset"].attributes
            t_queue = list(t_attrs.get("queue_arrivals", [])) + [tick] * n
            t_attrs["queue_arrivals"] = t_queue
            t_attrs["patient_queue"] = len(t_queue)

    return deaths


def apply_patient_transfer(
    graph: nx.DiGraph[str], from_hospital_id: str, to_hospital_id: str, count: int
) -> int:
    """dev doc §5.6's patient-transfer rule, enactment half (the decision
    of *which* pair and *how many* is `agents/rule_based.py`'s job — see
    its `_pick_patient_transfer`). Moves already-admitted patients
    (`beds_occupied`), not queued ones — `count` is capped by both what
    the source actually has occupied and what the destination actually
    has free (M8: `effective_free_beds`, scaled by the destination's
    `functional_level` — a power-starved hospital is no longer a valid
    dumping ground just because its *nominal* bed count looks free),
    since the agent's own count is a request, not a guarantee (twin
    state may have moved between the agent's decision and this call,
    e.g. the flood also acting this tick). Returns the number actually
    moved."""
    source = graph.nodes[from_hospital_id]["asset"]
    dest = graph.nodes[to_hospital_id]["asset"]
    source_attrs = source.attributes

    moved = max(0, min(count, int(source_attrs.get("beds_occupied", 0)), effective_free_beds(dest)))

    source_attrs["beds_occupied"] = int(source_attrs.get("beds_occupied", 0)) - moved
    dest.attributes["beds_occupied"] = int(dest.attributes.get("beds_occupied", 0)) + moved
    return moved
