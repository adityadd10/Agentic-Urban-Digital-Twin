"""Episode metrics (dev doc §5.4's reward terms, reduced to prototype-1
scope, module M4).

Every term dev doc §5.4 names except `water_shortage_hours` is now
computable. That term's blocker changed, not disappeared: the real
Kurla graph used to have 0 water facilities (a genuine data gap); M2's
2026-09-13 revision closed that (2 flagged-synthetic water assets, see
`data/manual_facilities.yaml`), but isolating a hospital's water-specific
shortage from its combined `functional_level = intrinsic x Pi sat(d)`
product needs per-edge `EdgeRuntimeState`/satisfaction in the tick
trace, which `TwinState` doesn't carry — a real, still-open, disclosed
gap of its own (see `MTP_Module_Planner.md`'s M2 row for the full
history). The rest:
`unmet_patient_hours`/`patient_deaths` (`twin/demand.py`),
`mean_ambulance_response_delay_hours`/`requests_completed`/
`requests_pending_at_end` (`twin/ambulances.py` + `twin/road_network.py`),
and `unserved_energy_mwh` (`twin/power.py` — shed load actually means
something now that overload has a real consequence). `patients_transferred`
isn't one of §5.4's named terms, but is reported for transparency:
without it, a transfer rule that never fires would look identical to
one that fires constantly. `safety_violations_attempted` (M7 addition,
dev doc §8's H3 metric) is read off `trace[-1].safety_violations_
attempted_cumulative` — set externally by whichever caller ran the
constraint check (`Simulator` itself never runs it, see `TwinState`'s
own docstring), so it's 0 for any trace whose caller doesn't call
`constraints/engine.py`'s `check(...)` at all (e.g. a bare `Simulator.
run()` with no harness wrapping it).

What this module computes is what the twin can honestly measure today.
This is not a claim that the *numbers* are validated against anything
real — see each contributing module's own docstring for its disclosed
placeholder constants.
"""

from __future__ import annotations

import numpy as np

from udt.common.models import AssetType, EpisodeMetrics, TwinState
from udt.twin.power import SHED_FRACTION_BY_TIER

CRITICAL_TYPES = frozenset({AssetType.HOSPITAL, AssetType.SUBSTATION, AssetType.WATER})

# Matches `Simulator`'s default `dt_minutes=5.0` (dev doc §3.1) — not
# threaded through from the caller because nothing in this prototype ever
# overrides the tick length; revisit if that changes.
DEFAULT_DT_HOURS = 5.0 / 60.0


def compute_episode_metrics(
    scenario_id: str,
    agent_name: str,
    trace: list[TwinState],
    *,
    dt_hours: float = DEFAULT_DT_HOURS,
) -> EpisodeMetrics:
    """`trace` is one episode's full list of per-tick `TwinState`
    snapshots, in tick order (as returned by `Simulator.run`/collected
    tick-by-tick by `experiments/runner.py`)."""
    hospital_levels: list[float] = []
    critical_levels: list[float] = []
    unmet_patient_hours = 0.0
    unserved_energy_mwh = 0.0
    response_times_hours: list[float] = []
    for state in trace:
        queued_this_tick = 0
        for asset in state.assets:
            if asset.asset_type == AssetType.HOSPITAL:
                hospital_levels.append(asset.functional_level)
                queued_this_tick += int(asset.attributes.get("patient_queue", 0))
            if asset.asset_type in CRITICAL_TYPES:
                critical_levels.append(asset.functional_level)
            if asset.asset_type == AssetType.SUBSTATION:
                shed_fraction = SHED_FRACTION_BY_TIER[int(asset.attributes.get("shed_tier", 0))]
                load_mw = float(asset.attributes.get("load_mw", 0.0))
                unserved_energy_mwh += load_mw * shed_fraction * dt_hours
        unmet_patient_hours += queued_this_tick * dt_hours
        response_times_hours.extend(state.ambulance_response_times_this_tick)

    casualty_outcomes = [h for snap in trace for h in snap.casualty_outcome_hours_this_tick]
    return EpisodeMetrics(
        scenario_id=scenario_id,
        agent_name=agent_name,
        cascading_failure_count=trace[-1].cascading_failure_count if trace else 0,
        mean_hospital_functional_level=(
            float(np.mean(hospital_levels)) if hospital_levels else 1.0
        ),
        mean_critical_functional_level=(
            float(np.mean(critical_levels)) if critical_levels else 1.0
        ),
        unmet_patient_hours=unmet_patient_hours,
        patient_deaths=trace[-1].patient_deaths_cumulative if trace else 0,
        uncollected_casualty_deaths=(
            trace[-1].uncollected_casualty_deaths_cumulative if trace else 0
        ),
        transfers_completed=trace[-1].transfers_completed_cumulative if trace else 0,
        jobs_completed=len(response_times_hours)
        + (trace[-1].transfers_completed_cumulative if trace else 0),
        mean_casualty_time_to_admission_hours=(
            float(np.mean(casualty_outcomes)) if casualty_outcomes else None
        ),
        fleet_contention_fraction=(
            sum(s.fleet_contention for s in trace) / len(trace) if trace else 0.0
        ),
        mean_ambulance_response_delay_hours=(
            float(np.mean(response_times_hours)) if response_times_hours else None
        ),
        requests_completed=len(response_times_hours),
        requests_pending_at_end=trace[-1].pending_requests_count if trace else 0,
        patients_transferred=trace[-1].patients_transferred_cumulative if trace else 0,
        unserved_energy_mwh=unserved_energy_mwh,
        safety_violations_attempted=(
            trace[-1].safety_violations_attempted_cumulative if trace else 0
        ),
        ticks_run=len(trace),
    )
