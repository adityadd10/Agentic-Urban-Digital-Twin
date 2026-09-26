"""Counterfactual simulation (dev doc §3.7, module M8).

Dev doc §3.7's exact spec: `simulate(state: TwinState, plan_or_actions,
horizon_ticks, n_rollouts, base_seed) -> SimulationResult` — "deep-
copies the twin, applies the plan/actions, runs n_rollouts (default
M=10) seeded rollouts of horizon_ticks (default 72 = 6h), returns
per-rollout outcome metrics + summary (mean, std, P(failure))".

Deferred since M1 ("no agent needs counterfactual rollouts yet") until
now — M8's risk engine (dev doc §9.1) is exactly that first agent:
`risk/engine.py`'s `compute_risk` needs `P_failure`/`Consequence` over a
batch of "what if we take this action" rollouts, which is precisely
what this module provides.

**Disclosed deviation from the literal `state: TwinState` signature:**
`simulate()` here takes a live `Simulator`, not a bare `TwinState`
snapshot. A `TwinState` (dev doc §3.5's per-tick *output*) doesn't carry
enough to actually continue a rollout — no edge buffer levels
(`EdgeRuntimeState`), no pending-ambulance-request queue, no rng state.
Every other harness in this codebase (`experiments/runner.py`, `envs/
single_env.py`, `envs/multi_env.py`) already threads a live `Simulator`
through for exactly this reason, never a bare `TwinState`; `simulate()`
matches that established convention instead of the dev doc's literal
(but under-specified) typing.

**Scope cut inside each rollout, disclosed:** a rollout does NOT call
`twin/ambulances.py`'s `generate_requests` or `RoadNetwork.
update_for_tick` — a counterfactual is answering "what if I take THIS
repair/transfer/shed/dispatch action right now and let the incident's
existing trajectory play out", not "what if a different sequence of new
ambulance calls happened to arrive" or "what if the flood's depth field
changes mid-rollout in some new way" — re-rolling those would fold a
second, unrelated source of randomness into a risk estimate that's
supposed to be about *this* decision's consequences. `update_substation_
load` (the incident-driven power-surge model) and the core cascade/
demand/repair/shed dynamics (`Simulator.step` itself) DO run every
tick — those are the twin's actual physics, not new exogenous events.

**Fragility is resampled per rollout (2026-09-26, dev doc §3.8 item 5).**
Unlike ambulance calls, a facility's critical flood depth is something the
decision-maker genuinely doesn't know. If the degradation callable supports
`with_fragility_seed` (the flood's `FloodDegradation` does), each rollout uses
a copy that redraws critical depths for facilities that haven't failed yet,
conditional on the depth they've already survived. Before this, the flood
was deterministic and every rollout agreed, so P(failure) was always 0 or 1.
"""

from __future__ import annotations

import numpy as np

from udt.common.models import AgentAction, AssetType, Incident, RolloutOutcome, SimulationResult
from udt.twin.power import update_substation_load
from udt.twin.simulator import DegradationFn, Simulator

# dev doc §3.7's own defaults.
HORIZON_TICKS_DEFAULT = 72  # 6h at dt=5min
N_ROLLOUTS_DEFAULT = 10

# Same set `logging/metrics.py`'s `CRITICAL_TYPES` uses — duplicated
# rather than imported, to keep `twin/` from depending on `logging/`
# (the dependency direction stays one-way: twin <- logging, same as
# twin <- agents/envs elsewhere in this codebase).
CRITICAL_TYPES = frozenset({AssetType.HOSPITAL, AssetType.SUBSTATION, AssetType.WATER})

# dev doc §9.1: "any critical asset (hospital) functional_level < 0.3".
FAILURE_THRESHOLD = 0.3


def _run_one_rollout(
    sim: Simulator,
    action: AgentAction,
    *,
    degradation_fn: DegradationFn | None,
    incident: Incident | None,
    horizon_ticks: int,
    seed: int,
) -> RolloutOutcome:
    clone = sim.clone_for_counterfactual(seed=seed)
    with_seed = getattr(degradation_fn, "with_fragility_seed", None)
    if with_seed is not None:
        degradation_fn = with_seed(seed)
    # `_ever_cascaded`'s length is exactly what each tick's `TwinState.
    # cascading_failure_count` reports (`Simulator.step` sets it that
    # way) — read once here, before the rollout, as the baseline the
    # last tick's snapshot below is compared against. Same-package
    # access (both live in `twin/`), not reaching across a module
    # boundary.
    starting_cascade_count = len(clone._ever_cascaded)

    min_hospital_level = 1.0
    min_critical_level = 1.0
    unmet_patient_hours = 0.0
    snapshot = None

    for tick_in_rollout in range(horizon_ticks):
        if incident is not None:
            update_substation_load(clone.graph, clone.tick, incident, clone.dt_minutes)
        snapshot = clone.step(
            degradation_fn=degradation_fn,
            # dev doc §3.7: "applies the plan/actions" once, at the start
            # of the rollout — same one-decision-per-window convention
            # every other harness in this codebase uses for `AgentAction`.
            repair_target=action.repair_target if tick_in_rollout == 0 else None,
            ambulance_assignment=action.ambulance_assignment if tick_in_rollout == 0 else None,
            patient_transfer=action.patient_transfer if tick_in_rollout == 0 else None,
            shed_tier=action.shed_tier if tick_in_rollout == 0 else None,
        )
        queued_this_tick = 0
        for asset in snapshot.assets:
            if asset.asset_type == AssetType.HOSPITAL:
                min_hospital_level = min(min_hospital_level, asset.functional_level)
                queued_this_tick += int(asset.attributes.get("patient_queue", 0))
            if asset.asset_type in CRITICAL_TYPES:
                min_critical_level = min(min_critical_level, asset.functional_level)
        unmet_patient_hours += queued_this_tick * clone.dt_hours

    assert snapshot is not None  # horizon_ticks >= 1 in every real call
    new_cascading_failures = snapshot.cascading_failure_count - starting_cascade_count
    return RolloutOutcome(
        min_hospital_functional_level=min_hospital_level,
        min_critical_functional_level=min_critical_level,
        unmet_patient_hours=unmet_patient_hours,
        new_cascading_failures=new_cascading_failures,
    )


def simulate(
    sim: Simulator,
    action: AgentAction,
    *,
    degradation_fn: DegradationFn | None,
    incident: Incident | None = None,
    horizon_ticks: int = HORIZON_TICKS_DEFAULT,
    n_rollouts: int = N_ROLLOUTS_DEFAULT,
    base_seed: int = 0,
) -> SimulationResult:
    """Dev doc §3.7. Runs `n_rollouts` independent seeded rollouts (seeds
    `base_seed`, `base_seed+1`, ..., `base_seed+n_rollouts-1` — dev doc
    §3.6's determinism guarantee: same `sim`/`action`/`base_seed` always
    reproduces the same `SimulationResult`) of `horizon_ticks` each, all
    starting from `sim`'s *current* state, and summarizes them. `sim`
    itself is never mutated — every rollout runs on its own
    `clone_for_counterfactual`."""
    rollouts = [
        _run_one_rollout(
            sim,
            action,
            degradation_fn=degradation_fn,
            incident=incident,
            horizon_ticks=horizon_ticks,
            seed=base_seed + i,
        )
        for i in range(n_rollouts)
    ]
    p_failure = float(
        np.mean([r.min_hospital_functional_level < FAILURE_THRESHOLD for r in rollouts])
    )
    outcomes = [r.unmet_patient_hours for r in rollouts]
    return SimulationResult(
        rollouts=rollouts,
        p_failure=p_failure,
        mean_outcome=float(np.mean(outcomes)),
        std_outcome=float(np.std(outcomes)),
    )
