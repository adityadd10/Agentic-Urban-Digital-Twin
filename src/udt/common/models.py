"""Shared cross-component Pydantic models (dev doc §12.2).

Every message that crosses a module boundary in this codebase is a Pydantic
model defined here — no bare dicts. Models are added module by module; see
`MTP_Module_Planner.md` for which module owns which addition.

M1 (twin core, dev doc §3) added: `AssetType`, `Asset`, `DependencyEdge`,
`TwinState`. Prototype-1 scope: flood-only, so `Asset.attributes` covers
hospital/substation/water/road only (ambulance/ECC deferred — no agents
exist yet to use them).

M3 (incidents, dev doc §4) added: `Incident`, `Scenario`.

M4 (rule-based baseline + harness, dev doc §5.6) added: `AgentAction`,
`EpisodeMetrics` — narrowed to prototype-1 scope, extended slice by
slice (repair-crew dispatch, then patient demand — `TwinState.
patient_deaths_cumulative` — then ambulance dispatch/routing); see each
model's own docstring for exactly what's implemented vs. still deferred.

M6b (goal-conditioning + policy registry, dev doc §5.5-§5.6) added:
`PolicyRegistryEntry` — `registry/policies.yaml`'s row schema.

M7 (constraint engine, dev doc §8) added: `ConstraintViolation`,
`ConstraintReport` — `constraints/engine.py`'s `check(...)` return type.

M8 (risk engine, dev doc §3.7/§9) added: `RolloutOutcome`,
`SimulationResult` (`twin/counterfactual.py`'s `simulate()`),
`RiskAssessment`, `RiskCalibration`, `AutonomyTier` (`risk/engine.py`,
`risk/gate.py`).

M9b (ID/OOD router + LLM live wiring, dev doc §6/§7/§15) added:
`RoutingDecision`, `RouterCalibration` — `routing/ood.py`'s `route(...)`
return type and its calibration reference data; `ExperimentConfig` —
`configs/experiments/*.yaml`'s schema.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# M1 — twin core (dev doc §3.2, §3.4)
# ---------------------------------------------------------------------------
class AssetType(str, Enum):
    """Dev doc §3.2. `ambulance`/`ecc` are part of the full spec but out of
    scope for prototype 1 (no agents exist yet to act on them)."""

    HOSPITAL = "hospital"
    SUBSTATION = "substation"
    WATER = "water"
    ROAD = "road"
    AMBULANCE = "ambulance"
    ECC = "ecc"


class Asset(BaseModel):
    """Dev doc §3.2. `attributes` holds the per-type fields from the dev
    doc's attribute table (e.g. `beds_total` for a hospital) — kept as a
    dict rather than a union of per-type models so the dependency-graph
    loader stays simple; downstream code (M1's cascade engine, M4's rule
    agent) reads named keys out of it per `asset_type`.
    """

    asset_id: str
    asset_type: AssetType
    geometry: dict[str, Any]  # GeoJSON (point or linestring), WGS84
    intrinsic_level: float = Field(ge=0.0, le=1.0, default=1.0)
    functional_level: float = Field(ge=0.0, le=1.0, default=1.0)
    attributes: dict[str, Any] = Field(default_factory=dict)


class DependencyEdge(BaseModel):
    """Dev doc §3.4. `kind="access"` edges use `demand`/`buffer_hours` as
    route-quality terms (§3.3): supply = 1 - blockage of the best route."""

    edge_id: str
    supplier: str
    consumer: str
    kind: Literal["power", "water", "access"]
    demand: float
    criticality: float = Field(ge=0.0, le=1.0)
    buffer_hours: float = Field(ge=0.0)
    floor: float = Field(ge=0.0, le=1.0)


class DependencyGraph(BaseModel):
    """Dev doc §3.4 — the twin's canonical input, written by
    `scripts/06_build_dependency_graph.py`."""

    version: str = "1.0"
    assets: list[Asset]
    edges: list[DependencyEdge]


class TwinState(BaseModel):
    """Dev doc §3.5 step 5 — per-tick snapshot emitted by the simulator.

    `patient_deaths_cumulative` (M4 addition): running total of dev doc
    §5.4's `patient_deaths'` proxy across the whole run so far, same
    convention as `cascading_failure_count` — a monotonically
    nondecreasing counter, not a per-tick value.

    `ambulance_response_times_this_tick`/`pending_requests_count` (M4
    ambulance-dispatch addition): the *first* is a per-tick delta (hours
    for each request picked up this tick, usually empty — summed across
    the trace by `logging/metrics.py`, not read from the last snapshot
    like the two cumulative counters above, to avoid carrying a
    forever-growing list in every snapshot); the second is a plain
    snapshot of the current queue length."""

    tick: int
    assets: list[Asset]
    cascading_failure_count: int = 0
    patient_deaths_cumulative: int = 0
    uncollected_casualty_deaths_cumulative: int = 0
    transfers_completed_cumulative: int = 0  # twin-v3 transfer jobs delivered
    pending_transfers_count: int = 0  # twin-v3 transfer jobs not yet picked up
    fleet_contention: bool = False  # twin-v3: calls + transfers waiting, idle < jobs
    casualty_outcome_hours_this_tick: list[float] = Field(default_factory=list)
    """Twin-v3 metric (per-tick delta, like `ambulance_response_times_this_tick`):
    hours from emergency call to bed for casualties admitted this tick, or to
    death for casualties who died waiting (queue or uncollected)."""
    """Part of `patient_deaths_cumulative`: calls never answered within the
    wait deadline (dev doc §3.9 item 2). Always 0 in twin-v2."""
    ambulance_response_times_this_tick: list[float] = Field(default_factory=list)
    pending_requests_count: int = 0
    patients_transferred_cumulative: int = 0
    safety_violations_attempted_cumulative: int = 0
    """M7 addition (dev doc §8's H3 metric): running count, across the
    whole run so far, of constraint violations `constraints/engine.py`'s
    `check(...)` has caught (and repaired) — same monotonically-
    nondecreasing-counter convention as `cascading_failure_count`. Unlike
    that field, `Simulator.step` never sets this itself — the constraint
    check happens in the *caller* (`experiments/runner.py`, `envs/
    single_env.py`, `envs/multi_env.py`), before `step()` is even
    invoked, so each caller sets this field on the snapshot `step()`
    returns rather than `Simulator` needing to know about constraints at
    all."""


# ---------------------------------------------------------------------------
# M3 — incidents & scenarios (dev doc §4.1, §4.3), flood-only for prototype 1
# ---------------------------------------------------------------------------
class Incident(BaseModel):
    """Dev doc §4.1. Prototype 1 only ever instantiates `type="flood"` —
    the field stays a free string (not an enum) because the dev doc's full
    scope includes `substation_failure`/`fire`/`cyberattack`/`unknown`,
    deliberately deferred here, not removed from the model."""

    incident_id: str
    type: str
    location: dict[str, Any]  # GeoJSON point or polygon footprint
    onset_tick: int
    severity: float = Field(ge=0.0, le=1.0)
    raw_signal: dict[str, Any] = Field(default_factory=dict)
    directly_affected_assets: list[str] = Field(default_factory=list)
    profile: dict[str, Any] = Field(default_factory=dict)


class Scenario(BaseModel):
    """Dev doc §4.3. `suite_version` guards the frozen-suite rule: once
    generated, a suite is never silently regenerated."""

    scenario_id: str
    incident: Incident
    initial_conditions: dict[str, Any] = Field(default_factory=dict)
    seed: int
    suite_version: str = "v1"


# ---------------------------------------------------------------------------
# M4 — rule-based baseline & harness (dev doc §5.6, §12.2 Agent.act interface)
# ---------------------------------------------------------------------------
class AgentAction(BaseModel):
    """Dev doc §5.6's `Agent.act(obs) -> actions` interface. Prototype-1
    scope: all four §5.6 rules are implemented — see MTP_Module_Planner.md's
    M4 row for the build order and the design decisions each one needed.
    `repair_target=None` means "no repair action this tick"."""

    repair_target: str | None = None
    ambulance_assignment: dict[str, str] | None = None
    """{ambulance_asset_id: request_id} for newly-dispatched idle
    ambulances this tick (dev doc §5.3's "assignment" action, §5.6's
    "dispatch nearest idle ambulance to oldest critical request") — only
    idle ambulances should be assigned; `Simulator.step` enacting this is
    what actually commits the routing decision (`twin/ambulances.py`'s
    `dispatch_ambulance`)."""
    ambulance_destination: dict[str, str] | None = None
    """Twin-v3 (dev doc §3.9, mechanic 1): {ambulance_asset_id: hospital_id},
    the hospital a casualty collected by that ambulance is taken to. A missing
    entry (or None) means the ambulance's home hospital, which is exactly
    twin-v2 behaviour."""
    transfer_requests: list[tuple[str, int, str]] | None = None
    """Twin-v3 (dev doc §3.9 mechanic 3), health's action: `(from_hospital_id,
    count, urgency)`; each patient becomes a transfer job that transport serves
    with `ambulance_assignment` / `ambulance_destination`."""
    divert: dict[str, bool] | None = None
    """Twin-v3 (dev doc §3.9 mechanic 2): {hospital_id: diverting?}. Persistent
    until changed, like `shed_tier`; a diverting hospital redirects a share of new
    walk-ins to the nearest accepting hospital (`twin/demand.py`)."""
    surge: dict[str, bool] | None = None
    """Twin-v3: {hospital_id: surge on?}. Adds beds until the hospital's surge
    staff-hours budget runs out (`twin/demand.py` SURGE_*)."""
    patient_transfer: tuple[str, str, int] | None = None
    """(from_hospital_id, to_hospital_id, count) — dev doc §5.3's "per
    hospital-pair transfer decision", §5.6's "transfer patients out of
    any hospital predicted [...] to lose power within its buffer window,
    nearest-available-bed first". One transfer per tick, same one-
    decision-per-tick convention as `repair_target`/
    `ambulance_assignment` — the dev doc's own wording is singular
    ("any hospital"), not "every at-risk hospital simultaneously"."""
    shed_tier: dict[str, int] | None = None
    """{substation_asset_id: new shed_tier (0-3)} — dev doc §5.3's shed-
    tier action, §5.6's "shed load tier-by-tier when a substation
    exceeds 95% capacity". Unlike the other three actions, every
    substation is independently eligible each tick (the dev doc's
    wording doesn't imply picking only the single worst substation among
    several, the way "the highest-criticality failed substation"/"the
    oldest critical request"/"any hospital" read for the other rules)."""


class EpisodeMetrics(BaseModel):
    """Dev doc §5.4's reward terms, reduced to what the twin can currently
    measure. `unmet_patient_hours`/`patient_deaths` became computable once
    `twin/demand.py` landed (M4); `unserved_energy_mwh`/
    `ambulance_response_delay_hours` still aren't — no overload-failure
    consequence or ambulance/routing model exists yet. See
    `logging/metrics.py`'s module docstring for the full scope note."""

    scenario_id: str
    agent_name: str
    cascading_failure_count: int
    mean_hospital_functional_level: float
    mean_critical_functional_level: float  # hospitals + substations + water
    unmet_patient_hours: float  # sum over ticks of (total queued patients x dt_hours)
    patient_deaths: int  # dev doc §5.4's simplified mortality proxy
    uncollected_casualty_deaths: int = 0  # part of patient_deaths (dev doc §3.9); 0 in twin-v2
    transfers_completed: int = 0  # twin-v3 transfer jobs delivered; 0 in twin-v2
    jobs_completed: int = 0  # requests_completed + transfers_completed
    mean_casualty_time_to_admission_hours: float | None = None  # call -> bed (deaths censored)
    fleet_contention_fraction: float = 0.0  # share of ticks with calls/transfers contention
    mean_ambulance_response_delay_hours: float | None = None  # None = no request completed
    requests_completed: int = 0
    requests_pending_at_end: int = 0
    patients_transferred: int = 0
    unserved_energy_mwh: float = 0.0  # dev doc §5.4's last remaining term, now computable
    safety_violations_attempted: int = 0  # M7 addition, dev doc §8's H3 metric
    ticks_run: int


# ---------------------------------------------------------------------------
# M6b — goal-conditioning + policy registry (dev doc §5.5-§5.6)
# ---------------------------------------------------------------------------
class PolicyRegistryEntry(BaseModel):
    """`registry/policies.yaml`'s row schema — dev doc §5.5 exactly:
    "{incident_type, policy_path, env_version, suite_version, val_score,
    trained_date} — the ID/OOD router reads this file." That router
    (M9b) doesn't exist yet, so nothing reads this back programmatically
    today; `scripts/train_mappo.py` writes an entry after training so
    the schema and the write path are both real ahead of M9b needing
    them, matching this project's own "frozen interface before the
    module that reads it" convention (dev doc §0.3).

    `val_score` is disclosed as NOT a true held-out validation score —
    no train/val/test scenario suite exists yet (M3's own deferral), so
    this is the training rollout's own mean episode reward, labeled as
    such via `val_score_is_true_holdout=False` rather than silently
    implying more rigor than exists."""

    incident_type: str
    policy_path: str
    env_version: str
    suite_version: str
    val_score: float
    val_score_is_true_holdout: bool = False
    trained_date: str


# ---------------------------------------------------------------------------
# M7 — constraint engine (dev doc §8)
# ---------------------------------------------------------------------------
class ConstraintViolation(BaseModel):
    """One rule's verdict against one joint action (dev doc §8's
    `ConstraintReport.violations[]`). A single `check(...)` call can
    produce more than one `ConstraintViolation` for the *same* rule_id —
    e.g. two ambulances in one joint action both proposing an unsafe
    route under `route_flood_safety` — so `ConstraintReport.violations`
    is a flat list across all rules and all offending sub-actions, not
    one-entry-per-rule.

    `on_violation` records which of the dev doc's three repair verbs
    ("clip magnitude -> drop sub-action -> replace with no-op") this
    violation triggered. `reject` ("replace with no-op") is part of the
    dev doc's vocabulary but isn't exercised by either rule implemented
    so far (`bed_capacity` clips, `route_flood_safety` drops) — kept in
    the Literal for whichever future rule needs it, not removed for
    tidiness."""

    rule_id: str
    scope: Literal["health", "power", "transport"]
    on_violation: Literal["clip", "drop", "reject"]
    detail: str


class ConstraintReport(BaseModel):
    """`constraints/engine.py`'s `check(state, joint_action) ->
    ConstraintReport` (dev doc §8's API, verbatim). `repaired_action` is
    always a valid `AgentAction` — when nothing needed repair it's
    exactly the input action; `passed` is `len(violations) == 0`, kept
    as its own field (not derived on read) so a report can be logged and
    reloaded without recomputing it."""

    passed: bool
    violations: list[ConstraintViolation] = Field(default_factory=list)
    repaired_action: AgentAction


# ---------------------------------------------------------------------------
# M8 — counterfactual simulation + risk engine (dev doc §3.7, §9)
# ---------------------------------------------------------------------------
class RolloutOutcome(BaseModel):
    """One counterfactual rollout's outcome (dev doc §3.7: `simulate()`
    returns "per-rollout outcome metrics"). `new_cascading_failures` is
    the delta *during this rollout* (the number of newly-cascaded assets
    from the rollout's start to its end), not the running total the twin
    otherwise reports — a rollout imagines a fixed horizon starting from
    "now", so the metric that matters here is what happens next, not
    what already happened before the rollout began."""

    min_hospital_functional_level: float
    min_critical_functional_level: float  # hospitals + substations + water
    unmet_patient_hours: float
    new_cascading_failures: int


class SimulationResult(BaseModel):
    """`twin/counterfactual.py`'s `simulate()` return type (dev doc §3.7:
    "returns per-rollout outcome metrics + summary (mean, std,
    P(failure))"). Two choices this model makes that §3.7's own text
    leaves open (it's written to be reusable by future callers, not
    tied to one specific metric):

    - `p_failure` uses dev doc §9.1's own literal condition — "any
      critical asset (hospital) functional_level < 0.3" — i.e. a
      *hospital* dropping under 0.3, read from `min_hospital_
      functional_level`, not the broader `min_critical_functional_level`
      (§9.1 is more specific than §3.7's generic "critical asset"
      wording, so it wins).
    - `mean_outcome`/`std_outcome` summarize `unmet_patient_hours` across
      rollouts — the one single-scalar demand-side outcome already used
      project-wide (`EpisodeMetrics.unmet_patient_hours`), picked as
      "the" outcome metric §3.7's abstract "mean, std" wording doesn't
      itself name.

    `risk/engine.py`'s `compute_risk` reads `rollouts` directly for
    dev doc §9.1's own Consequence formula (which needs the *failing*
    subset specifically), rather than being handed only this summary.
    """

    rollouts: list[RolloutOutcome]
    p_failure: float
    mean_outcome: float
    std_outcome: float


class RiskAssessment(BaseModel):
    """dev doc §9.1 (risk) + §9.2 (confidence): `risk_raw = p_failure x
    consequence` is the literal product before normalization; `risk` is
    that same value normalized to [0,1] (dev doc: "by the rule-based
    baseline's 95th percentile on val scenarios" — see `risk/engine.py`'s
    module docstring for the disclosed stand-in this session uses in
    place of a true held-out val suite, which doesn't exist yet, M3's
    own deferral). `risk/gate.py`'s tier decision reads `risk`/
    `confidence`; `risk_raw`/`consequence`/`p_failure` are kept for
    logging/recalibration, not read by the gate itself."""

    p_failure: float
    consequence: float
    risk_raw: float
    risk: float
    confidence: float


class RiskCalibration(BaseModel):
    """Reference values `risk/engine.py`'s `assess(...)` needs to
    normalize each of dev doc §9's raw signals to [0,1] — computed once
    (`risk/engine.py`'s `calibrate_from_samples`) and reused across
    every subsequent decision, not recomputed per call (recomputing per
    call would make the normalization scale silently drift decision to
    decision, defeating the point of a *fixed* reference — the same
    reason `configs/reward.yaml`'s normalizers are dev doc §5.4's own
    "frozen after Phase 3 and never changed mid-experiment")."""

    p95_risk_raw: float
    """dev doc §9.1's own wording: risk is "normalized to [0,1] by the
    rule-based baseline's 95th percentile on val scenarios"."""
    p95_disagreement_raw: float
    """Same normalization style, applied to §9.2's ensemble-disagreement
    signal — dev doc's own text only says "mapped to [0,1] via val-set
    quantiles" without naming which quantile; this session picks the
    95th, matching §9.1's own explicit choice for the sibling risk
    signal, rather than inventing an unrelated one."""
    conformal_threshold: float
    """§9.2's calibrated nonconformity threshold — already IS the
    interval's half-width (`risk/conformal.py`'s `interval_width`)."""
    p95_interval_width: float
    """Reference scale for normalizing interval width to [0,1] — see
    `risk/engine.py`'s `calibrate_from_samples` docstring for how this
    is estimated (bootstrap resampling of the calibration nonconformity
    scores, since a single calibration pass only ever produces one
    threshold, not a distribution to take a percentile of directly)."""


class AutonomyTier(str, Enum):
    """dev doc §9.3's gate output — one of three tiers, `HUMAN_APPROVAL`
    always winning when the router says OOD (`routing/ood.py`'s
    `RoutingDecision.path`, module M9b)."""

    AUTONOMOUS = "AUTONOMOUS"
    EXECUTE_AND_FLAG = "EXECUTE_AND_FLAG"
    HUMAN_APPROVAL = "HUMAN_APPROVAL"


# ---------------------------------------------------------------------------
# M9b — ID/OOD router (dev doc §6)
# ---------------------------------------------------------------------------
class RoutingDecision(BaseModel):
    """dev doc §6, verbatim: the router's output for one incident.
    `support_score` is the conformal p-value from `routing/ood.py`'s
    novelty check (dev doc's own naming — "conformal novelty p-value" —
    despite the field being called `support_score`, not `p_value`;
    kept exactly as dev-doc-named rather than renamed for clarity, since
    this is a frozen cross-boundary schema dev doc §10's decision log
    also embeds verbatim). `reasons` is a short human-readable audit
    trail — dev doc §10: "Every decision logs a full RoutingDecision"."""

    path: Literal["ID", "OOD"]
    registry_hit: bool
    support_score: float
    reasons: list[str] = Field(default_factory=list)


class RouterCalibration(BaseModel):
    """The reference data `routing/ood.py`'s `route()` needs for dev doc
    §6's novelty/support check — computed once per incident type
    (`scripts/calibrate_router.py`) and reused across every routing
    decision, same "calibrate once, reuse always" convention `risk/
    engine.py`'s `RiskCalibration` already established for M8.

    `training_features`: dev doc §6's "the training-scenario feature set
    of that type" — one row per training scenario, each row `[severity,
    footprint_area_km2, n_affected_assets, onset_hour_sin, onset_hour_
    cos, mean_criticality_of_affected]` (dev doc's own 6-dim `x`).
    `calibration_scores`: k-NN novelty scores computed on a *held-out*
    batch of scenarios (not the training set itself) — the distribution
    dev doc §6's conformal p-value is computed against."""

    incident_type: str
    env_version: str
    training_features: list[list[float]]
    calibration_scores: list[float]
    k: int = 5


# ---------------------------------------------------------------------------
# M9b — experiment configs (dev doc §15)
# ---------------------------------------------------------------------------
class ExperimentConfig(BaseModel):
    """dev doc §15: `configs/experiments/*.yaml`'s schema — "each
    experiment = one YAML under configs/experiments/, consumed by
    experiments/runner.py". Field types/values match dev doc §15's own
    `G_full.yaml` example exactly.

    **Disclosed scope:** this model makes the format real, frozen, and
    loadable (`experiments/config.py`'s `load_experiment_config`) —
    matching this project's own "frozen interface before the module
    that consumes it" precedent (`PolicyRegistryEntry`, M6b, built
    before M9b's router existed to read it). `experiments/runner.py`
    does NOT yet have a code path that reads one of these and actually
    dispatches to the right policy/planner/router/risk-gate combination
    end to end (loading a trained MAPPO policy, invoking the live LLM
    graph, gating on the risk engine, executing only when approved) —
    that full integration is dev doc's own Phase 10 scope ("Experiment
    G loop"), not this slice's.
    """

    experiment: str
    """A single letter, "A".."G" in the dev doc's own examples — kept as
    a plain string, not an enum, since dev doc §14's ablation ladder
    ("start from G_full.yaml, flip one field per run") implies more
    letters/combinations could exist later."""
    policy: Literal["rule", "ppo_single", "mappo", "mappo_constrained"]
    planner: Literal["none", "llm_direct", "llm_counterfactual"]
    router: Literal["enabled", "disabled"]
    risk_gate: Literal["enabled", "disabled"]
    human_model: dict[str, float] = Field(default_factory=dict)
    """dev doc's own literal YAML (`human_model: approve_if_pfail_lt:
    0.5`) isn't valid nested YAML as written — read here as shorthand
    for the nested mapping `configs/autonomy.yaml` already uses
    (`{"approve_if_pfail_lt": 0.5}`)."""
    suite: dict[str, Any]
    """dev doc's own shape: `{types: [...], split: "test"}` — loosely
    typed (a plain dict, not a nested model) since M3's own frozen
    train/val/test suite (`scenarios/suite.py`) is still deferred; this
    field documents the *shape* callers will eventually populate, not a
    real split that exists to validate against yet."""
    seeds: list[int]
