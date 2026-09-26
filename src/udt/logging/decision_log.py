"""Decision log (dev doc §10), module M10a.

Deferred since M4 ("no agent produces a 'decision' with reasoning/plans
to log yet, just a single repair-target action; worth adding once M9's
LLM planner exists to actually log") — M9a/M9b now exist, and M10a's
`experiments/decision_loop.py` is the first caller that actually
produces the full `{routing, llm, plans, marl, constraint_report,
risk, approval}` bundle dev doc §10 describes.

"Append-only JSONL per episode (`runs/<run_id>/decisions.jsonl`) **and**
Postgres (deployment)." Only the JSONL half is built here — the
Postgres mirror needs M10b's API layer + the `decisions` table (dev doc
§11.1), neither of which exist yet; JSONL alone already serves dev
doc's own stated purposes ("thesis auditability claims, the autonomy-
correctness experiment, counterfactual-regret computation, and
debugging") for a single-process headless run.

**Disclosed exception to "every cross-component model lives in
common/models.py" (dev doc §12.2):** `DecisionRecord` and its sub-models
live here instead, the same kind of disclosed exception `llm/schemas.py`
already established for M9a — `DecisionRecord` composes `Plan`/
`ActionProposal` (`llm/schemas.py`) with `ConstraintReport`/
`RiskAssessment`/`AutonomyTier`/`AgentAction` (`common/models.py`); it
sits *above* both, so putting it in `common/models.py` would need
`common/models.py` to import from `llm/schemas.py` — the wrong
dependency direction (`llm` already depends on `common`, not the
reverse) — while putting it here, where it's actually consumed, avoids
that entirely.

**`realized_outcome_next_interval` is disclosed as always `None` in
this build.** Dev doc's own field name implies a two-pass write (append
now, patch in the *next* interval's realized outcome once it's known) —
a real, buildable feature, just not built this slice; every other field
in `DecisionRecord` is populated for real.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, Field

from udt.common.models import (
    AgentAction,
    AutonomyTier,
    ConstraintReport,
    RoutingDecision,
    SimulationResult,
)
from udt.llm.schemas import Plan
from udt.twin.simulator import Simulator


def twin_state_digest(sim: Simulator) -> str:
    """dev doc §10's `twin_state_digest` — a short, stable hash of the
    current twin state (every asset's current functional level, sorted
    by asset id for determinism). Truncated to 16 hex chars: this is a
    log-readability aid for spotting "did the state actually change
    between two decisions", not a cryptographic commitment that needs
    the full 256 bits."""
    levels = sorted(
        (asset_id, round(sim.graph.nodes[asset_id]["asset"].functional_level, 6))
        for asset_id in sim.graph.nodes
    )
    payload = json.dumps(levels, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


class LLMCallSummary(BaseModel):
    """dev doc §10's `llm{prompt_version, model, tokens, cache_hit,
    fallback}` block. `prompt_version`/`cache_hit` are `None`/`False`
    when nothing was actually asked (e.g. `generate_plans` fell back to
    the rule-based baseline plan without a valid LLM response to
    version) — disclosed via the field values themselves, not a
    separate flag."""

    prompt_version: str | None = None
    model: str | None = None
    tokens: int = 0
    cache_hit: bool = False
    fallback: bool = False


class PlanSummary(BaseModel):
    """dev doc §10's `plans[{plan, simulation_summary}]` entries — one
    per ID-path candidate plan `simulate_each` actually scored."""

    plan: Plan
    simulation_summary: SimulationResult


class MarlActionSummary(BaseModel):
    """dev doc §10's `marl{policy_id, action, masked_actions_count}`
    block. `masked_actions_count` is the number of discrete choices
    `envs/multi_env.py`'s `action_mask()` ruled out before the policy
    ever sampled — `0` for any executor that isn't a masked MARL policy
    (e.g. `RuleBasedPolicyExecutor`, disclosed in that class's own
    docstring), not a claim that masking ran and found nothing."""

    policy_id: str
    action: dict[str, object]
    masked_actions_count: int = 0


class RiskSummary(BaseModel):
    """dev doc §10's `risk{P_failure, consequence, risk, confidence,
    tier}` block — deliberately narrower than `common/models.py`'s full
    `RiskAssessment` (which also carries `risk_raw` for calibration
    bookkeeping): this log matches dev doc §10's own literal field list,
    not everything `risk/engine.py` happens to compute internally."""

    p_failure: float
    consequence: float
    risk: float
    confidence: float
    tier: AutonomyTier


class ApprovalSummary(BaseModel):
    """dev doc §10's `approval{required, response, latency,
    timeout_triggered}` block. `latency` is wall-clock seconds between
    the gate's verdict and the approval response — always ~0 in a
    headless run (`risk/human_model.py`'s `HumanModel` answers
    instantly); `timeout_triggered` is always `False` here for the same
    reason (`risk/gate.py`'s `TIMEOUT_TICKS_DEFAULT` governs a live
    pending-approval loop that doesn't exist yet, see that module's own
    docstring)."""

    required: bool
    response: str | None = None
    latency: float = 0.0
    timeout_triggered: bool = False


class DecisionRecord(BaseModel):
    """dev doc §10, verbatim field list (see module docstring for why
    this lives here, not `common/models.py`). One record per decision
    interval — `experiments/decision_loop.py` builds and appends one of
    these every `DECISION_INTERVAL_TICKS` ticks."""

    decision_id: str
    episode_id: str
    tick: int
    twin_state_digest: str
    incident_id: str
    routing: RoutingDecision
    llm: LLMCallSummary
    plans: list[PlanSummary] = Field(default_factory=list)
    selected_plan_id: str | None = None
    marl: MarlActionSummary
    constraint_report: ConstraintReport
    risk: RiskSummary
    approval: ApprovalSummary
    executed_action: AgentAction
    realized_outcome_next_interval: dict[str, float] | None = None


class DecisionLogWriter:
    """Append-only JSONL writer (dev doc §10) — one file per episode,
    one line per decision, never rewritten (`realized_outcome_next_
    interval` staying `None` is exactly what keeps this true; see
    module docstring)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: DecisionRecord) -> None:
        with self.path.open("a") as f:
            f.write(record.model_dump_json())
            f.write("\n")


def load_decision_log(path: str | Path) -> list[DecisionRecord]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as f:
        return [DecisionRecord.model_validate_json(line) for line in f if line.strip()]


def build_decision_id(episode_id: str, tick: int) -> str:
    return f"{episode_id}_tick{tick}"
