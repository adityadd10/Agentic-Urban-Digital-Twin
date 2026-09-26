"""LLM structured-output schemas (dev doc §7.3, module M9a).

`Plan` (ID path) and `ActionProposal` (OOD path) are the LLM→MARL/
LLM→twin contract — dev doc §7.3, verbatim field-for-field. Every LLM
call in `llm/graph.py` is validated against one of these (or the
internal `_PlanBatch` wrapper) before anything downstream sees it.

**Disclosed deviation from the general "every cross-component model
lives in `common/models.py`" rule (dev doc §12.2):** the dev doc's own
repository layout (§12.1) explicitly names `llm/schemas.py` as a real
file distinct from `common/models.py` — the one other place this
project's layout calls out a package-local schemas file this
explicitly (`incidents/`, `agents/`, etc. all put their shared models in
`common/models.py` and were built that way). `GoalWeights`/`Directive`/
`Plan`/`ActionProposal` are structured-output-validation schemas for a
specific LLM contract (retried against a raw string response, dev doc
§7.1), not general cross-module messages in the same sense as
`AgentAction`/`TwinState` — the dev doc's own explicit file listing is
followed here rather than the general rule.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from udt.common.models import AgentAction

# dev doc §7.3: "each in [0.5, 2.0]" — same box M6b's `envs/multi_env.py`
# already samples goals from (`GOAL_LOW`/`GOAL_HIGH`) — duplicated as a
# plain literal rather than imported, to keep `llm/` from depending on
# `envs/` (the dependency direction stays one-way: envs/agents -> twin,
# never twin/llm -> envs, matching this project's other established
# duplications, e.g. `experiments/runner.py`'s own `DECISION_INTERVAL_
# TICKS`).
GOAL_WEIGHT_MIN = 0.5
GOAL_WEIGHT_MAX = 2.0


class GoalWeights(BaseModel):
    """dev doc §7.3/§5.4's `g = (g_health, g_power, g_transport,
    g_cost)`, each bounded `[0.5, 2.0]`."""

    g_health: float = Field(ge=GOAL_WEIGHT_MIN, le=GOAL_WEIGHT_MAX)
    g_power: float = Field(ge=GOAL_WEIGHT_MIN, le=GOAL_WEIGHT_MAX)
    g_transport: float = Field(ge=GOAL_WEIGHT_MIN, le=GOAL_WEIGHT_MAX)
    g_cost: float = Field(ge=GOAL_WEIGHT_MIN, le=GOAL_WEIGHT_MAX)


class Directive(BaseModel):
    """dev doc §7.3: "coarse, <=6: e.g. {"kind":"prepare_transfer",
    "from":"H2","to":"H1"}" — the dev doc gives only an example shape,
    not a full schema, so `kind` is the one structured field (a short
    label for what this directive is about) and everything else is a
    free-form `details` dict, matching the example's own `from`/`to`
    keys without inventing a rigid per-kind schema the dev doc never
    specified. `kind` is not a Python enum (any string is schema-valid,
    so a plan can carry directives no current interpreter understands)
    but the vocabulary this codebase actually interprets is closed and
    small: `agents/plan_interpreter.py`'s four kinds (`protect_repair`,
    `protect_transfer`, `protect_dispatch`, `shed_bias`), one per
    `AgentAction` field (dev doc §7.3.1, added 2026-09-22 — supersedes
    this schema's original "advisory context [...] not executed
    directly" semantics, which held only until that interpreter
    existed). Any other `kind` is simply not acted on — logged, not an
    error."""

    kind: str
    details: dict[str, str] = Field(default_factory=dict)


class Plan(BaseModel):
    """dev doc §7.3, verbatim. `priority_assets` (<=5) and `directives`
    (<=6) are the dev doc's own explicit caps."""

    plan_id: str
    objective: str
    goal_weights: GoalWeights
    priority_assets: list[str] = Field(default_factory=list, max_length=5)
    directives: list[Directive] = Field(default_factory=list, max_length=6)
    rationale: str
    assumptions: list[str] = Field(default_factory=list)
    expected_outcome: str


class ActionProposal(BaseModel):
    """dev doc §7.3, verbatim — the OOD path's output. `confidence_note`
    is explicitly free text per the dev doc: "NEVER parsed as a number,
    NEVER used by the gate" — `risk/gate.py`'s `decide_tier` only ever
    reads the real, separately-computed `RiskAssessment.confidence`
    (§9.2), never anything an LLM wrote."""

    actions: list[AgentAction]
    rationale: str
    confidence_note: str
