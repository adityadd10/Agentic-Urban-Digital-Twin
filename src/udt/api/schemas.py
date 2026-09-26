"""HTTP request/response schemas (dev doc §11.2), module M10b.

**Disclosed exception to "every cross-component model lives in
common/models.py":** same kind of exception `llm/schemas.py` (M9a) and
`logging/decision_log.py` (M10a) already established — these are
HTTP-transport-specific wrapper types (a request body, a response
envelope), not domain models that cross a module boundary inside the
twin/agents/risk/llm pipeline itself. Wherever a real domain model
already exists (`TwinState`, `DecisionRecord`, `AgentAction`), routes
return it directly rather than wrapping it in a redundant duplicate.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from udt.common.models import AgentAction, TwinState
from udt.logging.decision_log import DecisionRecord

# dev doc §4.3's own default range (`scenarios/generator.py`'s
# `DEFAULT_SEVERITY_RANGE`), duplicated here rather than imported so the
# API's own request schema doesn't reach into `scenarios/` just for one
# constant.
DEFAULT_SEVERITY_RANGE: tuple[float, float] = (0.5, 1.0)


class CreateEpisodeRequest(BaseModel):
    """dev doc §11.2's `POST /episodes` — "start episode from
    scenario_id". **Disclosed deviation:** no frozen scenario suite
    exists yet (M3's own deferral, unresolved all session), so there's
    no `scenario_id` to reference — this request instead carries what
    `scenarios/generator.py`'s `generate_flood_scenario` actually needs
    to build one fresh (`seed`, `severity_range`), same disclosed
    stand-in status every other "no frozen suite" workaround this
    session has used."""

    seed: int = 0
    severity_range: tuple[float, float] = DEFAULT_SEVERITY_RANGE
    n_ticks_max: int = 288  # dev doc §3.1: T_max = 24h at dt=5min
    constrained: bool = False
    """Mirrors `scripts/train_mappo.py`'s `--constrained` flag: turns on
    `constrained_reward`-equivalent behavior for this episode's risk/gate
    wiring where applicable. Currently informational only in this slice
    — the episode always runs the same `EpisodeSession` wiring regardless
    (disclosed, not silently ignored)."""


class CreateEpisodeResponse(BaseModel):
    episode_id: str
    incident_id: str
    severity: float


class StepResponse(BaseModel):
    """dev doc §11.2's `POST /episodes/{id}/step?n=1` response — the
    snapshots and any decisions produced by *this* call, not the whole
    episode's history so far (use `GET /decisions?episode_id=` for
    that)."""

    tick: int
    snapshots: list[TwinState]
    decisions: list[DecisionRecord]
    episode_ended: bool = False


class InjectIncidentRequest(BaseModel):
    """dev doc §11.2's `POST /incidents` — "inject incident (demo:
    'what if X happens now')". Replaces the incident driving an
    *existing* running episode from its next `step()` call forward —
    does not rewind or replay past ticks."""

    episode_id: str
    incident_type: str = "flood"
    severity: float = Field(ge=0.0, le=1.0)
    onset_tick: int | None = None
    """Defaults to the episode's current tick (the incident "happens
    now") when not given."""


class ApprovalOverrideRequest(BaseModel):
    """dev doc §11.2's `POST /approvals/{decision_id}` —
    `{approve|reject|modify, modified_action?}`.

    **Disclosed scope:** by the time a decision is logged, its approval
    has already been resolved headlessly (`risk/human_model.py`'s
    `HumanModel`, M10a) and its action has already executed against the
    twin — this endpoint overrides the *recorded* response for
    audit/annotation purposes (e.g. a human reviewer disagreeing after
    the fact), it does **not** rewind the simulation or re-execute a
    different action. A live, blocking pending-approval queue that
    pauses physics until a real human responds — the fuller reading of
    dev doc §11.3's "Approval modal" — needs the async coordination
    `risk/gate.py`'s own docstring already flags as not built yet
    (`TIMEOUT_TICKS_DEFAULT`/`safe_hold_action()` are the two stateless
    pieces that module can own on its own); this endpoint is the
    disclosed, real, simpler thing this slice ships instead.
    """

    response: Literal["approve", "reject", "modify"]
    modified_action: AgentAction | None = None
