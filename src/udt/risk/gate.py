"""Autonomy gate (dev doc §9.3, module M8).

```
if routing.path == "OOD":                    tier = HUMAN_APPROVAL     # unconditional
elif risk < 0.2 and confidence > 0.8:        tier = AUTONOMOUS
elif risk < 0.5 and confidence > 0.5:        tier = EXECUTE_AND_FLAG
else:                                        tier = HUMAN_APPROVAL
```

Thresholds are config, not hardcoded (dev doc: "thresholds are config
because experiments sweep them") — `configs/autonomy.yaml` holds the
real numbers, `load_autonomy_config` reads it; `decide_tier`'s defaults
match that file's own values so a bare call without a config still
implements dev doc §9.3 exactly.

**Router disclosure (M9b, not built yet):** nothing in this codebase
produces an "OOD" routing path today — `routing_path` defaults to
`"ID"` (in-support). The OOD branch is real, tested code (call
`decide_tier(..., routing_path="OOD")` directly), just never triggered
by a live router until M9b exists.

**Approval flow, partially built:** dev doc §9.3's "Timeout (default 10
simulated min): take the pre-declared safe hold action [...] and
re-escalate" needs a live pending-approval timer loop, which belongs to
the decision orchestrator (M9/M10, not built yet) — `TIMEOUT_TICKS_
DEFAULT` and `safe_hold_action()` below are the two pieces of that
behavior this module *can* own on its own (a threshold constant and
"what is the safe hold action", both pure/stateless), not the timer
loop itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from udt.common.models import AgentAction, AutonomyTier

# dev doc §9.3's exact default thresholds.
DEFAULT_AUTONOMOUS_RISK_MAX = 0.2
DEFAULT_AUTONOMOUS_CONFIDENCE_MIN = 0.8
DEFAULT_EXECUTE_AND_FLAG_RISK_MAX = 0.5
DEFAULT_EXECUTE_AND_FLAG_CONFIDENCE_MIN = 0.5

# dev doc §9.3: "Timeout (default 10 simulated min)" at dt=5min/tick.
TIMEOUT_TICKS_DEFAULT = 2


def decide_tier(
    risk: float,
    confidence: float,
    routing_path: str = "ID",
    *,
    autonomous_risk_max: float = DEFAULT_AUTONOMOUS_RISK_MAX,
    autonomous_confidence_min: float = DEFAULT_AUTONOMOUS_CONFIDENCE_MIN,
    execute_and_flag_risk_max: float = DEFAULT_EXECUTE_AND_FLAG_RISK_MAX,
    execute_and_flag_confidence_min: float = DEFAULT_EXECUTE_AND_FLAG_CONFIDENCE_MIN,
) -> AutonomyTier:
    """dev doc §9.3's exact gate logic, thresholds overridable (they're
    config in the real pipeline — see `load_autonomy_config`)."""
    if routing_path == "OOD":
        return AutonomyTier.HUMAN_APPROVAL
    if risk < autonomous_risk_max and confidence > autonomous_confidence_min:
        return AutonomyTier.AUTONOMOUS
    if risk < execute_and_flag_risk_max and confidence > execute_and_flag_confidence_min:
        return AutonomyTier.EXECUTE_AND_FLAG
    return AutonomyTier.HUMAN_APPROVAL


def load_autonomy_config(path: str | Path) -> dict[str, Any]:
    """Reads `configs/autonomy.yaml`'s threshold block — a plain dict,
    not a pydantic model: this file is *config* (dev doc's own framing,
    "thresholds are config because experiments sweep them"), not a
    message crossing a module boundary, so it doesn't need dev doc
    §12.2's "every cross-boundary message is a pydantic model" rule
    (`AutonomyTier`/`RiskAssessment`, which DO cross a boundary, already
    are)."""
    with open(path) as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


def decide_tier_from_config(
    risk: float, confidence: float, routing_path: str, config: dict[str, Any]
) -> AutonomyTier:
    """Same as `decide_tier`, reading thresholds from a loaded
    `configs/autonomy.yaml` dict instead of keyword overrides."""
    return decide_tier(
        risk,
        confidence,
        routing_path,
        autonomous_risk_max=config["autonomous"]["risk_max"],
        autonomous_confidence_min=config["autonomous"]["confidence_min"],
        execute_and_flag_risk_max=config["execute_and_flag"]["risk_max"],
        execute_and_flag_confidence_min=config["execute_and_flag"]["confidence_min"],
    )


def safe_hold_action() -> AgentAction:
    """dev doc §9.3's "pre-declared safe hold action" for an approval
    timeout: "no new commitments; standing safe actions like accepting
    arrivals continue" — every `AgentAction` field at its default `None`
    is exactly that: `Simulator.step` treats a `None` field as "nothing
    new commanded this tick", while physics that was already running
    (buffers draining/refilling, patients already being admitted,
    ambulances already en route) keeps going regardless, matching the
    dev doc's own wording precisely."""
    return AgentAction()
