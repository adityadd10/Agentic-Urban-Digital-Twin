"""Scripted human-in-the-loop stand-in (dev doc §9.3, module M8).

Dev doc §9.3: "In headless experiments a scripted HumanModel stands in
(default: approve-if-P_failure<0.5), so Experiment G runs unattended;
its behavior is config."

No real approval console/UI exists yet (M10, not built) — this is only
the *decision function* a headless experiment substitutes for an actual
human clicking approve/reject/modify, used wherever `risk/gate.py`'s
`decide_tier` returns `HUMAN_APPROVAL` during an unattended run.
"""

from __future__ import annotations

from dataclasses import dataclass

# dev doc §9.3's exact default.
DEFAULT_APPROVE_IF_PFAIL_LT = 0.5


@dataclass
class HumanModel:
    """`approve_if_pfail_lt` is config (dev doc: "its behavior is
    config") — `configs/experiments/*.yaml`'s own `human_model:
    approve_if_pfail_lt: 0.5` field feeds this constructor directly."""

    approve_if_pfail_lt: float = DEFAULT_APPROVE_IF_PFAIL_LT

    def decide(self, p_failure: float) -> bool:
        """Returns `True` (approve) iff `p_failure < approve_if_pfail_
        lt` — dev doc §9.3's literal rule, a strict less-than so a
        `p_failure` exactly at the threshold is rejected, matching
        `risk/gate.py`'s own strict-inequality convention for its
        thresholds."""
        return p_failure < self.approve_if_pfail_lt
