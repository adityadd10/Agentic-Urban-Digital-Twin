"""Scenario generation (dev doc §4.3, module M3 — flood-only scope).

Prototype-1 deviation: the dev doc's full frozen-suite machinery (20
train / 10 val / 10 test scenarios per incident type, written once and
never regenerated) is sized for the RL training pipeline (M5+), which
doesn't exist yet — there's no policy to train or evaluate against a
held-out split. This module implements just `generate_flood_scenario`
(single scenarios, for `scripts/run_flood_demo.py`); `suite.py`'s
frozen-suite-with-splits responsibility is deferred to whichever module
first needs train/val/test scenarios (M5 or M3's own later completion).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from udt.common.models import Incident, Scenario

DEFAULT_SEVERITY_RANGE = (0.5, 1.0)  # moderate-to-severe, prototype demo default


def generate_flood_scenario(
    *,
    scenario_id: str,
    ward_boundary_geojson: dict[str, Any],
    seed: int,
    onset_tick: int = 0,
    severity_range: tuple[float, float] = DEFAULT_SEVERITY_RANGE,
) -> Scenario:
    """Dev doc §4.3: samples parameter ranges for a flood incident. Uses
    the ward boundary as the incident's `location` (dev doc §4.1) — see
    `degradations/flood.py`'s docstring for why footprint isn't used as a
    separate per-tick spatial filter beyond that."""
    rng = np.random.default_rng(seed)
    severity = float(rng.uniform(*severity_range))

    incident = Incident(
        incident_id=f"{scenario_id}_incident",
        type="flood",
        location=ward_boundary_geojson["features"][0]["geometry"],
        onset_tick=onset_tick,
        severity=severity,
        raw_signal={"severity_range": list(severity_range)},
        directly_affected_assets=[],  # flood has no single directly-hit asset — spatial, not point
        profile={"growth_hours": 2.0, "hold_hours": 8.0, "recede_hours": 6.0},
    )
    return Scenario(
        scenario_id=scenario_id,
        incident=incident,
        initial_conditions={},
        seed=seed,
    )
