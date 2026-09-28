"""Reward-term normalisation (dev doc §5.4), shared by both envs.

§5.4: "All terms normalized to comparable magnitude (each term's running mean
absolute value ≈ 1 over the rule-based baseline; normalizers frozen in
`configs/reward.yaml` after Phase 3 and never changed mid-experiment)."
Until 2026-09-27 this was never implemented: raw terms were summed with the
§5.4 coefficients, so whichever term had the largest units dominated.

`scripts/fit_reward_normalizers.py` runs the rule-based agent over the frozen
suite's **train** split and writes each term's mean absolute per-tick value.
The envs divide each raw term by its normaliser before applying the §5.4
coefficients (and, in the multi-agent env, the goal weights). The attempted
safety-violation count is a penalty count and is not normalised.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from udt.common.config import REPO_ROOT
from udt.common.models import AssetType, TwinState
from udt.twin.power import SHED_FRACTION_BY_TIER

REWARD_CONFIG_PATH = REPO_ROOT / "configs" / "reward.yaml"
TERMS = (
    "unmet_patient_hours",
    "patient_deaths",
    "unserved_energy_mwh",
    "ambulance_response_delay_hours",
    "new_cascade_failures",
)


def load_reward_normalisers(path: str | Path | None = None) -> dict[str, float]:
    """Per-term normalisers from `configs/reward.yaml`. Missing file -> all 1.0
    (raw terms), which only happens before the normalisers are first fitted."""
    path = Path(path) if path is not None else REWARD_CONFIG_PATH
    if not path.exists():
        return dict.fromkeys(TERMS, 1.0)
    data: dict[str, Any] = yaml.safe_load(path.read_text())
    normalisers = {k: float(data["normalisers"][k]) for k in TERMS}
    if any(v <= 0 for v in normalisers.values()):
        raise ValueError(f"{path}: normalisers must be positive, got {normalisers}")
    return normalisers


AMBULANCE_MODES = ("pickup", "pending_accrual")


def load_reward_options(path: str | Path | None = None) -> dict[str, Any]:
    """Reward-design options (protocol 2026-09-28 reward ablations). Defaults
    reproduce the reward every reported policy was trained on:
    - `ambulance_delay_mode`: "pickup" = response time booked when a call is
      picked up (unanswered calls cost nothing); "pending_accrual" = every
      pending call accrues its waiting time each tick (condition C).
    - `cascade_coefficient`: weight of new cascade failures (5.0; condition D: 0)."""
    path = Path(path) if path is not None else REWARD_CONFIG_PATH
    data: dict[str, Any] = yaml.safe_load(path.read_text()) if path.exists() else {}
    options = data.get("options") or {}
    mode = str(options.get("ambulance_delay_mode", "pickup"))
    if mode not in AMBULANCE_MODES:
        raise ValueError(f"{path}: ambulance_delay_mode must be one of {AMBULANCE_MODES}")
    return {
        "ambulance_delay_mode": mode,
        "cascade_coefficient": float(options.get("cascade_coefficient", 5.0)),
    }


def tick_terms(
    snapshot: TwinState,
    dt_hours: float,
    prev_patient_deaths: int,
    prev_cascading_count: int,
    ambulance_delay_mode: str = "pickup",
) -> dict[str, float]:
    """The raw §5.4 terms for one tick (before normalisation/coefficients)."""
    queued = sum(
        int(a.attributes.get("patient_queue", 0))
        for a in snapshot.assets
        if a.asset_type == AssetType.HOSPITAL
    )
    unserved = 0.0
    for asset in snapshot.assets:
        if asset.asset_type == AssetType.SUBSTATION:
            shed = SHED_FRACTION_BY_TIER[int(asset.attributes.get("shed_tier", 0))]
            unserved += float(asset.attributes.get("load_mw", 0.0)) * shed * dt_hours
    return {
        "unmet_patient_hours": queued * dt_hours,
        "patient_deaths": float(snapshot.patient_deaths_cumulative - prev_patient_deaths),
        "unserved_energy_mwh": unserved,
        "ambulance_response_delay_hours": (
            float(snapshot.pending_requests_count) * dt_hours
            if ambulance_delay_mode == "pending_accrual"
            else float(sum(snapshot.ambulance_response_times_this_tick))
        ),
        "new_cascade_failures": float(
            max(0, snapshot.cascading_failure_count - prev_cascading_count)
        ),
    }


def trace_terms(trace: list[TwinState], dt_hours: float) -> list[dict[str, float]]:
    """`tick_terms` for every tick of a finished episode's trace."""
    out, deaths, cascades = [], 0, 0
    for snap in trace:
        out.append(tick_terms(snap, dt_hours, deaths, cascades))
        deaths, cascades = snap.patient_deaths_cumulative, snap.cascading_failure_count
    return out


def episode_reward(
    trace: list[TwinState],
    normalisers: dict[str, float],
    goal: tuple[float, float, float] = (1.0, 1.0, 1.0),
    dt_hours: float = 5.0 / 60.0,
) -> float:
    """Dev doc §5.4 reward summed over a finished episode's trace, with goal
    weights (g_health, g_power, g_transport); defaults = the §5.4 defaults.
    Excludes the safety-violation penalty (evaluation compares outcomes)."""
    g_health, g_power, g_transport = goal
    total = 0.0
    for t in trace_terms(trace, dt_hours):
        n = {k: v / normalisers[k] for k, v in t.items()}
        total -= (
            g_health * (n["unmet_patient_hours"] + 10.0 * n["patient_deaths"])
            + g_power * n["unserved_energy_mwh"]
            + g_transport * n["ambulance_response_delay_hours"]
            + 5.0 * n["new_cascade_failures"]
        )
    return total
