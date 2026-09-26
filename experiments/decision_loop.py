"""Full decision loop, headless (dev doc §10/§11's Experiment G, module
M10a — "Experiment G config running end-to-end without UI").

Thin wrapper around `udt.api.episode_session.EpisodeSession` — the real
"route -> plan -> execute -> constraint-check -> risk-assess -> gate ->
log" loop now lives there (module M10b factored it out of what was
originally a single big function here, so a live API server can also
advance an episode a few ticks at a time between requests, not just run
a whole episode in one blocking call). This module just builds one
`EpisodeSession` and runs it to completion in one call — the shape
every headless experiment script in this codebase already expects.

See `EpisodeSession`'s own module docstring for the disclosed scope:
`PolicyExecutor`'s one real implementation (`RuleBasedPolicyExecutor`,
which — since 2026-09-22, dev doc §7.3.1 — interprets a selected `Plan`
into a real biased action via `agents/plan_interpreter.py` rather than
ignoring it; still not a trained constrained-MARL checkpoint, which
doesn't exist at usable quality yet), `critic_returns` as a
caller-supplied stand-in (no 5 trained MAPPO seeds exist), and the
tick-physics ordering requirement.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from udt.api.episode_session import (  # noqa: E402
    EpisodeSession,
    PolicyDecision,
    PolicyExecutor,
    RuleBasedPolicyExecutor,
)
from udt.common.models import (  # noqa: E402
    DependencyGraph,
    Incident,
    PolicyRegistryEntry,
    RiskCalibration,
    RouterCalibration,
    TwinState,
)
from udt.llm.graph import LLMClient  # noqa: E402
from udt.logging.decision_log import DecisionRecord  # noqa: E402
from udt.risk.human_model import HumanModel  # noqa: E402
from udt.twin.road_network import RoadNetwork  # noqa: E402
from udt.twin.simulator import DegradationFn, Simulator  # noqa: E402

__all__ = [
    "PolicyDecision",
    "PolicyExecutor",
    "RuleBasedPolicyExecutor",
    "DecisionLoopResult",
    "run_decision_loop_episode",
]


@dataclass
class DecisionLoopResult:
    trace: list[TwinState]
    decisions: list[DecisionRecord]


def run_decision_loop_episode(
    sim: Simulator,
    incident: Incident,
    dep_graph: DependencyGraph,
    *,
    degradation_fn: DegradationFn | None,
    road_network: RoadNetwork | None,
    ward_polygon: object | None,
    llm_client: LLMClient,
    policy_executor: PolicyExecutor,
    env_version: str,
    registry_entries: list[PolicyRegistryEntry],
    router_calibration: RouterCalibration | None,
    risk_calibration: RiskCalibration,
    critic_returns: list[float],
    human_model: HumanModel,
    n_ticks: int,
    episode_id: str,
    decision_log_path: str | Path,
) -> DecisionLoopResult:
    """Runs one full episode, headless, decision by decision. See module
    docstring for where the actual loop logic lives."""
    session = EpisodeSession(
        episode_id=episode_id,
        sim=sim,
        incident=incident,
        dep_graph=dep_graph,
        degradation_fn=degradation_fn,
        road_network=road_network,
        ward_polygon=ward_polygon,
        llm_client=llm_client,
        policy_executor=policy_executor,
        env_version=env_version,
        registry_entries=registry_entries,
        router_calibration=router_calibration,
        risk_calibration=risk_calibration,
        critic_returns=critic_returns,
        human_model=human_model,
        decision_log_path=decision_log_path,
    )
    session.step(n_ticks)
    return DecisionLoopResult(trace=session.trace, decisions=session.decisions)
