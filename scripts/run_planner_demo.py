#!/usr/bin/env python3
"""LLM planner demo (dev doc §7/§6), modules M9a + M9b.

Runs the real compiled LangGraph flow (`llm/graph.py`) against the real
Kurla dependency graph, using a real routing decision (`routing/ood.py`,
if a calibration file is available) and a **real, billed** Claude Sonnet
call (`llm/anthropic_client.py`), wrapped in `llm/cache.py`'s disk cache
so re-running this script on the same scenario never re-bills after the
first call.

**This is the one script in this codebase that makes a real network
call to a paid API.** Every automated test in this repo uses a scripted
`LLMClient` instead (dev doc §13: "CI never calls a live API") — this
script is the deliberate, explicit exception, run by a human who has
set `UDT_LLM_API_KEY` in their own `.env` and wants to see the whole
pipeline (twin -> router -> LLM planner -> constraint check -> risk
simulation) work end to end for real. It refuses to run with a helpful
message if that key isn't set, rather than silently substituting a fake
response and calling that a "demo".

Usage:
  # 1. Get a key at https://console.anthropic.com/, add to .env:
  #      UDT_LLM_API_KEY=sk-ant-...
  # 2. (optional, for a real routing decision) uv run python
  #      scripts/calibrate_router.py --config configs/data.yaml
  # 3. uv run python scripts/run_planner_demo.py --config configs/data.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from shapely.geometry import shape  # noqa: E402

from _pipeline_common import configure_logging, load_config, resolve_path  # noqa: E402
from udt.agents.marl.registry import load_registry  # noqa: E402
from udt.agents.rule_based import RuleBasedAgent  # noqa: E402
from udt.common.config import get_settings  # noqa: E402
from udt.common.models import DependencyGraph, RouterCalibration  # noqa: E402
from udt.common.versions import ENV_VERSION  # noqa: E402
from udt.incidents.degradations.flood import (  # noqa: E402
    SusceptibilityRaster,
    make_flood_degradation_fn,
)
from udt.llm.anthropic_client import AnthropicLLMClient  # noqa: E402
from udt.llm.cache import CachedLLMClient, DiskCache  # noqa: E402
from udt.llm.graph import build_planner_graph, run_planner  # noqa: E402
from udt.routing.ood import route  # noqa: E402
from udt.scenarios.generator import generate_flood_scenario  # noqa: E402
from udt.twin.road_network import RoadNetwork  # noqa: E402
from udt.twin.simulator import Simulator  # noqa: E402

WARMUP_TICKS = 12  # 1h at dt=5min - reach a live mid-incident state before planning


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--router-calibration",
        default=str(REPO_ROOT / "configs" / "router_calibration.json"),
    )
    parser.add_argument(
        "--cache-dir",
        default=str(REPO_ROOT / "runs" / "llm_cache"),
        help="dev doc §7.1's disk cache location",
    )
    args = parser.parse_args()

    log = configure_logging()
    settings = get_settings()
    if not settings.llm_api_key:
        print(
            "UDT_LLM_API_KEY is not set — this demo makes a real, billed API call and "
            "refuses to fake one. Add it to your .env (see this script's own docstring), "
            "then re-run.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    cfg = load_config(args.config)
    processed_dir = resolve_path(cfg["paths"]["processed_dir"])

    with (processed_dir / "dependency_graph.json").open() as f:
        base_graph = DependencyGraph.model_validate(json.load(f))
    with (processed_dir / "ward_boundary.geojson").open() as f:
        ward_boundary = json.load(f)
    shape(ward_boundary["features"][0]["geometry"])  # validated, not otherwise needed here

    roads_full_path = processed_dir / "roads_full.graphml"
    scenario = generate_flood_scenario(
        scenario_id="planner_demo", ward_boundary_geojson=ward_boundary, seed=args.seed
    )
    incident = scenario.incident

    with SusceptibilityRaster(processed_dir / "flood_susceptibility.tif") as raster:
        degradation_fn = make_flood_degradation_fn(incident, raster)
        road_network = (
            RoadNetwork.load(roads_full_path, raster) if roads_full_path.exists() else None
        )

        sim = Simulator(base_graph.model_copy(deep=True), seed=args.seed, road_network=road_network)
        agent = RuleBasedAgent()
        for _ in range(WARMUP_TICKS):
            action = agent.act(
                sim.graph, sim.tick, road_network=road_network, edge_states=sim.edge_states
            )
            sim.step(
                degradation_fn=degradation_fn,
                repair_target=action.repair_target,
                ambulance_assignment=action.ambulance_assignment,
                patient_transfer=action.patient_transfer,
                shed_tier=action.shed_tier,
            )

        calibration_path = Path(args.router_calibration)
        calibration = (
            RouterCalibration.model_validate_json(calibration_path.read_text())
            if calibration_path.exists()
            else None
        )
        registry_entries = load_registry(REPO_ROOT / "registry" / "policies.yaml")
        decision = route(
            incident,
            base_graph,
            env_version=ENV_VERSION,
            registry_entries=registry_entries,
            calibration=calibration,
        )
        log.info("routing_decision", **decision.model_dump())
        if calibration is None:
            log.warning(
                "no_router_calibration_found",
                hint="run scripts/calibrate_router.py first for a real routing decision "
                "instead of the always-OOD fallback",
                path=str(calibration_path),
            )

        anthropic_client = AnthropicLLMClient(
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            max_tokens_per_decision=settings.llm_max_tokens_per_decision,
        )
        cache = DiskCache(args.cache_dir)
        llm_client = CachedLLMClient(anthropic_client, cache, model=settings.llm_model)

        graph = build_planner_graph()
        output = run_planner(
            graph,
            llm_client=llm_client,
            sim=sim,
            incident=incident,
            routing_path=decision.path,
            degradation_fn=degradation_fn,
            road_network=road_network,
        )

    print(f"\nRouting decision: {decision.path} (support_score={decision.support_score:.3f})")
    print(f"LLM fallback triggered: {output.llm_fallback}")
    if output.selected_plan is not None:
        print(f"\nSelected plan: {output.selected_plan.plan_id}")
        print(f"  Objective: {output.selected_plan.objective}")
        print(f"  Goal weights: {output.selected_plan.goal_weights}")
        print(f"  Rationale: {output.selected_plan.rationale}")
    if output.action_proposal is not None:
        print(f"\nAction proposal: {output.action_proposal.actions}")
        print(f"  Rationale: {output.action_proposal.rationale}")
        print(f"  Constraint report passed: {output.constraint_report.passed}")  # type: ignore[union-attr]
    print(f"\nTokens used this decision: {anthropic_client.tokens_used_this_decision}")
    print(f"LLM cache directory: {args.cache_dir}")


if __name__ == "__main__":
    main()
