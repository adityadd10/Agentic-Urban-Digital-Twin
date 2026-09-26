#!/usr/bin/env python3
"""Risk calibration (dev doc §9.1/§9.2, module M8).

Builds a real `RiskCalibration` (`risk/engine.py`'s `calibrate_from_
samples`) from a small batch of freshly generated flood scenarios run
under the rule-based baseline — a disclosed stand-in for the dev doc's
true held-out "val scenarios"/"val episodes" (no frozen train/val/test
scenario suite exists yet, M3's own deferral), same status as every
other "not a true held-out split" disclosure this session (e.g. M6b's
`val_score_is_true_holdout=False`).

For each of `--n-scenarios` scenarios:
1. Run `RuleBasedAgent` for `WARMUP_TICKS` to reach a live mid-incident
   state (not tick 0 — a counterfactual asked at the very start of an
   incident isn't representative of a "mid-crisis decision").
2. Call `twin/counterfactual.py`'s `simulate()` from that state, using
   the agent's *own current decision*, for `HORIZON_TICKS` ahead ->
   the "predicted" `SimulationResult`.
3. `risk/engine.py`'s `compute_risk` (fed a placeholder all-zero
   calibration, since raw risk doesn't depend on it — see that
   function's return tuple) gives one `risk_raw` sample.
4. Separately, clone the same state and run the *real* rule-based agent
   forward for the same `HORIZON_TICKS`, decision-by-decision, to get
   the "realized" outcome. Dev doc §9.2's nonconformity score is
   `|simulated mean outcome - realized outcome|` — computed here as
   `|result.mean_outcome - realized_unmet_patient_hours|`.
5. Disagreement samples: dev doc §9.2 wants "K=5 critics from the 5
   training seeds" — this session has no 5 real trained MAPPO seeds
   (no training run at the real budget yet). Stand-in, disclosed the
   same way `risk/ensemble.py`'s module docstring is: 5 freshly-
   initialized (never-trained) `CentralizedCritic` networks score the
   *same* real feature vector (the critical assets' current functional
   levels) — real networks, real forward passes, just not the
   literal "5 training seeds" dev doc §9.2 asks for.

**2026-09-27:** calibrates on the frozen suite's **val** split
(`flood_suite_v1`, dev doc §4.3), one sample per val scenario, with each
scenario's own initial conditions. This replaces the freshly generated
stand-in scenarios described above. The disagreement stand-in (untrained
critics) is unchanged until trained MAPPO critics exist.

Usage:
  uv run python scripts/calibrate_risk.py --config configs/data.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from shapely.geometry import shape  # noqa: E402

from _pipeline_common import configure_logging, load_config, resolve_path  # noqa: E402
from udt.agents.marl.networks import CentralizedCritic  # noqa: E402
from udt.agents.rule_based import RuleBasedAgent  # noqa: E402
from udt.common.models import AssetType, DependencyGraph, RiskCalibration, Scenario  # noqa: E402
from udt.incidents.degradations.flood import (  # noqa: E402
    SusceptibilityRaster,
    make_flood_degradation_fn,
)
from udt.risk.engine import calibrate_from_samples, compute_risk  # noqa: E402
from udt.risk.ensemble import compute_disagreement  # noqa: E402
from udt.scenarios.generator import apply_initial_conditions, onset_hour_of_day  # noqa: E402
from udt.scenarios.suite import DEFAULT_FLOOD_SUITE_DIR, load_suite  # noqa: E402
from udt.twin.counterfactual import (  # noqa: E402
    HORIZON_TICKS_DEFAULT,
    N_ROLLOUTS_DEFAULT,
    simulate,
)
from udt.twin.road_network import RoadNetwork  # noqa: E402
from udt.twin.simulator import Simulator  # noqa: E402

WARMUP_TICKS = 12  # 1h at dt=5min - reach a live mid-incident state before calibrating
N_CRITICS = 5  # dev doc §9.2's "K=5", see module docstring's disclosed stand-in
CRITIC_INPUT_DIM = 8  # fixed-size feature vector, see _critic_feature_vector

_PLACEHOLDER_CALIBRATION = RiskCalibration(
    p95_risk_raw=0.0, p95_disagreement_raw=0.0, conformal_threshold=0.0, p95_interval_width=0.0
)


def _critic_feature_vector(sim: Simulator) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Fixed-size (`CRITIC_INPUT_DIM`) feature vector for the disclosed
    stand-in critics: the current functional level of up to
    `CRITIC_INPUT_DIM` critical assets, padded with 1.0 (healthy) if
    there are fewer."""
    critical_types = {AssetType.HOSPITAL, AssetType.SUBSTATION, AssetType.WATER}
    levels = [
        sim.graph.nodes[a]["asset"].functional_level
        for a in sim.graph.nodes
        if sim.graph.nodes[a]["asset"].asset_type in critical_types
    ][:CRITIC_INPUT_DIM]
    levels += [1.0] * (CRITIC_INPUT_DIM - len(levels))
    return np.array(levels, dtype=np.float32)


def _rollout_realized_unmet_patient_hours(
    sim: Simulator, agent: RuleBasedAgent, degradation_fn: object, road_network: RoadNetwork | None
) -> float:
    total = 0.0
    for _ in range(HORIZON_TICKS_DEFAULT):
        action = agent.act(
            sim.graph, sim.tick, road_network=road_network, edge_states=sim.edge_states
        )
        snapshot = sim.step(
            degradation_fn=degradation_fn,  # type: ignore[arg-type]
            repair_target=action.repair_target,
            ambulance_assignment=action.ambulance_assignment,
            patient_transfer=action.patient_transfer,
            shed_tier=action.shed_tier,
        )
        queued = sum(
            int(a.attributes.get("patient_queue", 0))
            for a in snapshot.assets
            if a.asset_type == AssetType.HOSPITAL
        )
        total += queued * sim.dt_hours
    return total


def _collect_one_scenario(
    base_graph: DependencyGraph,
    raster: SusceptibilityRaster,
    road_network: RoadNetwork | None,
    scenario: Scenario,
    critics: list[CentralizedCritic],
) -> tuple[float, float, float]:
    """Returns `(risk_raw, disagreement_raw, nonconformity)` for one
    scenario."""
    seed = scenario.seed
    incident = scenario.incident
    degradation_fn = make_flood_degradation_fn(incident, raster)
    agent = RuleBasedAgent()

    sim = Simulator(
        apply_initial_conditions(base_graph, scenario),
        seed=seed,
        road_network=road_network,
        onset_hour_of_day=onset_hour_of_day(scenario),
    )
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

    decision = agent.act(
        sim.graph, sim.tick, road_network=road_network, edge_states=sim.edge_states
    )
    result = simulate(
        sim,
        decision,
        degradation_fn=degradation_fn,
        incident=incident,
        horizon_ticks=HORIZON_TICKS_DEFAULT,
        n_rollouts=N_ROLLOUTS_DEFAULT,
        base_seed=seed * 1000,
    )
    _, _, risk_raw, _ = compute_risk(result, _PLACEHOLDER_CALIBRATION)

    realized_sim = sim.clone_for_counterfactual(seed=seed * 1000 + 999)
    realized_unmet = _rollout_realized_unmet_patient_hours(
        realized_sim, agent, degradation_fn, road_network
    )
    nonconformity = abs(result.mean_outcome - realized_unmet)

    feature_vector = torch.as_tensor(_critic_feature_vector(sim))
    with torch.no_grad():
        critic_returns = [float(critic(feature_vector.unsqueeze(0)).item()) for critic in critics]
    disagreement_raw = compute_disagreement(critic_returns)

    return risk_raw, disagreement_raw, nonconformity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--split", default="val", help="frozen-suite split (dev doc §4.3)")
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "configs" / "risk_calibration.json"),
        help="where to write the calibrated RiskCalibration",
    )
    args = parser.parse_args()

    log = configure_logging()
    cfg = load_config(args.config)
    processed_dir = resolve_path(cfg["paths"]["processed_dir"])

    dep_graph_path = processed_dir / "dependency_graph.json"
    with dep_graph_path.open() as f:
        base_graph = DependencyGraph.model_validate(json.load(f))

    susceptibility_path = processed_dir / "flood_susceptibility.tif"
    with (processed_dir / "ward_boundary.geojson").open() as f:
        ward_boundary = json.load(f)
    shape(ward_boundary["features"][0]["geometry"])  # validated, not otherwise needed here

    roads_full_path = processed_dir / "roads_full.graphml"

    torch.manual_seed(0)
    critics = [CentralizedCritic(global_state_dim=CRITIC_INPUT_DIM) for _ in range(N_CRITICS)]

    risk_raw_samples: list[float] = []
    disagreement_raw_samples: list[float] = []
    nonconformity_scores: list[float] = []

    with SusceptibilityRaster(susceptibility_path) as raster:
        road_network = (
            RoadNetwork.load(roads_full_path, raster) if roads_full_path.exists() else None
        )
        for scenario in load_suite(DEFAULT_FLOOD_SUITE_DIR, args.split):
            seed = scenario.seed
            risk_raw, disagreement_raw, nonconformity = _collect_one_scenario(
                base_graph, raster, road_network, scenario, critics
            )
            risk_raw_samples.append(risk_raw)
            disagreement_raw_samples.append(disagreement_raw)
            nonconformity_scores.append(nonconformity)
            log.info(
                "calibration_scenario_complete",
                seed=seed,
                risk_raw=risk_raw,
                disagreement_raw=disagreement_raw,
                nonconformity=nonconformity,
            )

    calibration = calibrate_from_samples(
        risk_raw_samples, disagreement_raw_samples, nonconformity_scores
    )
    output_path = Path(args.output)
    output_path.write_text(calibration.model_dump_json(indent=2))
    log.info("risk_calibration_written", path=str(output_path), **calibration.model_dump())


if __name__ == "__main__":
    main()
