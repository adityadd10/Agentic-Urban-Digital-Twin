#!/usr/bin/env python3
"""Fit the reward-term normalisers (dev doc §5.4) and freeze them in configs/reward.yaml.

Runs the rule-based agent on every scenario of the frozen suite's **train**
split (never val/test), through the same `experiments.runner.run_episode`
harness Experiment A uses, and records each §5.4 term's mean absolute
per-tick value. A term that is always zero gets normaliser 1.0.

Refuses to overwrite an existing configs/reward.yaml: §5.4 says normalisers
are "never changed mid-experiment". Delete the file deliberately (and
retrain everything) to refit.

Usage:
  uv run python scripts/fit_reward_normalizers.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml
from shapely.geometry import shape

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "tests" / "fixtures"))

from _pipeline_common import configure_logging, load_config, resolve_path  # noqa: E402
from runner import run_episode  # noqa: E402
from udt.agents.rule_based import RuleBasedAgent  # noqa: E402
from udt.common.models import DependencyGraph  # noqa: E402
from udt.common.versions import ENV_VERSION  # noqa: E402
from udt.envs.reward import REWARD_CONFIG_PATH, TERMS, trace_terms  # noqa: E402
from udt.incidents.degradations.flood import (  # noqa: E402
    SusceptibilityRaster,
    make_flood_degradation_fn,
)
from udt.scenarios.generator import apply_initial_conditions, onset_hour_of_day  # noqa: E402
from udt.scenarios.suite import DEFAULT_FLOOD_SUITE_DIR, load_manifest, load_suite  # noqa: E402
from udt.twin.road_network import RoadNetwork  # noqa: E402

N_TICKS = 288


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    log = configure_logging()
    if REWARD_CONFIG_PATH.exists():
        raise SystemExit(f"{REWARD_CONFIG_PATH} exists — normalisers are frozen (dev doc §5.4)")

    processed = resolve_path(load_config(args.config)["paths"]["processed_dir"])
    base_graph = DependencyGraph.model_validate(
        json.loads((processed / "dependency_graph.json").read_text())
    )
    ward = json.loads((processed / "ward_boundary.geojson").read_text())
    ward_polygon = shape(ward["features"][0]["geometry"])
    scenarios = load_suite(DEFAULT_FLOOD_SUITE_DIR, "train")

    per_tick: dict[str, list[float]] = {k: [] for k in TERMS}
    with SusceptibilityRaster(processed / "flood_susceptibility.tif") as raster:
        road_network = RoadNetwork.load(processed / "roads_full.graphml", raster)
        for scenario in scenarios:
            trace = run_episode(
                apply_initial_conditions(base_graph, scenario),
                RuleBasedAgent(),
                make_flood_degradation_fn(scenario.incident, raster),
                N_TICKS,
                seed=scenario.seed,
                road_network=road_network,
                incident=scenario.incident,
                ward_polygon=ward_polygon,
                onset_hour=onset_hour_of_day(scenario),
            )
            for terms in trace_terms(trace, dt_hours=5.0 / 60.0):
                for k, v in terms.items():
                    per_tick[k].append(abs(v))
            log.info("normaliser_episode_done", scenario=scenario.scenario_id)

    means = {k: float(np.mean(v)) for k, v in per_tick.items()}
    normalisers = {k: (m if m > 0 else 1.0) for k, m in means.items()}
    manifest = load_manifest(DEFAULT_FLOOD_SUITE_DIR)
    doc = {
        "normalisers": normalisers,
        "raw_mean_abs_per_tick": means,
        "fitted_on": {
            "agent": "rule_based",
            "suite_version": manifest["suite_version"],
            "split": "train",
            "n_scenarios": len(scenarios),
            "n_ticks_per_episode": N_TICKS,
            "env_version": ENV_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
        },
        "note": "dev doc §5.4: frozen; never refit mid-experiment. Zero-mean terms get 1.0.",
    }
    REWARD_CONFIG_PATH.write_text(yaml.safe_dump(doc, sort_keys=False))
    print(yaml.safe_dump(doc, sort_keys=False))


if __name__ == "__main__":
    main()
