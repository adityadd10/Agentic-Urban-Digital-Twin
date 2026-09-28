#!/usr/bin/env python3
"""Build the reward configs for ablations C and D (protocol 2026-09-28_reward_ablations.md §1).

D: configs/reward.yaml with `options.cascade_coefficient: 0.0`; nothing else changes.
C: configs/reward.yaml with `options.ambulance_delay_mode: pending_accrual`, and the
   ambulance normaliser refitted for that term by the dev doc §5.4 rule: the rule-based
   agent's mean |term| per tick over the frozen train split (same harness and scenarios
   as scripts/fit_reward_normalizers.py). All other normalisers are unchanged.
Refuses to overwrite existing files.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "tests" / "fixtures"))

from shapely.geometry import shape  # noqa: E402

from runner import run_episode  # noqa: E402
from udt.agents.rule_based import RuleBasedAgent  # noqa: E402
from udt.common.models import DependencyGraph  # noqa: E402
from udt.envs.reward import REWARD_CONFIG_PATH, tick_terms  # noqa: E402
from udt.incidents.degradations.flood import (  # noqa: E402
    SusceptibilityRaster,
    make_flood_degradation_fn,
)
from udt.scenarios.generator import apply_initial_conditions, onset_hour_of_day  # noqa: E402
from udt.scenarios.suite import DEFAULT_FLOOD_SUITE_DIR, load_suite  # noqa: E402
from udt.twin.road_network import RoadNetwork  # noqa: E402

C_PATH = REPO_ROOT / "configs/reward_C_pending_calls.yaml"
D_PATH = REPO_ROOT / "configs/reward_D_no_cascade.yaml"
DT_HOURS = 5.0 / 60.0


def main() -> None:
    for p in (C_PATH, D_PATH):
        if p.exists():
            raise SystemExit(f"{p} exists — ablation configs are frozen once written")
    a = yaml.safe_load(REWARD_CONFIG_PATH.read_text())
    protocol = "results/protocol/2026-09-28_reward_ablations.md"

    d = dict(a)
    d["options"] = {"ambulance_delay_mode": "pickup", "cascade_coefficient": 0.0}
    d["note"] = f"Ablation D ({protocol}): configs/reward.yaml with the cascade term removed."
    D_PATH.write_text(yaml.safe_dump(d, sort_keys=False))

    processed = REPO_ROOT / "data/processed"
    base_graph = DependencyGraph.model_validate_json(
        (processed / "dependency_graph.json").read_text()
    )
    ward = json.loads((processed / "ward_boundary.geojson").read_text())
    ward_polygon = shape(ward["features"][0]["geometry"])
    values: list[float] = []
    with SusceptibilityRaster(processed / "flood_susceptibility.tif") as raster:
        road_network = RoadNetwork.load(processed / "roads_full.graphml", raster)
        for scenario in load_suite(DEFAULT_FLOOD_SUITE_DIR, "train"):
            trace = run_episode(
                apply_initial_conditions(base_graph, scenario),
                RuleBasedAgent(),
                make_flood_degradation_fn(scenario.incident, raster),
                288,
                seed=scenario.seed,
                road_network=road_network,
                incident=scenario.incident,
                ward_polygon=ward_polygon,
                onset_hour=onset_hour_of_day(scenario),
            )
            for snap in trace:
                t = tick_terms(snap, DT_HOURS, 0, 0, ambulance_delay_mode="pending_accrual")
                values.append(abs(t["ambulance_response_delay_hours"]))
    mean = float(np.mean(values))
    c = dict(a)
    c["normalisers"] = {
        **a["normalisers"],
        "ambulance_response_delay_hours": mean if mean > 0 else 1.0,
    }
    c["options"] = {"ambulance_delay_mode": "pending_accrual", "cascade_coefficient": 5.0}
    c["ablation_C_fit"] = {
        "term": "pending_requests_count x dt (hours accrued per tick)",
        "rule_based_mean_abs_per_tick_train": mean,
        "created_at": datetime.now(UTC).isoformat(),
    }
    c["note"] = (
        f"Ablation C ({protocol}): configs/reward.yaml with unanswered calls accruing cost; "
        "ambulance normaliser refitted by the dev doc §5.4 rule on the train split."
    )
    C_PATH.write_text(yaml.safe_dump(c, sort_keys=False))
    a_norm = a["normalisers"]["ambulance_response_delay_hours"]
    print(f"D written; C written (ambulance normaliser {mean:.4f} vs A's {a_norm:.4f})")


if __name__ == "__main__":
    main()
