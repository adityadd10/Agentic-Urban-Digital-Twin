#!/usr/bin/env python3
"""Router calibration (dev doc §6), module M9b.

Builds a real `RouterCalibration` (`routing/ood.py`) for one incident
type: generates `--n-training` fresh scenarios, featurizes each
(`featurize_incident`), and keeps those feature vectors as the
"training-scenario feature set of that type" (dev doc §6). Then
generates `--n-calibration` more (disjoint seeds) and computes each
one's k-NN novelty score *against the training set* — that batch of
scores is what `routing/ood.py`'s `conformal_p_value` compares a live
incident's own score against.

**Disclosed, same status as `scripts/calibrate_risk.py`'s own docstring:**
no frozen train/val/test scenario suite exists yet (M3's own deferral)
— both batches here are freshly generated with disjoint seeds, not a
true held-out split.

**Disclosed, same status as `routing/ood.py`'s own module docstring:**
this session's only scenario generator (`generate_flood_scenario`)
produces feature vectors where 4 of 6 dimensions are constant — the
calibration this script produces is real and internally consistent, but
its discriminating power today rides almost entirely on `severity`
alone.

Usage:
  uv run python scripts/calibrate_router.py --config configs/data.yaml \\
      --incident-type flood --n-training 20 --n-calibration 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from _pipeline_common import configure_logging, load_config, resolve_path  # noqa: E402
from udt.common.models import DependencyGraph, RouterCalibration  # noqa: E402
from udt.routing.ood import DEFAULT_K, featurize_incident, knn_novelty_score  # noqa: E402
from udt.scenarios.generator import generate_flood_scenario  # noqa: E402

N_TRAINING_DEFAULT = 20
N_CALIBRATION_DEFAULT = 20
# Kept clear of the training batch's own seed range, same convention
# `scripts/train_mappo.py`'s `EVAL_SEED_OFFSET` already uses.
CALIBRATION_SEED_OFFSET = 100_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--incident-type", default="flood")
    parser.add_argument("--env-version", default="udt_multi_env_v0")
    parser.add_argument("--n-training", type=int, default=N_TRAINING_DEFAULT)
    parser.add_argument("--n-calibration", type=int, default=N_CALIBRATION_DEFAULT)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "configs" / "router_calibration.json"),
    )
    args = parser.parse_args()

    log = configure_logging()
    cfg = load_config(args.config)
    processed_dir = resolve_path(cfg["paths"]["processed_dir"])

    with (processed_dir / "dependency_graph.json").open() as f:
        dep_graph = DependencyGraph.model_validate(json.load(f))
    with (processed_dir / "ward_boundary.geojson").open() as f:
        ward_boundary = json.load(f)

    training_features = []
    for i in range(args.n_training):
        scenario = generate_flood_scenario(
            scenario_id=f"router_train_{i}",
            ward_boundary_geojson=ward_boundary,
            seed=args.base_seed + i,
        )
        x = featurize_incident(scenario.incident, dep_graph)
        training_features.append(x.tolist())
    log.info("router_training_features_built", n=len(training_features))

    training_array = np.array(training_features, dtype=np.float64)
    calibration_scores = []
    for i in range(args.n_calibration):
        scenario = generate_flood_scenario(
            scenario_id=f"router_calib_{i}",
            ward_boundary_geojson=ward_boundary,
            seed=args.base_seed + CALIBRATION_SEED_OFFSET + i,
        )
        x = featurize_incident(scenario.incident, dep_graph)
        score = knn_novelty_score(x, training_array, k=args.k)
        calibration_scores.append(score)
    log.info("router_calibration_scores_built", n=len(calibration_scores))

    calibration = RouterCalibration(
        incident_type=args.incident_type,
        env_version=args.env_version,
        training_features=training_features,
        calibration_scores=calibration_scores,
        k=args.k,
    )
    output_path = Path(args.output)
    output_path.write_text(calibration.model_dump_json(indent=2))
    log.info(
        "router_calibration_written",
        path=str(output_path),
        incident_type=args.incident_type,
        n_training=len(training_features),
        n_calibration=len(calibration_scores),
    )


if __name__ == "__main__":
    main()
