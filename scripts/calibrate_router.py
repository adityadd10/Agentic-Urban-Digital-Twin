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

**2026-09-27:** now uses the frozen suite: training features = the 20
train scenarios, calibration scores = the 10 val scenarios, each with its
own onset hour. Footprint area now varies per scenario (Gaussian rainfall
footprint), so 4 of 6 features vary (severity, footprint area, onset sin/cos);
`n_affected_assets` and `mean_criticality_of_affected` are still constant.

Usage:
  uv run python scripts/calibrate_router.py --config configs/data.yaml
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
from udt.common.versions import ENV_VERSION  # noqa: E402
from udt.routing.ood import DEFAULT_K, featurize_incident, knn_novelty_score  # noqa: E402
from udt.scenarios.generator import onset_hour_of_day  # noqa: E402
from udt.scenarios.suite import DEFAULT_FLOOD_SUITE_DIR, load_suite  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--incident-type", default="flood")
    parser.add_argument("--env-version", default=ENV_VERSION)
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

    # 2026-09-27: training features from the frozen suite's train split,
    # calibration scores from its val split (dev doc §4.3, §6).
    training_features = [
        featurize_incident(sc.incident, dep_graph, onset_hour_of_day=onset_hour_of_day(sc)).tolist()
        for sc in load_suite(DEFAULT_FLOOD_SUITE_DIR, "train")
    ]
    log.info("router_training_features_built", n=len(training_features))

    training_array = np.array(training_features, dtype=np.float64)
    calibration_scores = [
        knn_novelty_score(
            featurize_incident(sc.incident, dep_graph, onset_hour_of_day=onset_hour_of_day(sc)),
            training_array,
            k=args.k,
        )
        for sc in load_suite(DEFAULT_FLOOD_SUITE_DIR, "val")
    ]
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
