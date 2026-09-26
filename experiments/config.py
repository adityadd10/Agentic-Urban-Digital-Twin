"""Experiment config loading (dev doc §15), module M9b.

`load_experiment_config` reads one `configs/experiments/*.yaml` and
validates it against `common/models.py`'s `ExperimentConfig` — see that
model's own docstring for what's real today (the format is frozen and
loadable) versus deferred (`experiments/runner.py` doesn't yet dispatch
on these fields end to end; that's Phase 10's job).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from udt.common.models import ExperimentConfig


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return ExperimentConfig.model_validate(data)
