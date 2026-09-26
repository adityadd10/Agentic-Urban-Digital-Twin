"""Dev doc §5.4 reward normalisers (`envs/reward.py`, `configs/reward.yaml`)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from udt.common.models import Asset, AssetType, TwinState
from udt.envs.reward import (
    REWARD_CONFIG_PATH,
    TERMS,
    episode_reward,
    load_reward_normalisers,
    trace_terms,
)

POINT = {"type": "Point", "coordinates": [72.88, 19.07]}


def _snap(tick: int, queue: int, deaths: int, cascades: int, shed_tier: int = 0) -> TwinState:
    return TwinState(
        tick=tick,
        assets=[
            Asset(
                asset_id="H1",
                asset_type=AssetType.HOSPITAL,
                geometry=POINT,
                attributes={"patient_queue": queue},
            ),
            Asset(
                asset_id="S1",
                asset_type=AssetType.SUBSTATION,
                geometry=POINT,
                attributes={"load_mw": 12.0, "shed_tier": shed_tier},
            ),
        ],
        cascading_failure_count=cascades,
        patient_deaths_cumulative=deaths,
    )


@pytest.mark.phase5
def test_committed_normalisers_are_frozen_positive_and_from_train_split() -> None:
    normalisers = load_reward_normalisers()
    assert set(normalisers) == set(TERMS)
    assert all(v > 0 for v in normalisers.values())
    meta = yaml.safe_load(REWARD_CONFIG_PATH.read_text())["fitted_on"]
    assert meta["split"] == "train" and meta["suite_version"] == "flood_suite_v1"


@pytest.mark.phase5
def test_missing_config_means_raw_terms(tmp_path: Path) -> None:
    assert load_reward_normalisers(tmp_path / "absent.yaml") == dict.fromkeys(TERMS, 1.0)


@pytest.mark.phase5
def test_trace_terms_are_per_tick_deltas() -> None:
    trace = [_snap(0, queue=2, deaths=0, cascades=0), _snap(1, queue=0, deaths=3, cascades=1)]
    terms = trace_terms(trace, dt_hours=0.5)
    assert terms[0]["unmet_patient_hours"] == pytest.approx(1.0)
    assert terms[1]["patient_deaths"] == 3 and terms[1]["new_cascade_failures"] == 1
    assert terms[1]["unmet_patient_hours"] == 0.0


@pytest.mark.phase5
def test_episode_reward_applies_normalisers_and_coefficients() -> None:
    trace = [_snap(0, queue=0, deaths=0, cascades=0), _snap(1, queue=0, deaths=2, cascades=1)]
    ones = dict.fromkeys(TERMS, 1.0)
    # 2 deaths x 10 + 1 cascade x 5 = 25
    assert episode_reward(trace, ones, dt_hours=0.5) == pytest.approx(-25.0)
    halved = {**ones, "patient_deaths": 0.5}
    assert episode_reward(trace, halved, dt_hours=0.5) == pytest.approx(-45.0)
