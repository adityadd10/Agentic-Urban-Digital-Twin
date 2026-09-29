"""Pass-rule statistics of the twin-v3 gates (scripts/gates_v3.py; protocol §4–§6, addendum 7)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from gates_v3 import (  # noqa: E402
    CRIT,
    DEATHS,
    UNMET,
    evaluate_gate,
    evaluate_headroom,
    holm,
    improvement,
    margin_for,
    per_scenario,
)


def _rows(values: dict[str, list[float | None]], metric: str) -> list[dict[str, object]]:
    return [{"scenario_id": s, metric: v} for s, vs in values.items() for v in vs]


def _paired(x: list[float], y: list[float], metrics: tuple[str, ...]) -> tuple[list, list]:
    xr = [{"scenario_id": f"s{i}", **{m: xv for m in metrics}} for i, xv in enumerate(x)]
    yr = [{"scenario_id": f"s{i}", **{m: yv for m in metrics}} for i, yv in enumerate(y)]
    return xr, yr


@pytest.mark.phase3
def test_per_scenario_means_seeds_and_skips_missing_values() -> None:
    rows = _rows({"a": [1.0, 3.0, None], "b": [None]}, DEATHS)
    assert per_scenario(rows, DEATHS) == {"a": 2.0}


@pytest.mark.phase3
def test_improvement_sign_follows_metric_direction() -> None:
    assert improvement({"s": 3.0}, {"s": 5.0}, DEATHS)[0] == 2.0  # fewer deaths is better
    assert improvement({"s": 0.9}, {"s": 0.7}, CRIT)[0] == pytest.approx(0.2)  # higher is better


@pytest.mark.phase3
def test_holm_step_down() -> None:
    assert holm([0.01, 0.04]) == [0.02, 0.04]
    assert holm([0.04, 0.01]) == [0.04, 0.02]


@pytest.mark.phase3
def test_margin_is_the_larger_of_twice_noise_and_practical_minimum() -> None:
    assert margin_for(DEATHS, noise=1.0, rbs_mean=60.0) == 3.0  # 5% of 60
    assert margin_for(DEATHS, noise=2.0, rbs_mean=60.0) == 4.0  # 2 x noise
    assert margin_for(CRIT, noise=0.001, rbs_mean=0.6) == 0.02  # absolute minimum


@pytest.mark.phase3
def test_directional_gate_needs_x_better_by_more_than_the_margin() -> None:
    rng = np.random.default_rng(1)
    y = list(60 + rng.normal(0, 1, 20))
    x_good = [v - 10 for v in y]
    x_bad = [v + 10 for v in y]
    margins = {DEATHS: 3.0, UNMET: 5.0}
    good = evaluate_gate(*_paired(x_good, y, (DEATHS, UNMET)), DEATHS, (UNMET,), True, margins)
    bad = evaluate_gate(*_paired(x_bad, y, (DEATHS, UNMET)), DEATHS, (UNMET,), True, margins)
    assert good["passes"] and good["mechanism"]["better"] == "X"
    assert not bad["passes"]


@pytest.mark.phase3
def test_two_sided_gate_passes_in_either_direction_and_outcome_must_follow() -> None:
    rng = np.random.default_rng(2)
    y = list(60 + rng.normal(0, 1, 20))
    x_worse = [v + 10 for v in y]  # Y is the better condition
    margins = {DEATHS: 3.0, UNMET: 5.0}
    res = evaluate_gate(*_paired(x_worse, y, (DEATHS, UNMET)), DEATHS, (UNMET,), False, margins)
    assert res["mechanism"]["better"] == "Y" and res["passes"]
    # mechanism favours Y but the outcome favours X: the gate fails
    xr, yr = _paired(x_worse, y, (DEATHS,))
    for r, v in zip(xr, [0.0] * 20, strict=True):
        r[UNMET] = v
    for r, v in zip(yr, [50.0] * 20, strict=True):
        r[UNMET] = v
    assert not evaluate_gate(xr, yr, DEATHS, (UNMET,), False, margins)["passes"]


@pytest.mark.phase3
def test_small_difference_inside_the_margin_fails() -> None:
    rng = np.random.default_rng(3)
    y = list(60 + rng.normal(0, 1, 20))
    x = [v - 1 for v in y]  # 1 death better, margin 3
    res = evaluate_gate(
        *_paired(x, y, (DEATHS, UNMET)), DEATHS, (UNMET,), False, {DEATHS: 3.0, UNMET: 5.0}
    )
    assert not res["mechanism"]["passes"] and not res["passes"]


@pytest.mark.phase3
def test_headroom_needs_fifteen_percent_and_a_positive_ci() -> None:
    rbs = _rows({f"s{i}": [100.0 + i] for i in range(20)}, DEATHS)
    ceil_big = _rows({f"s{i}": [80.0 + i] for i in range(20)}, DEATHS)  # 20% better
    ceil_small = _rows({f"s{i}": [95.0 + i] for i in range(20)}, DEATHS)  # 5% better
    assert evaluate_headroom(rbs, ceil_big)["passes"]
    assert not evaluate_headroom(rbs, ceil_small)["passes"]
