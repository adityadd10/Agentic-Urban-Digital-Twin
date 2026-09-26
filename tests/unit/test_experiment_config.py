"""Phase 9 acceptance tests: `experiments/config.py`'s `load_experiment_
config` against every real `configs/experiments/*.yaml` (dev doc §15),
module M9b."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from config import load_experiment_config  # noqa: E402

CONFIGS_DIR = REPO_ROOT / "configs" / "experiments"


@pytest.mark.phase9
@pytest.mark.parametrize(
    "filename,expected_letter",
    [
        ("A_rule_based.yaml", "A"),
        ("B_ppo_single.yaml", "B"),
        ("C_mappo.yaml", "C"),
        ("D_llm_direct.yaml", "D"),
        ("E_llm_counterfactual.yaml", "E"),
        ("F_router_and_risk_gate.yaml", "F"),
        ("G_full.yaml", "G"),
    ],
)
def test_every_real_experiment_config_loads_and_validates(
    filename: str, expected_letter: str
) -> None:
    config = load_experiment_config(CONFIGS_DIR / filename)
    assert config.experiment == expected_letter
    assert config.seeds == [0, 1, 2, 3, 4]
    assert config.suite["types"] == ["flood"]


@pytest.mark.phase9
def test_ablation_ladder_flips_a_small_disclosed_set_of_fields_each_step() -> None:
    """dev doc §15: "flip one field per run" as the ladder's general
    philosophy — this session's own disclosed C->D->E->F->G assignment
    (`D_llm_direct.yaml`'s header) mostly does exactly that, except
    router+risk_gate are flipped together at the E->F step (deliberate,
    also disclosed): a router with no gate to act on its decision, or a
    gate with nothing feeding it an ID/OOD verdict, are both awkward
    half-configurations, not meaningfully separate ablation points."""
    letters = ["C", "D", "E", "F", "G"]
    configs = {
        letter: load_experiment_config(CONFIGS_DIR / filename)
        for letter, filename in zip(
            letters,
            [
                "C_mappo.yaml",
                "D_llm_direct.yaml",
                "E_llm_counterfactual.yaml",
                "F_router_and_risk_gate.yaml",
                "G_full.yaml",
            ],
            strict=True,
        )
    }
    fields = ["policy", "planner", "router", "risk_gate"]
    expected_diffs = {
        ("C", "D"): {"planner"},
        ("D", "E"): {"planner"},
        ("E", "F"): {"router", "risk_gate"},
        ("F", "G"): {"policy"},
    }
    for (prev_letter, next_letter), expected in expected_diffs.items():
        prev, nxt = configs[prev_letter], configs[next_letter]
        differing = {f for f in fields if getattr(prev, f) != getattr(nxt, f)}
        assert differing == expected, f"{prev_letter} -> {next_letter}: got {differing}"


@pytest.mark.phase9
def test_g_full_matches_dev_docs_own_example_modulo_disclosed_fixes() -> None:
    g = load_experiment_config(CONFIGS_DIR / "G_full.yaml")
    assert g.policy == "mappo_constrained"  # dev doc's own "marl_constrained" read as a typo
    assert g.planner == "llm_counterfactual"
    assert g.router == "enabled"
    assert g.risk_gate == "enabled"
    assert g.human_model == {"approve_if_pfail_lt": 0.5}
