#!/usr/bin/env python3
"""Load-shedding pre-check (protocol 2026-09-29_twin_v3_success_criteria, addendum 2).

  run --cond on|off   frozen RB-S with its shedding rule (on) or with every shed tier
                      forced to 0 (off); 20 train scenarios x seeds k = 0, 1, 2
  analyse             pre-registered rule; writes results/twin_v3_gates/shedding_precheck.*

Rule: shedding is KEPT if, for deaths or unmet patient-hours, the paired difference
(on - off, per-scenario means over the 3 seeds) has a 95% bootstrap CI excluding 0,
a Holm-adjusted two-sided Wilcoxon p < 0.05 (Holm across the two metrics), and
|mean difference| >= 5% of RB-S's mean. Otherwise the power agent's shed head is removed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from tune_rbs_v3 import EVAL_SEED_STRIDE, EVAL_SEEDS, SUITE  # noqa: E402
from udt.agents.rbs_v3 import RuleBasedStrongV3  # noqa: E402
from udt.envs.multi_env_v3 import AGENT_POWER, UDTMultiAgentEnvV3  # noqa: E402
from udt.logging.metrics import compute_episode_metrics  # noqa: E402

RUN_DIR = REPO_ROOT / "runs" / "shedding_precheck_v3"
OUT = REPO_ROOT / "results" / "twin_v3_gates"
METRICS = ("patient_deaths", "unmet_patient_hours")
N_BOOT = 10_000
MIN_SHARE = 0.05


def run(cond: str) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    env = UDTMultiAgentEnvV3(
        processed_dir=REPO_ROOT / "data/processed",
        reward_config=REPO_ROOT / "configs/reward.yaml",
        suite_dir=SUITE,
        scenario_split="train",
        shedding_enabled=True,  # as run (before the pre-check removed the shed head)
    )
    rbs = RuleBasedStrongV3.frozen()
    rows = []
    for index, sc in enumerate(env._scenarios):
        for k in range(EVAL_SEEDS):
            seed = sc.seed + k * EVAL_SEED_STRIDE
            env.reset(seed=seed, options={"scenario_index": index, "goal": [1, 1, 1, 1]})
            done = False
            while not done:
                acts = rbs.act(env)
                if cond == "off":
                    acts[AGENT_POWER][: len(env._substation_ids)] = 0
                _, _, term, trunc, _ = env.step(acts)
                done = any(term.values()) or any(trunc.values())
            m = compute_episode_metrics(sc.scenario_id, f"rbs_shed_{cond}", env.episode_trace)
            rows.append({**m.model_dump(), "eval_seed": seed})
    (RUN_DIR / f"{cond}.json").write_text(json.dumps(rows))


def _per_scenario(rows: list[dict[str, Any]], metric: str) -> dict[str, float]:
    out: dict[str, list[float]] = {}
    for r in rows:
        out.setdefault(r["scenario_id"], []).append(float(r[metric]))
    return {k: float(np.mean(v)) for k, v in out.items()}


def analyse() -> None:
    on = json.loads((RUN_DIR / "on.json").read_text())
    off = json.loads((RUN_DIR / "off.json").read_text())
    rng = np.random.default_rng(0)
    res: dict[str, Any] = {}
    for metric in METRICS:
        a, b = _per_scenario(on, metric), _per_scenario(off, metric)
        ids = sorted(a)
        d = np.array([a[i] - b[i] for i in ids])
        boot = d[rng.integers(0, len(d), size=(N_BOOT, len(d)))].mean(axis=1)
        p = 1.0 if np.allclose(d, 0) else float(wilcoxon(d).pvalue)
        res[metric] = {
            "on_mean": float(np.mean(list(a.values()))),
            "off_mean": float(np.mean(list(b.values()))),
            "diff_on_minus_off": float(d.mean()),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "wilcoxon_p": p,
            "n_scenarios": len(d),
        }
    order = sorted(METRICS, key=lambda m: res[m]["wilcoxon_p"])  # Holm
    for rank, m in enumerate(order):
        res[m]["holm_p"] = min(
            1.0,
            max(res[o]["wilcoxon_p"] * (len(order) - i) for i, o in enumerate(order[: rank + 1])),
        )
    for m in METRICS:
        r = res[m]
        ci_excl = r["ci95"][0] > 0 or r["ci95"][1] < 0
        big = abs(r["diff_on_minus_off"]) >= MIN_SHARE * r["on_mean"]
        r["passes"] = bool(ci_excl and r["holm_p"] < 0.05 and big)
    keep = any(res[m]["passes"] for m in METRICS)
    decision = (
        "KEEP shedding in the v3 action space"
        if keep
        else "REMOVE the shed head (no causal effect)"
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "shedding_precheck.json").write_text(
        json.dumps({"results": res, "decision": decision}, indent=2)
    )
    lines = [
        "# Load-shedding pre-check (protocol addendum 2), train split",
        "",
        "Frozen RB-S with its shedding rule (on) vs every shed tier forced to 0 (off).",
        "20 scenarios x 3 seeds, per-scenario means, paired.",
        "",
        "| Metric | On | Off | On - Off [95% CI] | Holm p | Passes |",
        "|---|---|---|---|---|---|",
    ]
    for m in METRICS:
        r = res[m]
        lines.append(
            f"| {m} | {r['on_mean']:.2f} | {r['off_mean']:.2f} | {r['diff_on_minus_off']:+.2f} "
            f"[{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}] | {r['holm_p']:.3f} | {r['passes']} |"
        )
    lines += ["", f"**Decision: {decision}**"]
    (OUT / "shedding_precheck.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--cond", choices=["on", "off"], required=True)
    sub.add_parser("analyse")
    a = ap.parse_args()
    run(a.cond) if a.cmd == "run" else analyse()
