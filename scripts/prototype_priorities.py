#!/usr/bin/env python3
"""Feasibility test 4A (prototype only): changing priorities
(results/twin_v3_gates/prototype_hardq_plan.md). Frozen twin-v3, no acuity.

  run --part i --n-parts k   RB-S and per-objective oracle (30 strategies, perfect foresight)
  report                     -> results/twin_v3_gates/prototype_priorities_results.*
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import diagnostic_oracle_v3 as diag  # noqa: E402
from udt.agents.rbs_v3 import RuleBasedStrongV3  # noqa: E402
from udt.logging.metrics import compute_episode_metrics  # noqa: E402

RUN_DIR = REPO_ROOT / "runs" / "prototype_priorities"
OUT = REPO_ROOT / "results" / "twin_v3_gates"
D0, U0, C0, T0 = 65.42, 375.42, 1 - 0.6438, 2.2506
OBJECTIVES = {
    "O2_restore_infrastructure": {"d": 0.2, "u": 0.2, "c": 1.0, "t": 0.0},
    "O3_fast_response": {"d": 0.2, "u": 0.2, "c": 0.0, "t": 1.0},
}


def objective(m: Any, w: dict[str, float], deaths: float | None = None) -> float:
    """J for metrics `m` (EpisodeMetrics or dict); `deaths` overrides for partial traces."""
    get = (lambda k: getattr(m, k)) if not isinstance(m, dict) else m.get
    d = float(get("patient_deaths") if deaths is None else deaths)
    tta = get("mean_casualty_time_to_admission_hours")
    return (
        w["d"] * d / D0
        + w["u"] * float(get("unmet_patient_hours")) / U0
        + w["c"] * (1.0 - float(get("mean_critical_functional_level"))) / C0
        + w["t"] * (0.0 if tta is None else float(tta)) / T0
    )


def _score(scratch: Any, blob: bytes, cand: Any, rbs: Any, w: dict[str, float]) -> float:
    scratch.load_resume_state(blob)
    start = scratch.sim._patient_deaths_total
    for i in range(diag.HORIZON):
        if not scratch.agents:
            break
        scratch.step((cand if i < diag.CHOICE_EVERY else rbs).act(scratch))
    tr = scratch.episode_trace
    if not tr:
        return 0.0
    m = compute_episode_metrics("x", "x", tr)
    return objective(m, w, deaths=tr[-1].patient_deaths_cumulative - start)


def test_scenarios(env: Any) -> list[int]:
    feats = {
        f["scenario_id"]: f["severity_band"]
        for f in json.loads((diag.SUITE / "manifest.json").read_text())["scenario_features"][
            "train"
        ]
    }
    return [i for i, sc in enumerate(env._scenarios) if feats[sc.scenario_id] != "mild"][:10]


def run(part: int, n_parts: int) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    env, scratch = diag._env(), diag._env()
    params = RuleBasedStrongV3.frozen().params
    rbs = RuleBasedStrongV3(params)
    cands = [
        diag.Composite(h, p, t, params)
        for h, p, t in itertools.product(diag.HEALTH, diag.POWER, diag.TRANSPORT)
    ]
    jobs = [(o, i) for o in OBJECTIVES for i in test_scenarios(env)]
    for j, (oname, index) in enumerate(jobs):
        if j % n_parts != part:
            continue
        sc = env._scenarios[index]
        out = RUN_DIR / f"{oname}__{sc.scenario_id}.json"
        if out.exists():
            continue
        w = OBJECTIVES[oname]
        env.reset(seed=sc.seed, options={"scenario_index": index, "goal": [1, 1, 1, 1]})
        while env.agents:
            env.step(rbs.act(env))
        rbs_m = compute_episode_metrics(sc.scenario_id, "rbs", env.episode_trace).model_dump()
        env.reset(seed=sc.seed, options={"scenario_index": index, "goal": [1, 1, 1, 1]})
        chosen, picks, d = cands[0], [], 0
        while env.agents:
            if d % diag.CHOICE_EVERY == 0:
                blob = diag._snapshot(env)
                scores = [_score(scratch, blob, c, rbs, w) for c in cands]
                chosen = cands[int(np.argmin(scores))]
                picks.append(chosen.name)
            env.step(chosen.act(env))
            d += 1
        ora_m = compute_episode_metrics(sc.scenario_id, "oracle", env.episode_trace).model_dump()
        out.write_text(
            json.dumps(
                {
                    "objective": oname,
                    "rbs": rbs_m,
                    "oracle": ora_m,
                    "picks": picks,
                    "J_rbs": objective(rbs_m, w),
                    "J_oracle": objective(ora_m, w),
                }
            )
        )
        print(f"{oname} {sc.scenario_id} done", flush=True)


def report() -> None:
    lines = [
        "# Feasibility test 4A: changing priorities (prototype only)",
        "",
        "Plan: prototype_hardq_plan.md. 10 moderate/severe train scenarios, seed k = 0. "
        "O1 (patients) from the oracle diagnostic: +1.6%.",
        "",
        "| Objective | RB-S mean J | Oracle mean J | Oracle better by | Reading |",
        "|---|---|---|---|---|",
    ]
    out: dict[str, Any] = {}
    for oname in OBJECTIVES:
        rows = [json.loads(f.read_text()) for f in sorted(RUN_DIR.glob(f"{oname}__*.json"))]
        jr = float(np.mean([r["J_rbs"] for r in rows]))
        jo = float(np.mean([r["J_oracle"] for r in rows]))
        share = (jr - jo) / jr if jr else 0.0
        verdict = "HEADROOM (>= 15%)" if share >= 0.15 else "no meaningful headroom (< 15%)"
        out[oname] = {
            "n": len(rows),
            "J_rbs": jr,
            "J_oracle": jo,
            "share": share,
            "verdict": verdict,
            "rbs_critF": float(np.mean([r["rbs"]["mean_critical_functional_level"] for r in rows])),
            "oracle_critF": float(
                np.mean([r["oracle"]["mean_critical_functional_level"] for r in rows])
            ),
            "rbs_deaths": float(np.mean([r["rbs"]["patient_deaths"] for r in rows])),
            "oracle_deaths": float(np.mean([r["oracle"]["patient_deaths"] for r in rows])),
        }
        lines.append(
            f"| {oname} (n={len(rows)}) | {jr:.3f} | {jo:.3f} | {100 * share:+.1f}% "
            f"| **{verdict}** |"
        )
    lines += [
        "",
        "Detail (means): "
        + "; ".join(
            f"{k}: deaths RB-S {v['rbs_deaths']:.1f} / oracle {v['oracle_deaths']:.1f}, "
            "critical function "
            f"{v['rbs_critF']:.3f} / {v['oracle_critF']:.3f}"
            for k, v in out.items()
        ),
    ]
    (OUT / "prototype_priorities_results.json").write_text(json.dumps(out, indent=2))
    (OUT / "prototype_priorities_results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--part", type=int, required=True)
    r.add_argument("--n-parts", type=int, required=True)
    sub.add_parser("report")
    a = ap.parse_args()
    run(a.part, a.n_parts) if a.cmd == "run" else report()
