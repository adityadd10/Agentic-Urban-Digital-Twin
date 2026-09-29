#!/usr/bin/env python3
"""Twin-v3 design gates (protocol 2026-09-29_twin_v3_success_criteria §4–§8, addenda 6–7).

  run --policy NAME [--group g1|g2]   run a gate condition on the train split
                                      (g1: seeds k = 0, 1, 2; g2: k = 3, 4, 5, RB-S only)
  calibrate                           noise calibration (RB-S g1 vs g2) -> margins.json
  analyse                             apply the pre-registered pass rules -> gates.{json,md}

Train split only. Unit = scenario (mean over its 3 seeds); every condition uses the
same scenarios and seeds (common random numbers). `calibrate` refuses to overwrite
margins.json, and `analyse` refuses to run without it.
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

SUITE = REPO_ROOT / "data" / "scenarios" / "flood_v3"
RUN_DIR = REPO_ROOT / "runs" / "gates_v3"
OUT = REPO_ROOT / "results" / "twin_v3_gates"
MARGINS = OUT / "margins.json"
EVAL_SEED_STRIDE = 100_000
GROUPS = {"g1": (0, 1, 2), "g2": (3, 4, 5)}
N_BOOT = 10_000
ALPHA = 0.05

DEATHS, UNMET = "patient_deaths", "unmet_patient_hours"
TTA, JOBS, CRIT = (
    "mean_casualty_time_to_admission_hours",
    "jobs_completed",
    "mean_critical_functional_level",
)
LOWER_BETTER = {DEATHS: True, UNMET: True, TTA: True, JOBS: False, CRIT: False}
PRACTICAL_MIN = {
    DEATHS: ("share", 0.05),
    UNMET: ("share", 0.05),
    TTA: ("share", 0.05),
    JOBS: ("share", 0.05),
    CRIT: ("abs", 0.02),
}

# (gate, condition X, condition Y, mechanism metric, outcome metrics, directional?)
# Directional gates require X (RB-S or coordinated) to be the better one.
GATES: list[tuple[str, str, str, str, tuple[str, ...], bool]] = [
    ("A-health", "rbs", "a_health_fixed", UNMET, (DEATHS,), True),
    ("A-power", "rbs", "a_power_fixed", CRIT, (DEATHS, UNMET), True),
    ("A-transport", "rbs", "a_transport_fixed", TTA, (DEATHS, UNMET), True),
    ("B-destination", "rbs", "b_nearest_functioning", TTA, (DEATHS,), False),
    ("C-fleet", "c_calls_first", "c_transfers_first", JOBS, (DEATHS, UNMET), False),
    ("D-repair-order", "rbs", "d_nearest_reachable", CRIT, (UNMET,), False),
    ("E-coordination", "e_coordinated", "a_transport_fixed", DEATHS, (UNMET, CRIT), True),
]
HEADROOM_SHARE = 0.15


# ------------------------------------------------------------------ statistics
def per_scenario(rows: list[dict[str, Any]], metric: str) -> dict[str, float]:
    vals: dict[str, list[float]] = {}
    for r in rows:
        if r.get(metric) is not None:
            vals.setdefault(r["scenario_id"], []).append(float(r[metric]))
    return {k: float(np.mean(v)) for k, v in vals.items()}


def improvement(x: dict[str, float], y: dict[str, float], metric: str) -> np.ndarray:
    """Per-scenario improvement of X over Y (positive = X better); shared scenarios only."""
    ids = sorted(set(x) & set(y))
    sign = 1.0 if LOWER_BETTER[metric] else -1.0
    return np.array([sign * (y[i] - x[i]) for i in ids])


def bootstrap_ci(d: np.ndarray, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), size=(N_BOOT, len(d)))].mean(axis=1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def wilcoxon_p(d: np.ndarray) -> float:
    return 1.0 if len(d) == 0 or np.allclose(d, 0) else float(wilcoxon(d).pvalue)


def holm(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, pvalues[i] * (len(pvalues) - rank)))
        adjusted[i] = running
    return adjusted


def noise_level(g1: list[dict[str, Any]], g2: list[dict[str, Any]], metric: str) -> float:
    """97.5th percentile of |bootstrap mean| of per-scenario (G1 - G2) differences."""
    a, b = per_scenario(g1, metric), per_scenario(g2, metric)
    ids = sorted(set(a) & set(b))
    d = np.array([a[i] - b[i] for i in ids])
    rng = np.random.default_rng(0)
    boot = d[rng.integers(0, len(d), size=(N_BOOT, len(d)))].mean(axis=1)
    return float(np.percentile(np.abs(boot), 97.5))


def margin_for(metric: str, noise: float, rbs_mean: float) -> float:
    kind, value = PRACTICAL_MIN[metric]
    practical = value * abs(rbs_mean) if kind == "share" else value
    return max(2.0 * noise, practical)


def evaluate_gate(
    x_rows: list[dict[str, Any]],
    y_rows: list[dict[str, Any]],
    mechanism: str,
    outcomes: tuple[str, ...],
    directional: bool,
    margins: dict[str, float],
) -> dict[str, Any]:
    """Protocol §6 + addendum 7. Improvements are 'X better than Y' in each metric's units."""
    d = improvement(per_scenario(x_rows, mechanism), per_scenario(y_rows, mechanism), mechanism)
    lo, hi = bootstrap_ci(d)
    m = margins[mechanism]
    if directional:
        mech_pass, better = lo > m, "X"
    elif lo > m:
        mech_pass, better = True, "X"
    elif hi < -m:
        mech_pass, better = True, "Y"
    else:
        mech_pass, better = False, None
    mech = {
        "metric": mechanism,
        "n": len(d),
        "mean_improvement_X_over_Y": float(d.mean()) if len(d) else None,
        "ci95": [lo, hi],
        "margin": m,
        "passes": bool(mech_pass),
        "better": better,
    }
    outs = []
    for metric in outcomes:
        od = improvement(per_scenario(x_rows, metric), per_scenario(y_rows, metric), metric)
        if better == "Y":
            od = -od
        olo, ohi = bootstrap_ci(od)
        outs.append(
            {
                "metric": metric,
                "n": len(od),
                "mean_improvement_of_better": float(od.mean()) if len(od) else None,
                "ci95": [olo, ohi],
                "wilcoxon_p": wilcoxon_p(od),
            }
        )
    for o, adj in zip(outs, holm([o["wilcoxon_p"] for o in outs]), strict=True):
        o["holm_p"] = adj
        o["passes"] = bool(better is not None and o["ci95"][0] > 0 and adj < ALPHA)
    outcome_pass = any(o["passes"] for o in outs)
    return {"mechanism": mech, "outcomes": outs, "passes": bool(mech_pass and outcome_pass)}


def evaluate_headroom(rbs: list[dict[str, Any]], ceiling: list[dict[str, Any]]) -> dict[str, Any]:
    a = per_scenario(rbs, DEATHS)
    d = improvement(per_scenario(ceiling, DEATHS), a, DEATHS)  # ceiling over RB-S
    lo, hi = bootstrap_ci(d)
    rbs_mean = float(np.mean([a[i] for i in a]))
    share = float(d.mean()) / rbs_mean if rbs_mean else 0.0
    return {
        "rbs_mean_deaths": rbs_mean,
        "mean_improvement": float(d.mean()),
        "improvement_share": share,
        "ci95": [lo, hi],
        "threshold_share": HEADROOM_SHARE,
        "passes": bool(share >= HEADROOM_SHARE and lo > 0),
    }


# ------------------------------------------------------------------ run
def run(policy_name: str, group: str) -> None:
    from udt.agents.gate_heuristics_v3 import make_policy
    from udt.envs.multi_env_v3 import UDTMultiAgentEnvV3
    from udt.logging.metrics import compute_episode_metrics

    if group == "g2" and policy_name != "rbs":
        raise SystemExit("g2 is only for RB-S noise calibration")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    out = RUN_DIR / f"{policy_name}_{group}.json"
    if out.exists():
        raise SystemExit(f"{out} exists")
    env = UDTMultiAgentEnvV3(
        processed_dir=REPO_ROOT / "data/processed",
        reward_config=REPO_ROOT / "configs/reward.yaml",
        suite_dir=SUITE,
        scenario_split="train",
    )
    policy = make_policy(policy_name)
    rows = []
    for index, sc in enumerate(env._scenarios):
        for k in GROUPS[group]:
            seed = sc.seed + k * EVAL_SEED_STRIDE
            env.reset(seed=seed, options={"scenario_index": index, "goal": [1, 1, 1, 1]})
            done = False
            while not done:
                _, _, term, trunc, _ = env.step(policy.act(env))
                done = any(term.values()) or any(trunc.values())
            m = compute_episode_metrics(sc.scenario_id, policy_name, env.episode_trace).model_dump()
            rows.append({**m, "eval_seed": seed})
        print(f"{policy_name} {group}: {sc.scenario_id} done", flush=True)
    out.write_text(json.dumps(rows))


def _load(name: str, group: str = "g1") -> list[dict[str, Any]]:
    return json.loads((RUN_DIR / f"{name}_{group}.json").read_text())


def calibrate() -> None:
    if MARGINS.exists():
        raise SystemExit(f"{MARGINS} exists: margins are fixed")
    g1, g2 = _load("rbs", "g1"), _load("rbs", "g2")
    margins, detail = {}, {}
    for metric in LOWER_BETTER:
        vals = per_scenario(g1, metric)
        rbs_mean = float(np.mean(list(vals.values())))
        noise = noise_level(g1, g2, metric)
        margins[metric] = margin_for(metric, noise, rbs_mean)
        detail[metric] = {
            "rbs_mean_g1": rbs_mean,
            "noise": noise,
            "two_noise": 2 * noise,
            "practical_min": PRACTICAL_MIN[metric],
            "margin": margins[metric],
        }
    OUT.mkdir(parents=True, exist_ok=True)
    MARGINS.write_text(json.dumps({"margins": margins, "detail": detail}, indent=2))
    print(json.dumps(detail, indent=2))


def _strata() -> dict[str, str]:
    manifest = json.loads((SUITE / "manifest.json").read_text())
    return {
        f["scenario_id"]: f"{f['sector']}|{f['severity_band']}"
        for f in manifest["scenario_features"]["train"]
    }


def analyse() -> None:
    if not MARGINS.exists():
        raise SystemExit("run calibrate first (margins must be fixed before any gate)")
    margins = json.loads(MARGINS.read_text())["margins"]
    strata = _strata()
    results: dict[str, Any] = {}
    for gate, x, y, mech, outcomes, directional in GATES:
        xr, yr = _load(x), _load(y)
        res = evaluate_gate(xr, yr, mech, outcomes, directional, margins)
        res["conditions"] = {"X": x, "Y": y}
        xs, ys = per_scenario(xr, mech), per_scenario(yr, mech)
        by: dict[str, list[float]] = {}
        for sid in set(xs) & set(ys):
            sign = 1.0 if LOWER_BETTER[mech] else -1.0
            by.setdefault(strata[sid], []).append(sign * (ys[sid] - xs[sid]))
        res["per_stratum_mean_improvement_X_over_Y"] = {
            k: float(np.mean(v)) for k, v in sorted(by.items())
        }
        results[gate] = res
    results["F-headroom"] = evaluate_headroom(_load("rbs"), _load("f_lookahead_ceiling"))
    all_pass = all(r["passes"] for r in results.values())
    (OUT / "gates.json").write_text(
        json.dumps({"results": results, "all_pass": all_pass, "margins": margins}, indent=2)
    )
    lines = [
        "# Twin-v3 design gates (train split)",
        "",
        "Protocol 2026-09-29_twin_v3_success_criteria §6 + addendum 7. "
        "X better than Y is positive.",
        "",
        "| Gate | X vs Y | Mechanism (margin) | Improvement [95% CI] | Mech. "
        "| Outcome check | Gate |",
        "|---|---|---|---|---|---|---|",
    ]
    for gate, x, y, *_ in GATES:
        r = results[gate]
        mech = r["mechanism"]
        outs = "; ".join(
            f"{o['metric']} {o['mean_improvement_of_better']:+.2f} "
            f"[{o['ci95'][0]:+.2f}, {o['ci95'][1]:+.2f}] Holm p={o['holm_p']:.3f}"
            for o in r["outcomes"]
            if o["mean_improvement_of_better"] is not None
        )
        lines.append(
            f"| {gate} | {x} vs {y} | {mech['metric']} ({mech['margin']:.3f}) "
            f"| {mech['mean_improvement_X_over_Y']:+.3f} "
            f"[{mech['ci95'][0]:+.3f}, {mech['ci95'][1]:+.3f}] "
            f"| {'pass' if mech['passes'] else 'fail'} (better: {mech['better']}) "
            f"| {outs} | **{'PASS' if r['passes'] else 'FAIL'}** |"
        )
    f = results["F-headroom"]
    lines += [
        "",
        f"**F-headroom:** RB-S deaths {f['rbs_mean_deaths']:.2f}; ceiling improvement "
        f"{f['mean_improvement']:+.2f} ({100 * f['improvement_share']:.1f}%, "
        f"CI [{f['ci95'][0]:+.2f}, {f['ci95'][1]:+.2f}]); threshold 15% -> "
        f"**{'PASS' if f['passes'] else 'FAIL'}**",
        "",
        f"**All gates pass: {all_pass}**",
    ]
    (OUT / "gates.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--policy", required=True)
    r.add_argument("--group", default="g1", choices=list(GROUPS))
    sub.add_parser("calibrate")
    sub.add_parser("analyse")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.policy, a.group)
    elif a.cmd == "calibrate":
        calibrate()
    else:
        analyse()
