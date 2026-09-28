# Full-budget (2M) training — outcome (2026-09-28)

Governed by `results/protocol/2026-09-27_energy_ablation_and_full_budget.md` §7 (and the
2026-09-28 addendum and cascade-artefact note, all committed before the run). Run exactly as
pre-registered: reward A, twin-v2 / `udt_multi_env_v2` / `flood_suite_v2` train split, frozen code
`train-2M-start` (`d24ae43`, clean, read-only worktree), Apple M5. MAPPO seeds 0–4 × 2,000,128
steps, PPO seeds 0–2 × 2,000,896 steps, final checkpoints, no interruptions or resumes. Models
and checksums are in `manifest.json` and `models/`. **No deviations.**

Evaluation: frozen test split, 10 scenarios × 3 evaluation seeds, per-scenario means, 95%
bootstrap CI, two-sided Wilcoxon, α = 0.05 (`vs_rule_based_*`, `vs_baseline_300k.*`).

## 1. Against rule-based (primary metrics)

| | Rule-based | PPO 2M | MAPPO 2M |
|---|---|---|---|
| Death proxy | 43.57 | 49.81 (+6.24 [−4.67, +20.02], p = 0.85) | **43.43** (−0.14 [−12.73, +15.18], p = 0.43) |
| Unmet patient-hours | 272.79 | 285.71 (+12.92, p = 0.92) | **271.74** (−1.06, p = 0.77) |

**No learned policy is significantly better than rule-based on any metric.** MAPPO is
indistinguishable from rule-based on both primary metrics. Secondary: both learned policies are
significantly worse at ambulance work (requests completed: PPO 0.30, MAPPO 4.47, rule-based 9.20);
MAPPO sheds far more load (50.8 vs 2.7 MWh); PPO has lower hospital function (0.64 vs 0.70,
p = 0.027).

## 2. Against baseline-300k (effect of 6.7× more training)

| Primary metric | PPO 300k → 2M | MAPPO 300k → 2M |
|---|---|---|
| Death proxy | 50.46 → 49.81 (−0.64 [−1.96, +0.78], p = 0.15) | 43.39 → 43.43 (+0.03 [−1.88, +1.93], p = 0.94) |
| Unmet patient-hours | 290.2 → 285.7 (−4.48 [−9.63, +0.59], p = 0.049) | 278.4 → 271.7 (−6.61 [−13.33, −0.80], p = 0.13) |

**More training barely changed patient outcomes.** The largest primary change, PPO unmet
patient-hours, is borderline (p = 0.049, CI includes 0), and 20 comparisons at α = 0.05 were run
with no correction. Secondary changes are larger and in a consistent direction: ambulance
requests completed **fell** (PPO 2.50 → 0.30, p = 0.004; MAPPO 7.48 → 4.47, p = 0.002), and MAPPO's
load shedding **rose** (39.8 → 50.8 MWh, p = 0.049), while episode reward edged up.

## 3. Conclusion for RQ1/H1

At 2M steps, still at the lower end of the planned 2–5M budget, **H1 (MARL outperforms rule-based)
is not supported.** MAPPO matches rule-based on patient outcomes but not better; the extra
training did not move primary outcomes. The honest reading is no longer simply "under-trained":
more training made the policies *better at the reward* but not better for patients, which points
to the reward, below.

## 4. Post-hoc finding (hypothesis, not a pre-registered result)

Found **after** seeing these test results, so it's a hypothesis to test, not a conclusion.

**The reward penalises answered emergency calls and ignores unanswered ones.** In
`envs/reward.py` the ambulance term adds a response-time penalty only when a casualty is picked
up. Unanswered calls never expire and never enter the reward or the death proxy (they are only
counted in the reported `requests_pending_at_end`). A delivered casualty also joins a hospital
queue, which increases unmet patient-hours and the death proxy. Not dispatching is therefore
reward-optimal for the transport agent, and dispatch collapsing with more training is what that
predicts. Together with the cascade-term artefact (note of 2026-09-28) and the energy floor, this
is the third reward-design issue that learned policies respond to.

**Next step (to pre-register before running):** a reward ablation on the validation split that
costs unanswered calls (e.g. waiting time accrues for pending requests, or an expiry that counts
toward the death proxy), with the prediction that dispatch recovers and the decision rule fixed in
advance. The robustness study (pre-registered 2026-09-28) runs as planned regardless.
